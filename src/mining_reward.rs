//! Mining Efficiency Dopamine Reward Framework
//!
//! Computes a multi-dimensional reward signal from hardware telemetry,
//! treating mining efficiency as a biological survival signal for the SNN.
//!
//! Core equation: R = α·MiningEfficiency − β·ThermalStress − γ·EnergyWaste
//!
//! The output `mining_dopamine` is EMA-smoothed and fed into the STDP
//! learning rate blend in `engine.rs`, gating synaptic plasticity based
//! on whether the system is operating within its homeostatic envelope.

use crate::telemetry::GpuTelemetry;

// ── Reward component weights ─────────────────────────────────────────────────

/// Weight for the positive mining-efficiency term.
const ALPHA_EFFICIENCY: f32 = 0.6;

/// Weight for the thermal/power stress penalty.
const BETA_THERMAL: f32 = 0.3;

/// Weight for the energy-waste penalty.
const GAMMA_WASTE: f32 = 0.1;

// ── EMA smoothing ────────────────────────────────────────────────────────────

/// EMA blending factor for the instant reward (0.1 = 10-tick window ≈ 1 s).
const EMA_ALPHA: f32 = 0.1;

/// Hard clamp on the final EMA-smoothed reward.  Prevents saturation so that
/// the STDP blend always retains some influence from the event-driven dopamine.
const REWARD_CLAMP: f32 = 0.8;

// ── Thermal / power thresholds ───────────────────────────────────────────────

/// GPU temperature at which the thermal penalty kicks in (°C).
/// Matches NVIDIA's thermal throttle onset for the RTX 5080.
const GPU_THERMAL_PENALTY_ONSET: f32 = 85.0;

/// GPU thermal penalty divisor — maps [onset, onset+10] → [0, 1].
const GPU_THERMAL_DIVISOR: f32 = 10.0;

/// CPU temperature at which the thermal penalty kicks in (°C).
/// Zen 5 (Ryzen 9 9950X) is designed to boost aggressively; Tctl regularly
/// sits at 75–80 °C under sustained all-core load.  Penalising below 85 °C
/// would put the SNN in a state of "chronic pain" during normal syncs.
const CPU_THERMAL_PENALTY_ONSET: f32 = 85.0;

/// CPU thermal penalty divisor — maps [onset, onset+10] → [0, 1].
const CPU_THERMAL_DIVISOR: f32 = 10.0;

/// GPU power draw at which the power-excess penalty starts (W).
const POWER_PENALTY_ONSET: f32 = 400.0;

/// Power penalty divisor — maps [onset, onset+50] → [0, 1].
const POWER_PENALTY_DIVISOR: f32 = 50.0;

// ── Efficiency normalisation ─────────────────────────────────────────────────

/// Target hashrate (MH/s) used to normalise the hashrate stability delta.
/// 0.015 MH/s ≈ RTX 5080 Dynex peak (slightly above the 0.0105 midpoint
/// used for the base dopamine signal, giving headroom before saturation).
const TARGET_HASHRATE_MH: f32 = 0.015;

/// Target efficiency (MH/s per watt) = TARGET_HASHRATE / OPTIMAL_POWER.
const TARGET_EFFICIENCY: f32 = TARGET_HASHRATE_MH / 350.0;

/// Nominal RTX 5080 boost clock (MHz).  Used to detect thermal throttling.
const NOMINAL_CLOCK_MHZ: f32 = 2640.0;

/// Clock floor below which we consider the GPU severely throttled.
const THROTTLE_CLOCK_FLOOR: f32 = 2000.0;

/// Throttle clock range for proportional penalty.
const THROTTLE_CLOCK_RANGE: f32 = NOMINAL_CLOCK_MHZ - THROTTLE_CLOCK_FLOOR;

// ── Q8.8 helpers ─────────────────────────────────────────────────────────────

/// Maximum representable value in unsigned Q8.8 (0xFF.FF = 255 + 255/256).
const Q8_8_MAX: u16 = 0xFFFF;

/// Convert a `[0.0, 1.0]` reward to Q8.8 fixed-point (unsigned).
///
/// Clamps to `[0.0, 1.0]` before conversion and caps the result at
/// `Q8_8_MAX` to prevent bit-overflow during reward spikes.
#[inline]
pub fn reward_to_q8_8(reward: f32) -> u16 {
    let clamped = reward.clamp(0.0, 1.0);
    let raw = (clamped * 256.0) as u32; // u32 intermediate prevents u16 overflow
    (raw.min(Q8_8_MAX as u32)) as u16
}

// ── Homeostatic setpoints ────────────────────────────────────────────────────

/// Optimal operating-point targets for the RTX 5080 + Ryzen 9 9950X system.
#[derive(Debug, Clone, Copy)]
pub struct ThermalSetpoint {
    /// GPU junction temperature sweet-spot (°C).
    pub optimal_gpu_temp_c: f32,
    /// CPU Tctl sweet-spot (°C).  Zen 5 runs hotter by design.
    pub optimal_cpu_temp_c: f32,
    /// Target board power (W) — below TDP but above idle.
    pub optimal_power_w: f32,
    /// Temperature tolerance band (°C) — beyond optimal ± tolerance the
    /// homeostatic reward goes negative.
    pub temp_tolerance_c: f32,
    /// Power tolerance band (W).
    pub power_tolerance_w: f32,
}

impl Default for ThermalSetpoint {
    fn default() -> Self {
        Self {
            optimal_gpu_temp_c: 75.0,
            optimal_cpu_temp_c: 70.0, // Zen 5 — designed to boost high
            optimal_power_w: 350.0,
            temp_tolerance_c: 15.0,
            power_tolerance_w: 80.0,
        }
    }
}

// ── Reward state machine ─────────────────────────────────────────────────────

/// EMA-smoothed mining-efficiency reward computer.
///
/// All state is in fixed-size scalar fields.  `compute()` performs only
/// stack arithmetic — **zero heap allocation** on the hot path.
#[derive(Debug, Clone)]
pub struct MiningRewardState {
    /// Homeostatic operating-point targets.
    pub setpoint: ThermalSetpoint,
    /// Exponential moving average of the composite reward.
    ema_reward: f32,
    /// Previous-tick normalised hashrate (for stability delta).
    prev_hashrate_norm: f32,
    /// Ticks where the GPU was NOT throttled (clock ≥ THROTTLE_CLOCK_FLOOR).
    uptime_ticks: u64,
    /// Total ticks observed (denominator for uptime ratio).
    total_ticks: u64,
}

impl Default for MiningRewardState {
    fn default() -> Self {
        Self::new()
    }
}

impl MiningRewardState {
    pub fn new() -> Self {
        Self {
            setpoint: ThermalSetpoint::default(),
            ema_reward: 0.0,
            prev_hashrate_norm: 0.0,
            uptime_ticks: 0,
            total_ticks: 0,
        }
    }

    /// Compute the mining-efficiency dopamine signal for this tick.
    ///
    /// Accepts the current GPU telemetry and an optional `SystemTelemetry`
    /// reference for CPU thermal data.  Returns the EMA-smoothed reward
    /// in `[-0.8, 0.8]`, suitable for direct assignment to
    /// `NeuroModulators.mining_dopamine`.
    ///
    /// # Hot-path guarantee
    ///
    /// This function touches only stack temporaries and `self` scalars.
    /// No `Vec`, `String`, `Box`, or any heap allocation.
    pub fn compute(
        &mut self,
        telem: &GpuTelemetry,
        cpu_temp_c: Option<f32>,
    ) -> f32 {
        // ── bookkeeping ──────────────────────────────────────────────
        self.total_ticks += 1;
        if telem.gpu_clock_mhz >= THROTTLE_CLOCK_FLOOR {
            self.uptime_ticks += 1;
        }

        // ── 1. Mining Efficiency (positive term) ─────────────────────
        let hashrate_norm =
            (telem.hashrate_mh / TARGET_HASHRATE_MH).clamp(0.0, 1.0);
        let hashrate_stability =
            1.0 - (hashrate_norm - self.prev_hashrate_norm).abs();
        self.prev_hashrate_norm = hashrate_norm;

        let hash_per_watt = (telem.hashrate_mh / telem.power_w.max(1.0))
            / TARGET_EFFICIENCY;
        let hash_per_watt_clamped = hash_per_watt.clamp(0.0, 1.0);

        let uptime_ratio = if self.total_ticks > 0 {
            self.uptime_ticks as f32 / self.total_ticks as f32
        } else {
            1.0
        };

        let mining_efficiency =
            hashrate_stability * hash_per_watt_clamped * uptime_ratio;

        // ── 2. Thermal Stress (negative term) ────────────────────────
        let gpu_thermal = ((telem.gpu_temp_c - GPU_THERMAL_PENALTY_ONSET)
            / GPU_THERMAL_DIVISOR)
            .clamp(0.0, 1.0);

        let power_excess = ((telem.power_w - POWER_PENALTY_ONSET)
            / POWER_PENALTY_DIVISOR)
            .clamp(0.0, 1.0);

        let cpu_thermal = cpu_temp_c
            .map(|t| {
                ((t - CPU_THERMAL_PENALTY_ONSET) / CPU_THERMAL_DIVISOR)
                    .clamp(0.0, 1.0)
            })
            .unwrap_or(0.0);

        // Worst-case drives the penalty — we want ANY thermal breach to
        // suppress learning, not average them away.
        let thermal_stress =
            gpu_thermal.max(power_excess).max(cpu_thermal);

        // ── 3. Energy Waste (negative term) ──────────────────────────
        let throttle_penalty = if telem.gpu_clock_mhz < NOMINAL_CLOCK_MHZ {
            ((NOMINAL_CLOCK_MHZ - telem.gpu_clock_mhz) / THROTTLE_CLOCK_RANGE)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        let power_inefficiency = (1.0 - hash_per_watt_clamped).clamp(0.0, 1.0);

        let energy_waste =
            throttle_penalty * 0.7 + power_inefficiency * 0.3;

        // ── Composite ────────────────────────────────────────────────
        let raw_reward = ALPHA_EFFICIENCY * mining_efficiency
            - BETA_THERMAL * thermal_stress
            - GAMMA_WASTE * energy_waste;

        // ── EMA smoothing ────────────────────────────────────────────
        self.ema_reward =
            (1.0 - EMA_ALPHA) * self.ema_reward + EMA_ALPHA * raw_reward;

        self.ema_reward.clamp(-REWARD_CLAMP, REWARD_CLAMP)
    }

    /// Bell-curve homeostatic reward.
    ///
    /// Returns 1.0 at `optimal`, decays quadratically toward 0.0 at
    /// `optimal ± tolerance`, and goes negative beyond.
    #[inline]
    pub fn homeostatic_reward(
        value: f32,
        optimal: f32,
        tolerance: f32,
    ) -> f32 {
        let deviation = ((value - optimal) / tolerance).powi(2);
        (1.0 - deviation).clamp(-0.5, 1.0)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a `GpuTelemetry` with the given overrides on top of
    /// "healthy RTX 5080" defaults.
    fn telem(
        hashrate_mh: f32,
        power_w: f32,
        gpu_temp_c: f32,
        gpu_clock_mhz: f32,
    ) -> GpuTelemetry {
        GpuTelemetry {
            hashrate_mh,
            power_w,
            gpu_temp_c,
            gpu_clock_mhz,
            vddcr_gfx_v: 1.0,
            fan_speed_pct: 60.0,
            mem_clock_mhz: 2400.0,
            ..Default::default()
        }
    }

    #[test]
    fn optimal_conditions_positive_reward() {
        let mut state = MiningRewardState::new();
        // Prime the EMA with one tick so prev_hashrate_norm is set.
        let t = telem(0.012, 340.0, 72.0, 2640.0);
        state.compute(&t, Some(68.0));

        // Second tick at same conditions — stability = 1.0.
        let r = state.compute(&t, Some(68.0));
        assert!(r > 0.0, "optimal conditions should yield positive reward, got {r}");
    }

    #[test]
    fn thermal_stress_negative() {
        let mut state = MiningRewardState::new();
        // Thermal emergency: 95°C GPU + throttled clock + excess power.
        // When the GPU is thermally throttling, hashrate drops AND temp is high.
        let t = telem(0.003, 440.0, 95.0, 1900.0);
        // Warm up EMA.
        for _ in 0..20 {
            state.compute(&t, Some(68.0));
        }
        let r = state.compute(&t, Some(68.0));
        assert!(r < 0.0, "thermal emergency should produce negative reward, got {r}");
    }

    #[test]
    fn power_excess_penalty() {
        let mut state = MiningRewardState::new();
        let t = telem(0.012, 450.0, 72.0, 2640.0); // 450W
        for _ in 0..20 {
            state.compute(&t, Some(68.0));
        }
        let r_high_power = state.compute(&t, Some(68.0));

        let mut state2 = MiningRewardState::new();
        let t2 = telem(0.012, 300.0, 72.0, 2640.0); // 300W
        for _ in 0..20 {
            state2.compute(&t2, Some(68.0));
        }
        let r_low_power = state2.compute(&t2, Some(68.0));

        assert!(
            r_low_power > r_high_power,
            "300W should reward higher than 450W: {r_low_power} vs {r_high_power}"
        );
    }

    #[test]
    fn ema_smoothing_dampens_spikes() {
        let mut state = MiningRewardState::new();
        let good = telem(0.012, 340.0, 72.0, 2640.0);
        // Build up positive EMA.
        for _ in 0..50 {
            state.compute(&good, Some(68.0));
        }
        let before = state.ema_reward;

        // Sudden bad tick.
        let bad = telem(0.001, 450.0, 95.0, 1800.0);
        let after = state.compute(&bad, Some(92.0));

        // EMA should not crash all the way to the raw bad value.
        assert!(
            after > -0.5,
            "EMA should dampen a single bad tick, got {after}"
        );
        assert!(
            after < before,
            "bad tick should still pull EMA down: {before} -> {after}"
        );
    }

    #[test]
    fn zero_hashrate_near_zero_efficiency() {
        let mut state = MiningRewardState::new();
        let t = telem(0.0, 150.0, 55.0, 2640.0); // idle GPU
        for _ in 0..20 {
            state.compute(&t, None);
        }
        let r = state.compute(&t, None);
        // Should be slightly negative (energy waste from power_inefficiency)
        // but not a severe penalty.
        assert!(
            r > -0.3,
            "zero hashrate at idle power should not be severely penalised, got {r}"
        );
    }

    #[test]
    fn cpu_thermal_zen5_no_chronic_pain() {
        let mut state = MiningRewardState::new();
        let t = telem(0.012, 340.0, 72.0, 2640.0);
        // Zen 5 at 80°C — normal all-core boost.  Should NOT trigger penalty.
        for _ in 0..20 {
            state.compute(&t, Some(80.0));
        }
        let r = state.compute(&t, Some(80.0));
        assert!(
            r > 0.0,
            "Zen 5 at 80°C should not drag reward negative, got {r}"
        );
    }

    #[test]
    fn q8_8_conversion_roundtrip() {
        assert_eq!(reward_to_q8_8(0.0), 0);
        assert_eq!(reward_to_q8_8(1.0), 256);
        assert_eq!(reward_to_q8_8(0.5), 128);
        // Overflow protection.
        assert_eq!(reward_to_q8_8(1.5), 256); // clamped to 1.0
        assert_eq!(reward_to_q8_8(-0.5), 0);  // clamped to 0.0
    }

    #[test]
    fn homeostatic_bell_curve() {
        // At optimal → 1.0.
        let r = MiningRewardState::homeostatic_reward(75.0, 75.0, 15.0);
        assert!((r - 1.0).abs() < 1e-6, "at optimal: {r}");

        // At tolerance boundary → 0.0.
        let r = MiningRewardState::homeostatic_reward(90.0, 75.0, 15.0);
        assert!(r.abs() < 1e-6, "at tolerance boundary: {r}");

        // Beyond tolerance → negative.
        let r = MiningRewardState::homeostatic_reward(100.0, 75.0, 15.0);
        assert!(r < 0.0, "beyond tolerance should be negative: {r}");
    }

    #[test]
    fn reward_clamp_prevents_saturation() {
        let mut state = MiningRewardState::new();
        let perfect = telem(0.015, 300.0, 65.0, 2700.0);
        for _ in 0..500 {
            state.compute(&perfect, Some(60.0));
        }
        assert!(
            state.ema_reward <= REWARD_CLAMP,
            "EMA should never exceed clamp: {}",
            state.ema_reward
        );
        assert!(
            state.ema_reward >= -REWARD_CLAMP,
            "EMA should never go below -clamp: {}",
            state.ema_reward
        );
    }
}
