//! Hardware Telemetry Types
//!
//! Lightweight telemetry structs consumed by the reward and modulator modules.
//! These mirror the fields from Eagle-Lander's GpuTelemetry that are relevant
//! to reward computation and neuromodulation.

use serde::{Deserialize, Serialize};

/// Real-time hardware readings from the GPU/mining system.
///
/// Populate this struct from your hardware monitoring layer (NVML, sysfs, etc.)
/// and pass it to `MiningRewardState::compute()` or `NeuroModulators::from_telemetry()`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuTelemetry {
    /// GPU core voltage in Volts (e.g., ~0.7V idle, ~1.05V load).
    pub vddcr_gfx_v: f32,
    /// GPU junction temperature in Celsius.
    pub gpu_temp_c: f32,
    /// Mining hashrate in MH/s.
    pub hashrate_mh: f32,
    /// Total board power draw in Watts.
    pub power_w: f32,
    /// GPU core clock in MHz.
    pub gpu_clock_mhz: f32,
    /// Memory clock in MHz.
    pub mem_clock_mhz: f32,
    /// Fan speed percentage (0-100).
    pub fan_speed_pct: f32,
    /// Ocean Predictoor signal (0.0 = no data, 0.0-1.0 = prediction confidence).
    #[serde(default)]
    pub ocean_intel: f32,
}

/// Events emitted by the mining pool client.
///
/// These drive phasic dopamine/cortisol spikes in the neuromodulator system.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// A submitted share was accepted by the pool.
    ShareAccepted { latency_ms: u64 },
    /// A block was found (rare — strongest dopamine burst).
    BlockFound { block_height: u64, reward_dnx: f64 },
    /// Pool connection switched (cortisol spike).
    PoolSwitch { reason: String },
    /// A share was rejected (mild cortisol).
    ShareRejected { reason: String },
}
