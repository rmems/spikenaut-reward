//! # spikenaut-reward
//!
//! Homeostatic reward computation for cyber-physical systems in the Spikenaut ecosystem.
//!
//! This crate provides:
//! - **MiningRewardState** — EMA-smoothed multi-dimensional reward from hardware telemetry
//!   (R = alpha*efficiency - beta*thermal - gamma*waste)
//! - **NeuroModulators** — 7-system neuromodulator bank (dopamine, cortisol, acetylcholine,
//!   tempo, fpga_stress, market_volatility, mining_dopamine)
//! - **ThermalSetpoint** — Bell-curve homeostatic reward function
//! - **Q8.8 helpers** — Fixed-point reward export for FPGA deployment
//!
//! Complements the `neuromod` crate (v0.2.1) which provides the SNN neurons and STDP learning.
//!
//! ## Provenance
//!
//! Extracted from Eagle-Lander, the author's own private neuromorphic GPU supervisor
//! repository (closed-source). The reward and neuromodulator system ran in production
//! driving a 16-neuron LIF SNN for Dynex/Quai/Qubic mining optimization before being
//! open-sourced as a standalone crate.
//!
//! ## Quick Start
//!
//! ```rust
//! use spikenaut_reward::{MiningRewardState, GpuTelemetry};
//!
//! let mut state = MiningRewardState::new();
//! let telem = GpuTelemetry {
//!     hashrate_mh: 0.012,
//!     power_w: 340.0,
//!     gpu_temp_c: 72.0,
//!     gpu_clock_mhz: 2640.0,
//!     vddcr_gfx_v: 1.0,
//!     ..Default::default()
//! };
//! let reward = state.compute(&telem, Some(68.0));
//! println!("Mining dopamine: {reward:.4}");
//! ```

pub mod telemetry;
pub mod mining_reward;
pub mod modulators;

// Re-export public API
pub use telemetry::{GpuTelemetry, PoolEvent};
pub use mining_reward::{MiningRewardState, ThermalSetpoint, reward_to_q8_8};
pub use modulators::NeuroModulators;
