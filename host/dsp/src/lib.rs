//! Resona host DSP: resampling, SOLA, metering, and (from M5) the effect chains.

pub mod meter;
pub mod resample;
pub mod sola;

pub use meter::{AtomicF32, Meter, MeterState};
pub use resample::Resampler;
pub use sola::Sola;

/// Linear amplitude → dBFS (floored at -120).
pub fn to_db(x: f32) -> f32 {
    20.0 * x.max(1e-6).log10()
}

/// dB → linear amplitude.
pub fn from_db(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}
