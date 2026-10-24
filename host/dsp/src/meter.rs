//! Peak/RMS level meter with ballistic decay, read from another thread via atomics.

use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Default)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    pub fn store(&self, v: f32) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}

/// Written on the audio thread, read by the UI. Values are linear amplitudes.
#[derive(Default)]
pub struct Meter {
    pub peak: AtomicF32,
    pub rms: AtomicF32,
}

/// Audio-thread state for a [`Meter`].
pub struct MeterState {
    peak: f32,
    ms: f32,
    peak_decay: f32,
    ms_coef: f32,
}

impl MeterState {
    /// `sr` sample rate; peak falls ~20 dB/s, RMS integrates over ~300 ms.
    pub fn new(sr: f32) -> Self {
        Self { peak: 0.0, ms: 0.0, peak_decay: 10f32.powf(-1.0 / sr), ms_coef: 1.0 - (-1.0 / (0.3 * sr)).exp() }
    }

    pub fn process(&mut self, x: &[f32], out: &Meter) {
        for &v in x {
            let a = v.abs();
            self.peak = if a > self.peak { a } else { self.peak * self.peak_decay };
            self.ms += self.ms_coef * (v * v - self.ms);
        }
        out.peak.store(self.peak);
        out.rms.store(self.ms.sqrt());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_rms() {
        let m = Meter::default();
        let mut s = MeterState::new(48000.0);
        let x: Vec<f32> = (0..96000).map(|i| (i as f32 * 0.1).sin()).collect();
        s.process(&x, &m);
        assert!((m.rms.load() - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.02);
        assert!(m.peak.load() > 0.99);
    }
}
