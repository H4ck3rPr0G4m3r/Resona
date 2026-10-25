//! RNNoise (nnnoiseless) noise suppression, plus the voice-activity tracker that drives the
//! output gate. Both work on 480-sample (10 ms) frames at 48 kHz.

use nnnoiseless::DenoiseState;

use super::{Effect, Smooth};

pub const FRAME: usize = DenoiseState::FRAME_SIZE; // 480
const SCALE: f32 = 32768.0; // RNNoise works on 16-bit-range floats

pub struct Denoise {
    st: Box<DenoiseState<'static>>,
    inbuf: [f32; FRAME],
    raw: [f32; FRAME],  // the previous frame, unprocessed (dry, latency-matched)
    den: [f32; FRAME],  // the previous frame, denoised
    k: usize,
    amount: Smooth,
    pub last_vad: f32,
}

impl Denoise {
    pub fn new() -> Self {
        Self { st: DenoiseState::new(), inbuf: [0.0; FRAME], raw: [0.0; FRAME], den: [0.0; FRAME], k: 0, amount: Smooth::new(1.0), last_vad: 0.0 }
    }
}

impl Effect for Denoise {
    fn set(&mut self, _idx: usize, v: f32) {
        self.amount.target = v;
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            self.inbuf[self.k] = *s * SCALE;
            let a = self.amount.next();
            let r = self.raw[self.k];
            *s = (r + (self.den[self.k] - r) * a) / SCALE;
            self.k += 1;
            if self.k == FRAME {
                self.k = 0;
                self.raw = self.inbuf;
                self.last_vad = self.st.process_frame(&mut self.den, &self.inbuf);
            }
        }
    }
    fn reset(&mut self) {
        self.st = DenoiseState::new();
        self.inbuf = [0.0; FRAME];
        self.raw = [0.0; FRAME];
        self.den = [0.0; FRAME];
        self.k = 0;
        self.amount.snap();
    }
    fn latency(&self) -> usize {
        FRAME
    }
}

/// Voice-activity probability per 10 ms frame, on the capture timeline.
pub struct VadTracker {
    st: Box<DenoiseState<'static>>,
    frame: [f32; FRAME],
    scratch: [f32; FRAME],
    k: usize,
}

impl VadTracker {
    pub fn new() -> Self {
        Self { st: DenoiseState::new(), frame: [0.0; FRAME], scratch: [0.0; FRAME], k: 0 }
    }

    /// Feed samples; `on_frame(probability)` is called once per completed 10 ms frame.
    pub fn feed(&mut self, x: &[f32], mut on_frame: impl FnMut(f32)) {
        for &s in x {
            self.frame[self.k] = s * SCALE;
            self.k += 1;
            if self.k == FRAME {
                self.k = 0;
                on_frame(self.st.process_frame(&mut self.scratch, &self.frame));
            }
        }
    }
}
