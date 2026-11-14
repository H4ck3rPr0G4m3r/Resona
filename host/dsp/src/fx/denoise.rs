//! RNNoise (nnnoiseless) noise suppression, plus the voice-activity tracker that drives the
//! output gate. Both work on 480-sample (10 ms) frames at 48 kHz.

use nnnoiseless::DenoiseState;

use super::{Effect, Smooth};

pub const FRAME: usize = DenoiseState::FRAME_SIZE; // 480
const SCALE: f32 = 32768.0; // RNNoise works on 16-bit-range floats

/// RNNoise's own delay: its overlap-add output for frame n only completes with frame n + 1.
/// On top of that a frame can only be processed once it is complete. A caller that hands over
/// whole frames (every effect chain does) gets the result back in the same call, so the total
/// stays at one frame; a caller with arbitrary chunk sizes needs a second frame of slack.
pub const LATENCY_FRAMED: usize = FRAME;
const LATENCY_ANY: usize = 2 * FRAME;

pub struct Denoise {
    st: Box<DenoiseState<'static>>,
    inbuf: [f32; FRAME], // the frame being collected
    prev: [f32; FRAME],  // the frame before it, unprocessed: the dry signal RNNoise's output lines up with
    den: [f32; FRAME],
    k: usize,
    /// Finished samples waiting to leave (a ring: `head` is the oldest, `len` are queued).
    out: [f32; 3 * FRAME],
    head: usize,
    len: usize,
    /// Zeros queued at reset: 0 for whole-frame callers, one frame otherwise.
    lead: usize,
    amount: Smooth,
    pub last_vad: f32,
}

impl Denoise {
    /// For any chunk size (20 ms latency).
    pub fn new() -> Self {
        Self::with_lead(FRAME)
    }

    /// For callers that always pass whole 480-sample frames, starting on a frame boundary
    /// (10 ms latency). Other chunk sizes run the queue dry, which comes out as silence.
    pub fn framed() -> Self {
        Self::with_lead(0)
    }

    fn with_lead(lead: usize) -> Self {
        Self {
            st: DenoiseState::new(),
            inbuf: [0.0; FRAME],
            prev: [0.0; FRAME],
            den: [0.0; FRAME],
            k: 0,
            out: [0.0; 3 * FRAME],
            head: 0,
            len: lead,
            lead,
            amount: Smooth::new(1.0),
            last_vad: 0.0,
        }
    }

    fn ingest(&mut self, x: &[f32]) {
        for &s in x {
            self.inbuf[self.k] = s * SCALE;
            self.k += 1;
            if self.k < FRAME {
                continue;
            }
            self.k = 0;
            // Output: the denoised version of `prev` (one frame behind the input).
            self.last_vad = self.st.process_frame(&mut self.den, &self.inbuf);
            for i in 0..FRAME {
                let a = self.amount.next();
                let r = self.prev[i];
                let tail = (self.head + self.len) % self.out.len();
                self.out[tail] = (r + (self.den[i] - r) * a) / SCALE;
                self.len += 1;
            }
            self.prev = self.inbuf;
        }
    }
}

impl Effect for Denoise {
    fn set(&mut self, _idx: usize, v: f32) {
        self.amount.target = v;
    }
    fn process(&mut self, x: &mut [f32]) {
        // At most one frame per pass, so the queue never holds more than three.
        for chunk in x.chunks_mut(FRAME) {
            self.ingest(chunk);
            for s in chunk.iter_mut() {
                if self.len == 0 {
                    *s = 0.0; // a framed instance driven off the frame grid
                    continue;
                }
                *s = self.out[self.head];
                self.head = (self.head + 1) % self.out.len();
                self.len -= 1;
            }
        }
    }
    fn reset(&mut self) {
        self.st = DenoiseState::new();
        self.inbuf = [0.0; FRAME];
        self.prev = [0.0; FRAME];
        self.den = [0.0; FRAME];
        self.k = 0;
        self.out = [0.0; 3 * FRAME];
        self.head = 0;
        self.len = self.lead;
        self.amount.snap();
    }
    fn latency(&self) -> usize {
        if self.lead == 0 {
            LATENCY_FRAMED
        } else {
            LATENCY_ANY
        }
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
