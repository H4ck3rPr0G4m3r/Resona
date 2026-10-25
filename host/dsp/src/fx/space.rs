//! Reverb (Schroeder/Moorer network after Freeverb, mono) and a two-voice chorus.

use super::{Effect, Smooth, SR};

struct Comb {
    buf: Vec<f32>,
    pos: usize,
    store: f32,
}

impl Comb {
    fn new(n: usize) -> Self {
        Self { buf: vec![0.0; n], pos: 0, store: 0.0 }
    }
    #[inline]
    fn tick(&mut self, x: f32, feedback: f32, damp: f32) -> f32 {
        let y = self.buf[self.pos];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.pos] = x + self.store * feedback;
        self.pos = (self.pos + 1) % self.buf.len();
        y
    }
}

struct Allpass {
    buf: Vec<f32>,
    pos: usize,
}

impl Allpass {
    fn new(n: usize) -> Self {
        Self { buf: vec![0.0; n], pos: 0 }
    }
    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let b = self.buf[self.pos];
        self.buf[self.pos] = x + b * 0.5;
        self.pos = (self.pos + 1) % self.buf.len();
        b - x
    }
}

const COMBS: [usize; 8] = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
const ALLPASSES: [usize; 4] = [556, 441, 341, 225];

pub struct Reverb {
    combs: Vec<Comb>,
    aps: Vec<Allpass>,
    pre: Vec<f32>,
    pre_pos: usize,
    pre_len: Smooth,
    size: Smooth,
    damp: Smooth,
    mix: Smooth,
}

impl Reverb {
    pub fn new() -> Self {
        let scale = |n: usize| (n as f32 * SR / 44_100.0) as usize;
        Self {
            combs: COMBS.iter().map(|&n| Comb::new(scale(n))).collect(),
            aps: ALLPASSES.iter().map(|&n| Allpass::new(scale(n))).collect(),
            pre: vec![0.0; (0.1 * SR) as usize + 2],
            pre_pos: 0,
            pre_len: Smooth::new(480.0),
            size: Smooth::new(0.5),
            damp: Smooth::new(0.5),
            mix: Smooth::new(0.2),
        }
    }
}

impl Effect for Reverb {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.size.target = v,
            1 => self.damp.target = v,
            2 => self.pre_len.target = v * 1e-3 * SR,
            _ => self.mix.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        let n = self.pre.len();
        for s in x.iter_mut() {
            let feedback = self.size.next() * 0.28 + 0.7;
            let damp = self.damp.next() * 0.4;
            let mix = self.mix.next();
            // Pre-delay with a fractional, smoothed read position (no zipper when moved).
            self.pre[self.pre_pos] = *s;
            let d = self.pre_len.next().clamp(0.0, (n - 2) as f32);
            let rp = (self.pre_pos as f32 - d + n as f32) % n as f32;
            let i = rp as usize;
            let f = rp - i as f32;
            let pd = self.pre[i] * (1.0 - f) + self.pre[(i + 1) % n] * f;
            self.pre_pos = (self.pre_pos + 1) % n;

            let input = pd * 0.015;
            let mut wet = 0.0;
            for c in &mut self.combs {
                wet += c.tick(input, feedback, damp);
            }
            for a in &mut self.aps {
                wet = a.tick(wet);
            }
            // Equal-ish loudness crossfade: dry falls slower than wet rises.
            *s = *s * (1.0 - mix * mix) + wet * 3.0 * mix;
        }
    }
    fn reset(&mut self) {
        for c in &mut self.combs {
            c.buf.fill(0.0);
            c.store = 0.0;
        }
        for a in &mut self.aps {
            a.buf.fill(0.0);
        }
        self.pre.fill(0.0);
        self.pre_len.snap();
        self.size.snap();
        self.damp.snap();
        self.mix.snap();
    }
}

pub struct Chorus {
    buf: Vec<f32>,
    pos: usize,
    phase: f32,
    rate: Smooth,
    depth: Smooth,
    delay: Smooth,
    feedback: Smooth,
    mix: Smooth,
    fb: f32,
}

impl Chorus {
    pub fn new() -> Self {
        Self {
            buf: vec![0.0; (0.05 * SR) as usize],
            pos: 0,
            phase: 0.0,
            rate: Smooth::new(0.8),
            depth: Smooth::new(3.0),
            delay: Smooth::new(12.0),
            feedback: Smooth::new(0.1),
            mix: Smooth::new(0.35),
            fb: 0.0,
        }
    }
    #[inline]
    fn read(&self, delay_samples: f32) -> f32 {
        let n = self.buf.len();
        let rp = (self.pos as f32 - delay_samples + n as f32) % n as f32;
        let i = rp as usize;
        let f = rp - i as f32;
        // 4-point cubic (Catmull-Rom) interpolation keeps the modulated delay clean.
        let p = |k: isize| self.buf[((i as isize + k).rem_euclid(n as isize)) as usize];
        let (y0, y1, y2, y3) = (p(-1), p(0), p(1), p(2));
        y1 + 0.5 * f * (y2 - y0 + f * (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3 + f * (3.0 * (y1 - y2) + y3 - y0)))
    }
}

impl Effect for Chorus {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.rate.target = v,
            1 => self.depth.target = v,
            2 => self.delay.target = v,
            3 => self.feedback.target = v,
            _ => self.mix.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        let ms = SR * 1e-3;
        for s in x.iter_mut() {
            let (rate, depth, base, fb, mix) = (self.rate.next(), self.depth.next(), self.delay.next(), self.feedback.next(), self.mix.next());
            self.phase = (self.phase + rate / SR) % 1.0;
            let tau = std::f32::consts::TAU;
            let d1 = (base + depth * (tau * self.phase).sin()) * ms;
            let d2 = (base + depth * (tau * self.phase + std::f32::consts::PI).sin()) * ms;
            self.buf[self.pos] = *s + self.fb * fb;
            let wet = 0.5 * (self.read(d1) + self.read(d2));
            self.fb = wet;
            self.pos = (self.pos + 1) % self.buf.len();
            *s = *s * (1.0 - 0.5 * mix) + wet * mix;
        }
    }
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.fb = 0.0;
        self.rate.snap();
        self.depth.snap();
        self.delay.snap();
        self.feedback.snap();
        self.mix.snap();
    }
}
