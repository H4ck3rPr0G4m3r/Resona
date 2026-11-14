//! Primitives for composite filters: the small single-purpose stages a filter file chains
//! together. They are ordinary [`Effect`]s (smoothed controls, no allocation while running), but
//! they are not offered as chain effects by themselves; the full effects cover that.

use super::biquad::Shape;
use super::svf::Svf;
use super::{p, pchoice, plog, Effect, KindSpec, ParamSpec, Smooth, Stage, SR};
use crate::from_db;

const FREQ: (f32, f32) = (20.0, 20_000.0);

const fn tone(kind: &'static str, name: &'static str, params: &'static [ParamSpec]) -> KindSpec {
    KindSpec { kind, name, stage: Stage::Both, params }
}

pub static PRIMS: &[KindSpec] = &[
    tone("highpass", "High-pass", &[plog("freq", "Frequency", FREQ.0, FREQ.1, 80.0, "Hz"), plog("q", "Q", 0.5, 12.0, 0.707, "")]),
    tone("lowpass", "Low-pass", &[plog("freq", "Frequency", FREQ.0, FREQ.1, 8_000.0, "Hz"), plog("q", "Q", 0.5, 12.0, 0.707, "")]),
    tone("bandpass", "Band-pass", &[plog("freq", "Frequency", FREQ.0, FREQ.1, 1_000.0, "Hz"), plog("q", "Q", 0.5, 12.0, 0.707, "")]),
    tone(
        "peak",
        "Peak",
        &[plog("freq", "Frequency", FREQ.0, FREQ.1, 1_000.0, "Hz"), plog("q", "Q", 0.5, 12.0, 1.0, ""), p("gain_db", "Gain", -24.0, 24.0, 0.0, "dB")],
    ),
    tone(
        "lowshelf",
        "Low shelf",
        &[plog("freq", "Frequency", FREQ.0, FREQ.1, 200.0, "Hz"), plog("q", "Q", 0.5, 12.0, 0.707, ""), p("gain_db", "Gain", -24.0, 24.0, 0.0, "dB")],
    ),
    tone(
        "highshelf",
        "High shelf",
        &[plog("freq", "Frequency", FREQ.0, FREQ.1, 6_000.0, "Hz"), plog("q", "Q", 0.5, 12.0, 0.707, ""), p("gain_db", "Gain", -24.0, 24.0, 0.0, "dB")],
    ),
    KindSpec { kind: "downsample", name: "Downsample", stage: Stage::Both, params: &[plog("rate", "Rate", 500.0, 48_000.0, 48_000.0, "Hz")] },
    KindSpec { kind: "bitcrush", name: "Bit crush", stage: Stage::Both, params: &[p("bits", "Bits", 1.0, 24.0, 24.0, "")] },
    KindSpec {
        kind: "saturate",
        name: "Saturate",
        stage: Stage::Both,
        params: &[pchoice("mode", "Mode", &["soft", "hard", "tube"], 0.0), plog("drive", "Drive", 1.0, 32.0, 1.0, "x")],
    },
    KindSpec { kind: "noise", name: "Noise", stage: Stage::Both, params: &[p("level_db", "Level", -120.0, -10.0, -120.0, "dB")] },
    KindSpec {
        kind: "crackle",
        name: "Crackle",
        stage: Stage::Both,
        params: &[p("rate", "Rate", 0.0, 40.0, 0.0, "/s"), p("level", "Level", 0.0, 1.0, 0.5, "")],
    },
    KindSpec {
        kind: "delay",
        name: "Delay",
        stage: Stage::Both,
        params: &[plog("time_ms", "Time", 1.0, 1_000.0, 120.0, "ms"), p("feedback", "Feedback", 0.0, 0.95, 0.3, ""), p("mix", "Mix", 0.0, 1.0, 0.3, "")],
    },
    KindSpec { kind: "gain", name: "Gain", stage: Stage::Both, params: &[p("gain_db", "Gain", -60.0, 24.0, 0.0, "dB")] },
];

pub fn spec(kind: &str) -> Option<&'static KindSpec> {
    PRIMS.iter().find(|k| k.kind == kind)
}

pub fn create(kind: &str) -> Option<Box<dyn Effect>> {
    let ks = spec(kind)?;
    let mut fx: Box<dyn Effect> = match kind {
        "highpass" => Box::new(Tone::new(Shape::HighPass)),
        "lowpass" => Box::new(Tone::new(Shape::LowPass)),
        "bandpass" => Box::new(Tone::new(Shape::BandPass)),
        "peak" => Box::new(Tone::new(Shape::Peak)),
        "lowshelf" => Box::new(Tone::new(Shape::LowShelf)),
        "highshelf" => Box::new(Tone::new(Shape::HighShelf)),
        "downsample" => Box::new(Downsample::new()),
        "bitcrush" => Box::new(Bitcrush::new()),
        "saturate" => Box::new(Saturate::new()),
        "noise" => Box::new(Noise::new()),
        "crackle" => Box::new(Crackle::new()),
        "delay" => Box::new(Delay::new()),
        "gain" => Box::new(GainDb::new()),
        _ => return None,
    };
    for (i, ps) in ks.params.iter().enumerate() {
        fx.set(i, ps.default);
    }
    fx.reset();
    Some(fx)
}

/// xorshift32 → [-1, 1).
#[derive(Clone, Copy)]
struct Rng(u32);

impl Rng {
    #[inline]
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    /// [0, 1)
    #[inline]
    fn unit(&mut self) -> f32 {
        self.next() * 0.5 + 0.5
    }
}

// -- filters -------------------------------------------------------------------------------------

/// One second-order section: params freq, q, (gain_db for peak and shelves).
struct Tone {
    shape: Shape,
    lf: Smooth, // log2 Hz, so sweeps move evenly across octaves
    q: Smooth,
    gain: Smooth,
    f: Svf,
    n: usize,
}

impl Tone {
    fn new(shape: Shape) -> Self {
        Self { shape, lf: Smooth::new(10.0), q: Smooth::new(0.707), gain: Smooth::new(0.0), f: Svf::default(), n: 0 }
    }
    fn design(&mut self) {
        self.f.set(self.shape, self.lf.cur.exp2(), self.q.cur, self.gain.cur);
    }
}

impl Effect for Tone {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.lf.target = v.max(1.0).log2(),
            1 => self.q.target = v,
            _ => self.gain.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            if self.n == 0 && !(self.lf.settled() && self.q.settled() && self.gain.settled()) {
                self.lf.advance(4);
                self.q.advance(4);
                self.gain.advance(4);
                self.design();
            }
            self.n = (self.n + 1) % 4;
            *s = self.f.tick(*s);
        }
    }
    fn reset(&mut self) {
        self.lf.snap();
        self.q.snap();
        self.gain.snap();
        self.design();
        self.f.reset();
    }
}

// -- degraders -----------------------------------------------------------------------------------

/// Sample-and-hold at `rate`, with no filtering: the aliasing is the point.
struct Downsample {
    rate: Smooth,
    phase: f32,
    held: f32,
}

impl Downsample {
    fn new() -> Self {
        Self { rate: Smooth::new(SR), phase: 1.0, held: 0.0 }
    }
}

impl Effect for Downsample {
    fn set(&mut self, _idx: usize, v: f32) {
        self.rate.target = v;
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            let step = (self.rate.next() / SR).min(1.0);
            self.phase += step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                self.held = *s;
            }
            *s = self.held;
        }
    }
    fn reset(&mut self) {
        self.rate.snap();
        self.phase = 1.0;
        self.held = 0.0;
    }
}

struct Bitcrush {
    bits: Smooth,
}

impl Bitcrush {
    fn new() -> Self {
        Self { bits: Smooth::new(24.0) }
    }
}

impl Effect for Bitcrush {
    fn set(&mut self, _idx: usize, v: f32) {
        self.bits.target = v;
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            // Fractional bit depths sweep smoothly; 2^(bits-1) steps per unit of amplitude.
            let levels = (self.bits.next() - 1.0).exp2();
            *s = (*s * levels).round() / levels;
        }
    }
    fn reset(&mut self) {
        self.bits.snap();
    }
}

/// Waveshaper. `drive` 1 is clean; above it the curve bends harder. Every curve tops out at ±1,
/// so more drive means louder and denser, never beyond full scale.
struct Saturate {
    mode: u8,
    drive: Smooth,
    dc_in: f32,
    dc_out: f32,
}

impl Saturate {
    fn new() -> Self {
        Self { mode: 0, drive: Smooth::new(1.0), dc_in: 0.0, dc_out: 0.0 }
    }
}

impl Effect for Saturate {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.mode = v.round().clamp(0.0, 2.0) as u8,
            _ => self.drive.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            let d = self.drive.next().max(1.0);
            // tanh(k x) / tanh(k) is the identity as k goes to 0 and keeps full scale at full
            // scale for any k, so the curve grows out of a straight line as drive leaves 1.
            let k = (d - 1.0).max(1e-3);
            *s = match self.mode {
                0 => (*s * k).tanh() / k.tanh(),
                1 => (*s * d).clamp(-1.0, 1.0),
                _ => {
                    // Asymmetric: the negative half bends earlier, which adds even harmonics.
                    let v = *s * k;
                    let y = if v >= 0.0 { v.tanh() } else { (v * 1.6).tanh() / 1.6 } / k.tanh();
                    // Remove the DC that asymmetry creates (one-pole high-pass, ~10 Hz).
                    let out = y - self.dc_in + 0.9987 * self.dc_out;
                    self.dc_in = y;
                    self.dc_out = out;
                    out
                }
            };
        }
    }
    fn reset(&mut self) {
        self.drive.snap();
        self.dc_in = 0.0;
        self.dc_out = 0.0;
    }
}

/// Adds hiss at a fixed level.
struct Noise {
    level: Smooth, // dB
    rng: Rng,
}

impl Noise {
    fn new() -> Self {
        Self { level: Smooth::new(-120.0), rng: Rng(0x9E37_79B9) }
    }
}

impl Effect for Noise {
    fn set(&mut self, _idx: usize, v: f32) {
        self.level.target = v;
    }
    fn process(&mut self, x: &mut [f32]) {
        if self.level.settled() && self.level.target <= -119.0 {
            return;
        }
        for s in x.iter_mut() {
            *s += self.rng.next() * from_db(self.level.next());
        }
    }
    fn reset(&mut self) {
        self.level.snap();
    }
}

/// A bad connection: random clicks and short dropouts, `rate` of them per second on average.
struct Crackle {
    rate: Smooth,
    level: Smooth,
    rng: Rng,
    click: f32,
    drop_left: usize,
    gain: f32,
}

impl Crackle {
    fn new() -> Self {
        Self { rate: Smooth::new(0.0), level: Smooth::new(0.5), rng: Rng(0x2545_F491), click: 0.0, drop_left: 0, gain: 1.0 }
    }
}

impl Effect for Crackle {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.rate.target = v,
            _ => self.level.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        if self.rate.settled() && self.rate.target <= 0.0 && self.drop_left == 0 && self.gain > 0.999 && self.click.abs() < 1e-5 {
            return;
        }
        // Dropouts open and close over ~0.4 ms: audible as a tick, not as a speaker-popping step.
        const EDGE: f32 = 0.05;
        for s in x.iter_mut() {
            let (rate, level) = (self.rate.next(), self.level.next());
            if self.rng.unit() < rate / SR {
                if self.rng.unit() < 0.6 {
                    self.click = (0.1 + 0.5 * self.rng.unit()) * level * self.rng.next().signum();
                } else {
                    self.drop_left = ((0.003 + 0.04 * self.rng.unit() * self.rng.unit()) * SR) as usize;
                }
            }
            let target = if self.drop_left > 0 {
                self.drop_left -= 1;
                1.0 - level
            } else {
                1.0
            };
            self.gain += (target - self.gain) * EDGE;
            *s = *s * self.gain + self.click;
            self.click *= -0.55; // a short ringing tick
        }
    }
    fn reset(&mut self) {
        self.rate.snap();
        self.level.snap();
        self.click = 0.0;
        self.drop_left = 0;
        self.gain = 1.0;
    }
}

// -- delay, gain ---------------------------------------------------------------------------------

const DELAY_MAX: usize = SR as usize + 8; // one second

struct Delay {
    buf: Vec<f32>,
    pos: usize,
    time: Smooth, // samples
    feedback: Smooth,
    mix: Smooth,
}

impl Delay {
    fn new() -> Self {
        Self { buf: vec![0.0; DELAY_MAX], pos: 0, time: Smooth::new(0.12 * SR), feedback: Smooth::new(0.3), mix: Smooth::new(0.3) }
    }
}

impl Effect for Delay {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.time.target = (v * 1e-3 * SR).clamp(1.0, (DELAY_MAX - 4) as f32),
            1 => self.feedback.target = v,
            _ => self.mix.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        let n = self.buf.len();
        for s in x.iter_mut() {
            let (t, fb, mix) = (self.time.next(), self.feedback.next(), self.mix.next());
            // Linear interpolation between the two taps around `t` samples back.
            let whole = t as usize;
            let frac = t - whole as f32;
            let a = self.buf[(self.pos + n - whole) % n];
            let b = self.buf[(self.pos + n - whole - 1) % n];
            let wet = a + (b - a) * frac;
            self.buf[self.pos] = *s + wet * fb;
            self.pos = (self.pos + 1) % n;
            *s += (wet - *s) * mix;
        }
    }
    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.time.snap();
        self.feedback.snap();
        self.mix.snap();
    }
}

struct GainDb {
    g: Smooth,
}

impl GainDb {
    fn new() -> Self {
        Self { g: Smooth::new(1.0) }
    }
}

impl Effect for GainDb {
    fn set(&mut self, _idx: usize, v: f32) {
        self.g.target = from_db(v);
    }
    fn process(&mut self, x: &mut [f32]) {
        if self.g.settled() {
            let g = self.g.target;
            x.iter_mut().for_each(|s| *s *= g);
        } else {
            x.iter_mut().for_each(|s| *s *= self.g.next());
        }
    }
    fn reset(&mut self) {
        self.g.snap();
    }
}
