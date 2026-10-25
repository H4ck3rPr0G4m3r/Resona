//! Gate, compressor, split-band de-esser, true-peak lookahead limiter.

use super::biquad::Shape;
use super::svf::Svf;
use super::{Effect, Smooth, SR};
use crate::{from_db, to_db};

fn coef(ms: f32) -> f32 {
    // one-pole coefficient for a time constant in ms
    1.0 - (-1.0 / (ms.max(0.01) * 1e-3 * SR)).exp()
}

// -- gate ------------------------------------------------------------------------------------------

pub struct Gate {
    thr_db: f32,
    attack: f32,
    hold: usize,
    release: f32,
    floor: Smooth, // closed gain (linear), smoothed so the range control doesn't click
    env: f32,
    held: usize,
    g: f32,
}

impl Gate {
    pub fn new() -> Self {
        Self { thr_db: -50.0, attack: coef(1.0), hold: 0, release: coef(150.0), floor: Smooth::new(0.001), env: 0.0, held: 0, g: 0.0 }
    }
}

impl Effect for Gate {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.thr_db = v,
            1 => self.attack = coef(v),
            2 => self.hold = (v * 1e-3 * SR) as usize,
            3 => self.release = coef(v),
            _ => self.floor.target = from_db(v),
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        let thr = from_db(self.thr_db);
        let close_thr = thr * 0.7; // ~3 dB hysteresis so the gate doesn't chatter
        let det_rel = coef(20.0);
        for s in x.iter_mut() {
            let a = s.abs();
            self.env = if a > self.env { a } else { self.env + (a - self.env) * det_rel };
            let floor = self.floor.next();
            let open = if self.env > thr {
                self.held = self.hold;
                true
            } else if self.env > close_thr || self.held > 0 {
                self.held = self.held.saturating_sub(1);
                self.g > floor + 1e-6 && (self.env > close_thr || self.held > 0)
            } else {
                false
            };
            let target = if open { 1.0 } else { floor };
            let k = if target > self.g { self.attack } else { self.release };
            self.g += (target - self.g) * k;
            *s *= self.g;
        }
    }
    fn reset(&mut self) {
        self.floor.snap();
        self.env = 0.0;
        self.held = 0;
        self.g = self.floor.cur;
    }
}

// -- compressor ------------------------------------------------------------------------------------

pub struct Compressor {
    thr: Smooth,
    ratio: Smooth,
    knee: Smooth,
    makeup: Smooth,
    attack: f32,
    release: f32,
    gr_db: f32, // current gain reduction (<= 0)
}

impl Compressor {
    pub fn new() -> Self {
        Self {
            thr: Smooth::new(-18.0),
            ratio: Smooth::new(3.0),
            knee: Smooth::new(6.0),
            makeup: Smooth::new(0.0),
            attack: coef(5.0),
            release: coef(120.0),
            gr_db: 0.0,
        }
    }

    /// Static curve: gain change in dB for an input level in dB (soft knee).
    pub fn curve(level: f32, thr: f32, ratio: f32, knee: f32) -> f32 {
        let over = level - thr;
        let slope = 1.0 / ratio - 1.0;
        if knee > 0.0 && over.abs() <= knee / 2.0 {
            slope * (over + knee / 2.0).powi(2) / (2.0 * knee)
        } else if over > 0.0 {
            slope * over
        } else {
            0.0
        }
    }
}

impl Effect for Compressor {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.thr.target = v,
            1 => self.ratio.target = v,
            2 => self.attack = coef(v),
            3 => self.release = coef(v),
            4 => self.knee.target = v,
            _ => self.makeup.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            let (thr, ratio, knee, makeup) = (self.thr.next(), self.ratio.next(), self.knee.next(), self.makeup.next());
            let want = Self::curve(to_db(s.abs()), thr, ratio, knee);
            let k = if want < self.gr_db { self.attack } else { self.release };
            self.gr_db += (want - self.gr_db) * k;
            *s *= from_db(self.gr_db + makeup);
        }
    }
    fn reset(&mut self) {
        self.thr.snap();
        self.ratio.snap();
        self.knee.snap();
        self.makeup.snap();
        self.gr_db = 0.0;
    }
}

// -- de-esser --------------------------------------------------------------------------------------

/// Splits at `freq` (complementary: low = x - high, so the bands always sum back exactly) and
/// turns the high band down when its level passes the threshold.
pub struct DeEsser {
    lf: Smooth,
    thr: Smooth,
    max_red: Smooth,
    hp: [Svf; 2],
    env: f32,
    gr_db: f32,
    n: usize,
}

impl DeEsser {
    pub fn new() -> Self {
        Self { lf: Smooth::new(6000f32.log2()), thr: Smooth::new(-30.0), max_red: Smooth::new(8.0), hp: [Svf::default(); 2], env: 0.0, gr_db: 0.0, n: 0 }
    }
    fn design(&mut self) {
        let f = self.lf.cur.exp2();
        self.hp[0].set(Shape::HighPass, f, 0.541, 0.0);
        self.hp[1].set(Shape::HighPass, f, 1.307, 0.0);
    }
}

impl Effect for DeEsser {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.lf.target = v.log2(),
            1 => self.thr.target = v,
            _ => self.max_red.target = v,
        }
        self.design();
    }
    fn process(&mut self, x: &mut [f32]) {
        let (att, rel, det) = (coef(1.0), coef(60.0), coef(5.0));
        for s in x.iter_mut() {
            if self.n == 0 && !self.lf.settled() {
                self.lf.advance(4);
                self.design();
            }
            self.n = (self.n + 1) % 4;
            let h0 = self.hp[0].tick(*s);
            let hi = self.hp[1].tick(h0);
            let lo = *s - hi;
            let a = hi.abs();
            self.env += (a - self.env) * if a > self.env { att } else { det };
            let over = to_db(self.env) - self.thr.next();
            let want = if over > 0.0 { -(over * 0.75).min(self.max_red.next()) } else { 0.0 };
            self.gr_db += (want - self.gr_db) * if want < self.gr_db { att } else { rel };
            *s = lo + hi * from_db(self.gr_db);
        }
    }
    fn reset(&mut self) {
        self.lf.snap();
        self.thr.snap();
        self.max_red.snap();
        self.design();
        self.hp.iter_mut().for_each(Svf::reset);
        self.env = 0.0;
        self.gr_db = 0.0;
    }
}

// -- limiter ---------------------------------------------------------------------------------------

const LOOK: usize = 64; // 1.33 ms lookahead
const TP_DELAY: usize = 4; // interpolator needs 4 future samples
const HIST: usize = 128; // power of two > LOOK + TP_DELAY + taps

/// True-peak limiter. Peaks are estimated at 4x oversampling (inter-sample overs are what clip
/// after a lossy codec), the required gain is min-filtered over the lookahead and box-averaged,
/// so gain reduction ramps in linearly and is fully in place when the peak plays.
pub struct Limiter {
    ceil: Smooth,
    release: f32,
    hist: [f32; HIST],
    req: [f32; LOOK + 1],
    boxbuf: [f32; LOOK + 1],
    bpos: usize,
    sum: f32,
    pos: usize,
    rpos: usize,
    g: f32,
    pub enabled: bool,
}

/// 8-tap windowed-sinc interpolation kernels for fractional positions 1/4, 2/4, 3/4.
fn kernels() -> [[f32; 8]; 3] {
    let mut k = [[0f32; 8]; 3];
    for (p, row) in k.iter_mut().enumerate() {
        let frac = (p + 1) as f32 / 4.0;
        let mut sum = 0.0;
        for (j, v) in row.iter_mut().enumerate() {
            let t = j as f32 - 3.0 - frac; // taps at -3..4 around the sample
            let sinc = if t.abs() < 1e-6 { 1.0 } else { (std::f32::consts::PI * t).sin() / (std::f32::consts::PI * t) };
            let w = 0.5 + 0.5 * (std::f32::consts::PI * t / 4.5).cos();
            *v = sinc * w;
            sum += *v;
        }
        row.iter_mut().for_each(|v| *v /= sum);
    }
    k
}

impl Limiter {
    pub fn new() -> Self {
        Self {
            ceil: Smooth::new(from_db(-1.0)),
            release: coef(60.0),
            hist: [0.0; HIST],
            req: [1.0; LOOK + 1],
            boxbuf: [1.0; LOOK + 1],
            bpos: 0,
            sum: (LOOK + 1) as f32,
            pos: 0,
            rpos: 0,
            g: 1.0,
            enabled: true,
        }
    }
    pub const LATENCY: usize = LOOK + TP_DELAY;

    /// Push into the box-average ring, returning the value that falls out.
    fn boxbuf_pop(&mut self, v: f32) -> f32 {
        let old = self.boxbuf[self.bpos];
        self.boxbuf[self.bpos] = v;
        self.bpos = (self.bpos + 1) % (LOOK + 1);
        if self.bpos == 0 {
            self.sum = self.boxbuf.iter().sum::<f32>() + old - v; // resync (caller adds v - old)
        }
        old
    }
}

impl Effect for Limiter {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.ceil.target = from_db(v),
            _ => self.release = coef(v),
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        let k = kernels();
        let m = HIST - 1;
        for s in x.iter_mut() {
            self.hist[self.pos] = *s;
            // True peak around the sample TP_DELAY back.
            let c = (self.pos + HIST - TP_DELAY) & m;
            let mut peak = self.hist[c].abs();
            for row in &k {
                let mut v = 0.0;
                for (j, w) in row.iter().enumerate() {
                    v += w * self.hist[(c + HIST - 3 + j) & m];
                }
                peak = peak.max(v.abs());
            }
            let ceil = self.ceil.next();
            let need = if peak > ceil { ceil / peak } else { 1.0 };
            // Sliding minimum over the lookahead, then a box average of that minimum.
            self.req[self.rpos] = need;
            self.rpos = (self.rpos + 1) % (LOOK + 1);
            let mn = self.req.iter().copied().fold(1.0f32, f32::min);
            // running box average over the last LOOK+1 minima
            self.sum += mn - self.boxbuf_pop(mn);
            let target = self.sum / (LOOK + 1) as f32;
            self.g = if target < self.g { target } else { self.g + (target - self.g) * self.release };
            let raw = self.hist[(self.pos + HIST - Self::LATENCY) & m];
            self.pos = (self.pos + 1) & m;
            // Safety net for interpolation error: never exceed the ceiling by more than 0.1 dB.
            *s = if self.enabled { (raw * self.g).clamp(-ceil * 1.0116, ceil * 1.0116) } else { raw };
        }
    }
    fn reset(&mut self) {
        self.ceil.snap();
        self.hist = [0.0; HIST];
        self.req = [1.0; LOOK + 1];
        self.boxbuf = [1.0; LOOK + 1];
        self.bpos = 0;
        self.sum = (LOOK + 1) as f32;
        self.g = 1.0;
    }
    fn latency(&self) -> usize {
        Self::LATENCY
    }
}
