//! Gain, resonant LP/HP filter, 5-band parametric EQ.

use super::biquad::Shape;
use super::svf::Svf;
use super::{Effect, Smooth};
use crate::from_db;

const UPDATE: usize = 4; // coefficient refresh interval while a parameter moves

pub struct Gain {
    g: Smooth,
}

impl Gain {
    pub fn new() -> Self {
        Self { g: Smooth::new(1.0) }
    }
}

impl Effect for Gain {
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

/// Frequencies are smoothed in log2 space so sweeps move evenly across octaves.
pub struct Filter {
    highpass: bool,
    slope24: bool,
    lf: Smooth,
    q: Smooth,
    st: [Svf; 2],
    n: usize,
}

impl Filter {
    pub fn new() -> Self {
        Self { highpass: false, slope24: false, lf: Smooth::new(13.0), q: Smooth::new(0.707), st: [Svf::default(); 2], n: 0 }
    }
    fn design(&mut self) {
        let f = self.lf.cur.exp2();
        let shape = if self.highpass { Shape::HighPass } else { Shape::LowPass };
        if self.slope24 {
            // 4th-order Butterworth stages; resonance scales the high-Q stage.
            self.st[0].set(shape, f, 0.541, 0.0);
            self.st[1].set(shape, f, 1.307 * self.q.cur / 0.707, 0.0);
        } else {
            self.st[0].set(shape, f, self.q.cur, 0.0);
        }
    }
}

impl Effect for Filter {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.highpass = v >= 0.5,
            1 => self.lf.target = v.max(1.0).log2(),
            2 => self.q.target = v,
            _ => self.slope24 = v >= 0.5,
        }
        self.design();
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            if self.n == 0 && !(self.lf.settled() && self.q.settled()) {
                self.lf.advance(UPDATE);
                self.q.advance(UPDATE);
                self.design();
            }
            self.n = (self.n + 1) % UPDATE;
            let y = self.st[0].tick(*s);
            *s = if self.slope24 { self.st[1].tick(y) } else { y };
        }
    }
    fn reset(&mut self) {
        self.lf.snap();
        self.q.snap();
        self.design();
        self.st.iter_mut().for_each(Svf::reset);
    }
}

/// Low shelf, three peaks, high shelf.
pub struct Eq5 {
    // (log2 freq, gain dB, q) per band
    f: [Smooth; 5],
    g: [Smooth; 5],
    q: [Smooth; 5],
    bands: [Svf; 5],
    active: [bool; 5],
    n: usize,
}

impl Eq5 {
    pub fn new() -> Self {
        Self {
            f: [Smooth::new(6.6); 5],
            g: [Smooth::new(0.0); 5],
            q: [Smooth::new(0.707); 5],
            bands: [Svf::default(); 5],
            active: [false; 5],
            n: 0,
        }
    }
    fn shape(i: usize) -> Shape {
        match i {
            0 => Shape::LowShelf,
            4 => Shape::HighShelf,
            _ => Shape::Peak,
        }
    }
    fn design(&mut self, i: usize) {
        self.bands[i].set(Self::shape(i), self.f[i].cur.exp2(), self.q[i].cur, self.g[i].cur);
        // A flat band costs nothing; skip it until its gain moves away from 0 dB.
        self.active[i] = self.g[i].cur.abs() > 1e-3 || self.g[i].target.abs() > 1e-3;
    }
}

impl Effect for Eq5 {
    fn set(&mut self, idx: usize, v: f32) {
        // f1 g1 | f2 g2 q2 | f3 g3 q3 | f4 g4 q4 | f5 g5
        let (band, what) = match idx {
            0 => (0, 'f'),
            1 => (0, 'g'),
            12 => (4, 'g'),
            11 => (4, 'f'),
            i => ((i - 2) / 3 + 1, ['f', 'g', 'q'][(i - 2) % 3]),
        };
        match what {
            'f' => self.f[band].target = v.max(1.0).log2(),
            'g' => self.g[band].target = v,
            _ => self.q[band].target = v,
        }
        self.design(band);
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            if self.n == 0 {
                for i in 0..5 {
                    if !(self.f[i].settled() && self.g[i].settled() && self.q[i].settled()) {
                        self.f[i].advance(UPDATE);
                        self.g[i].advance(UPDATE);
                        self.q[i].advance(UPDATE);
                        self.design(i);
                    }
                }
            }
            self.n = (self.n + 1) % UPDATE;
            let mut y = *s;
            for i in 0..5 {
                if self.active[i] {
                    y = self.bands[i].tick(y);
                }
            }
            *s = y;
        }
    }
    fn reset(&mut self) {
        for i in 0..5 {
            self.f[i].snap();
            self.g[i].snap();
            self.q[i].snap();
            self.design(i);
            self.bands[i].reset();
        }
    }
}
