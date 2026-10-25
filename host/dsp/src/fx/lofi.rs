//! Lo-fi mic: band-limit, drive, bit depth, sample-rate reduction and hiss.

use super::biquad::Shape;
use super::svf::Svf;
use super::{Effect, Smooth};
use crate::from_db;

pub struct LoFi {
    low: Smooth,  // log2 Hz
    high: Smooth, // log2 Hz
    drive: Smooth,
    bits: Smooth,
    ds: Smooth,
    noise: Smooth,
    mix: Smooth,
    hp: Svf,
    lp: [Svf; 2],
    held: f32,
    phase: f32,
    rng: u32,
    n: usize,
}

impl LoFi {
    pub fn new() -> Self {
        let mut s = Self {
            low: Smooth::new(300f32.log2()),
            high: Smooth::new(3400f32.log2()),
            drive: Smooth::new(6.0),
            bits: Smooth::new(10.0),
            ds: Smooth::new(3.0),
            noise: Smooth::new(-62.0),
            mix: Smooth::new(1.0),
            hp: Svf::default(),
            lp: [Svf::default(); 2],
            held: 0.0,
            phase: 0.0,
            rng: 0x1234_5678,
            n: 0,
        };
        s.design();
        s
    }
    fn design(&mut self) {
        self.hp.set(Shape::HighPass, self.low.cur.exp2(), 0.707, 0.0);
        let f = self.high.cur.exp2();
        self.lp[0].set(Shape::LowPass, f, 0.541, 0.0);
        self.lp[1].set(Shape::LowPass, f, 1.307, 0.0);
    }
    #[inline]
    fn white(&mut self) -> f32 {
        // xorshift32 → [-1, 1)
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

impl Effect for LoFi {
    fn set(&mut self, idx: usize, v: f32) {
        match idx {
            0 => self.low.target = v.log2(),
            1 => self.high.target = v.log2(),
            2 => self.drive.target = v,
            3 => self.bits.target = v,
            4 => self.ds.target = v,
            5 => self.noise.target = v,
            _ => self.mix.target = v,
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            if self.n == 0 && !(self.low.settled() && self.high.settled()) {
                self.low.advance(4);
                self.high.advance(4);
                self.design();
            }
            self.n = (self.n + 1) % 4;
            let dry = *s;
            let drive = from_db(self.drive.next());
            let levels = self.bits.next().exp2() * 0.5;
            let ds = self.ds.next().max(1.0);
            let noise = from_db(self.noise.next());
            let mix = self.mix.next();

            let mut y = self.hp.tick(*s);
            y = (y * drive).tanh() / drive.tanh().max(1e-3);
            // Sample-and-hold at SR/ds (fractional rate, so the control sweeps smoothly).
            self.phase += 1.0;
            if self.phase >= ds {
                self.phase -= ds;
                self.held = (y * levels).round() / levels;
            }
            y = self.held + self.white() * noise;
            let l0 = self.lp[0].tick(y);
            y = self.lp[1].tick(l0);
            *s = dry + (y - dry) * mix;
        }
    }
    fn reset(&mut self) {
        for sm in [&mut self.low, &mut self.high, &mut self.drive, &mut self.bits, &mut self.ds, &mut self.noise, &mut self.mix] {
            sm.snap();
        }
        self.design();
        self.hp.reset();
        self.lp.iter_mut().for_each(Svf::reset);
        self.held = 0.0;
        self.phase = 0.0;
    }
}
