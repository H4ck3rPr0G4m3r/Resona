//! RBJ-cookbook biquads (transposed direct form II), with designs for every shape the chains use.

use std::f32::consts::PI;

pub const SR: f32 = 48_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    LowPass,
    HighPass,
    BandPass,
    Peak,
    LowShelf,
    HighShelf,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Coefs {
    pub b0: f32,
    pub b1: f32,
    pub b2: f32,
    pub a1: f32,
    pub a2: f32,
}

impl Coefs {
    pub const IDENTITY: Coefs = Coefs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };

    /// `freq` Hz, `q` quality, `gain_db` for peak/shelf shapes.
    pub fn design(shape: Shape, freq: f32, q: f32, gain_db: f32) -> Coefs {
        let f = freq.clamp(10.0, SR * 0.49);
        let w = 2.0 * PI * f / SR;
        let (sw, cw) = w.sin_cos();
        let q = q.max(0.05);
        let alpha = sw / (2.0 * q);
        let a = 10f32.powf(gain_db / 40.0);
        let (b0, b1, b2, a0, a1, a2) = match shape {
            Shape::LowPass => ((1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
            Shape::HighPass => ((1.0 + cw) / 2.0, -(1.0 + cw), (1.0 + cw) / 2.0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
            Shape::BandPass => (alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * cw, 1.0 - alpha),
            Shape::Peak => (1.0 + alpha * a, -2.0 * cw, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * cw, 1.0 - alpha / a),
            Shape::LowShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) - (a - 1.0) * cw + s),
                    2.0 * a * ((a - 1.0) - (a + 1.0) * cw),
                    a * ((a + 1.0) - (a - 1.0) * cw - s),
                    (a + 1.0) + (a - 1.0) * cw + s,
                    -2.0 * ((a - 1.0) + (a + 1.0) * cw),
                    (a + 1.0) + (a - 1.0) * cw - s,
                )
            }
            Shape::HighShelf => {
                let s = 2.0 * a.sqrt() * alpha;
                (
                    a * ((a + 1.0) + (a - 1.0) * cw + s),
                    -2.0 * a * ((a - 1.0) + (a + 1.0) * cw),
                    a * ((a + 1.0) + (a - 1.0) * cw - s),
                    (a + 1.0) - (a - 1.0) * cw + s,
                    2.0 * ((a - 1.0) - (a + 1.0) * cw),
                    (a + 1.0) - (a - 1.0) * cw - s,
                )
            }
        };
        Coefs { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 }
    }

    /// Magnitude response in dB at `freq` (used by tests; the UI mirrors this in TypeScript).
    pub fn response_db(&self, freq: f32) -> f32 {
        let w = 2.0 * PI * freq / SR;
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let nr = self.b0 + self.b1 * c1 + self.b2 * c2;
        let ni = -(self.b1 * s1 + self.b2 * s2);
        let dr = 1.0 + self.a1 * c1 + self.a2 * c2;
        let di = -(self.a1 * s1 + self.a2 * s2);
        10.0 * ((nr * nr + ni * ni) / (dr * dr + di * di)).max(1e-20).log10()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Biquad {
    pub c: Coefs,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub fn new(c: Coefs) -> Self {
        Self { c, z1: 0.0, z2: 0.0 }
    }

    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let c = &self.c;
        let y = c.b0 * x + self.z1;
        self.z1 = c.b1 * x - c.a1 * y + self.z2;
        self.z2 = c.b2 * x - c.a2 * y;
        y
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_have_expected_responses() {
        let lp = Coefs::design(Shape::LowPass, 1000.0, 0.707, 0.0);
        assert!(lp.response_db(100.0).abs() < 0.1);
        assert!((lp.response_db(1000.0) + 3.0).abs() < 0.2);
        assert!(lp.response_db(10_000.0) < -35.0);
        let pk = Coefs::design(Shape::Peak, 2000.0, 1.0, 6.0);
        assert!((pk.response_db(2000.0) - 6.0).abs() < 0.05);
        assert!(pk.response_db(100.0).abs() < 0.2);
        let ls = Coefs::design(Shape::LowShelf, 200.0, 0.707, -4.0);
        assert!((ls.response_db(30.0) + 4.0).abs() < 0.2);
        assert!(ls.response_db(5000.0).abs() < 0.1);
    }
}
