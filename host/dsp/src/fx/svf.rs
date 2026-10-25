//! Trapezoidal state-variable filter (Andrew Simper / Cytomic). Unlike direct-form biquads it
//! stays clean when its frequency or gain is modulated quickly, so every swept filter uses it.

use super::biquad::Shape;
use super::SR;

#[derive(Clone, Copy, Debug, Default)]
pub struct Svf {
    a1: f32,
    a2: f32,
    a3: f32,
    m0: f32,
    m1: f32,
    m2: f32,
    ic1: f32,
    ic2: f32,
}

impl Svf {
    pub fn set(&mut self, shape: Shape, freq: f32, q: f32, gain_db: f32) {
        let f = freq.clamp(10.0, SR * 0.49);
        let a = 10f32.powf(gain_db / 40.0);
        let base = (std::f32::consts::PI * f / SR).tan();
        let q = q.max(0.05);
        let (g, k, m0, m1, m2) = match shape {
            Shape::LowPass => (base, 1.0 / q, 0.0, 0.0, 1.0),
            Shape::HighPass => (base, 1.0 / q, 1.0, -1.0 / q, -1.0),
            Shape::BandPass => (base, 1.0 / q, 0.0, 1.0 / q, 0.0),
            Shape::Peak => {
                let k = 1.0 / (q * a);
                (base, k, 1.0, k * (a * a - 1.0), 0.0)
            }
            Shape::LowShelf => {
                let k = 1.0 / q;
                (base / a.sqrt(), k, 1.0, k * (a - 1.0), a * a - 1.0)
            }
            Shape::HighShelf => {
                let k = 1.0 / q;
                (base * a.sqrt(), k, a * a, k * (1.0 - a) * a, 1.0 - a * a)
            }
        };
        self.a1 = 1.0 / (1.0 + g * (g + k));
        self.a2 = g * self.a1;
        self.a3 = g * self.a2;
        self.m0 = m0;
        self.m1 = m1;
        self.m2 = m2;
    }

    #[inline]
    pub fn tick(&mut self, v0: f32) -> f32 {
        let v3 = v0 - self.ic2;
        let v1 = self.a1 * self.ic1 + self.a2 * v3;
        let v2 = self.ic2 + self.a2 * self.ic1 + self.a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;
        self.m0 * v0 + self.m1 * v1 + self.m2 * v2
    }

    pub fn reset(&mut self) {
        self.ic1 = 0.0;
        self.ic2 = 0.0;
    }
}
