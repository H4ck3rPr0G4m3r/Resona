//! SOLA crossfade between consecutive engine blocks (twin of engine/resona_engine/pipeline/sola.py).

pub struct Sola {
    x: usize,
    s: usize,
    fade_in: Vec<f32>,
    fade_out: Vec<f32>,
    tail: Vec<f32>,
    has_tail: bool,
    pub last_offset: usize,
    /// Normalized correlation at the chosen offset (1.0 = seamless).
    pub last_ncc: f32,
}

impl Sola {
    pub fn new(crossfade: usize, search: usize) -> Self {
        // Raised-cosine (equal-gain) fade: SOLA aligns the two signals so they are correlated,
        // where an equal-power fade would bump the level by up to +3 dB at every seam.
        let fade_in: Vec<f32> = (0..crossfade)
            .map(|i| ((i as f32 + 0.5) / crossfade as f32 * std::f32::consts::FRAC_PI_2).sin().powi(2))
            .collect();
        let fade_out = fade_in.iter().map(|v| 1.0 - v).collect();
        Self { x: crossfade, s: search, fade_in, fade_out, tail: vec![0.0; crossfade], has_tail: false, last_offset: search / 2, last_ncc: 1.0 }
    }

    pub fn reset(&mut self) {
        self.has_tail = false;
    }

    fn best_offset(&self, y: &[f32]) -> (usize, f32) {
        if !self.has_tail || self.x == 0 {
            return (self.s / 2, 1.0);
        }
        let tail_e: f32 = self.tail.iter().map(|v| v * v).sum();
        let mut best = (self.s / 2, f32::MIN);
        for k in 0..=self.s {
            let w = &y[k..k + self.x];
            let (mut num, mut e) = (0.0f32, 0.0f32);
            for (a, b) in w.iter().zip(&self.tail) {
                num += a * b;
                e += a * a;
            }
            let ncc = num / (e * tail_e + 1e-12).sqrt();
            if ncc > best.1 {
                best = (k, ncc);
            }
        }
        best
    }

    /// `y` has `block + crossfade + search` samples; writes `block` samples to `out`.
    pub fn process(&mut self, y: &[f32], block: usize, out: &mut [f32]) {
        assert!(y.len() >= block + self.x + self.s && out.len() >= block);
        let (k, ncc) = self.best_offset(y);
        self.last_offset = k;
        self.last_ncc = ncc;
        out[..block].copy_from_slice(&y[k..k + block]);
        if self.has_tail {
            for i in 0..self.x {
                out[i] = out[i] * self.fade_in[i] + self.tail[i] * self.fade_out[i];
            }
        }
        self.tail.copy_from_slice(&y[k + block..k + block + self.x]);
        self.has_tail = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_signal_passes_through() {
        // Engine blocks cut from one continuous sine with the nominal offset must reassemble it.
        let (block, x, s) = (480, 48, 48);
        // Voice-like: several inharmonic partials so the correlation peak is unique.
        let sig: Vec<f32> = (0..20_000)
            .map(|i| {
                let t = i as f32;
                (t * 0.031).sin() + 0.5 * (t * 0.113).sin() + 0.3 * (t * 0.271).sin()
            })
            .collect();
        let mut sola = Sola::new(x, s);
        let mut out = vec![0f32; block];
        let mut joined = Vec::new();
        for b in 0..30 {
            let start = 1000 + b * block - s / 2;
            sola.process(&sig[start..start + block + x + s], block, &mut out);
            joined.extend_from_slice(&out);
        }
        for (i, v) in joined.iter().enumerate().skip(block) {
            assert!((v - sig[1000 + i]).abs() < 1e-3, "sample {i}");
        }
    }

    #[test]
    fn realigns_shifted_block() {
        let (block, x, s) = (480, 48, 48);
        let sig: Vec<f32> = (0..20_000).map(|i| (i as f32 * 0.05).sin() * (i as f32 * 0.0007).cos()).collect();
        let mut sola = Sola::new(x, s);
        let mut out = vec![0f32; block];
        sola.process(&sig[976..976 + block + x + s], block, &mut out);
        // Next block arrives 5 samples late; SOLA should pick offset nominal-5.
        let start = 976 + block + 5;
        sola.process(&sig[start..start + block + x + s], block, &mut out);
        assert_eq!(sola.last_offset, s / 2 - 5);
        assert!(sola.last_ncc > 0.99);
    }
}
