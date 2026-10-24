//! Streaming rational (L/M) polyphase resampler with a Kaiser-windowed sinc prototype.
//!
//! Real-time safe: all buffers are allocated in `new`; `process` never allocates.

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
    while term > 1e-12 * sum {
        term *= (x / (2.0 * k)).powi(2);
        sum += term;
        k += 1.0;
    }
    sum
}

pub struct Resampler {
    up: usize,
    down: usize,
    taps: usize,
    /// coeffs[phase][j] multiplies x[base - j]
    coeffs: Vec<Vec<f32>>,
    /// Input history: the last `taps - 1` samples followed by the current chunk.
    hist: Vec<f32>,
    hist_len: usize,
    /// Upsampled-domain position of the next output, relative to hist[0] (in units of 1/up input samples).
    pos: usize,
    max_in: usize,
}

impl Resampler {
    /// `taps` is the number of input taps per output (filter length / up). 32–64 is plenty here.
    pub fn new(in_rate: u32, out_rate: u32, max_in: usize, taps: usize) -> Self {
        let g = gcd(in_rate, out_rate);
        let (up, down) = ((out_rate / g) as usize, (in_rate / g) as usize);
        let n = up * taps;
        // Cutoff at 0.95 × the lower Nyquist, expressed in cycles per upsampled sample.
        let fc = 0.5 * 0.95 / up.max(down) as f64;
        let beta = 8.0;
        let i0b = bessel_i0(beta);
        let mid = (n as f64 - 1.0) / 2.0;
        let mut proto = vec![0f64; n];
        for (i, h) in proto.iter_mut().enumerate() {
            let t = i as f64 - mid;
            let sinc = if t == 0.0 { 1.0 } else { (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t) / (2.0 * fc) };
            let r = (2.0 * i as f64 / (n as f64 - 1.0)) - 1.0;
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            *h = 2.0 * fc * sinc * w * up as f64;
        }
        // Normalize each phase to unity DC gain (removes passband ripple between phases).
        let mut coeffs = vec![vec![0f32; taps]; up];
        for (p, c) in coeffs.iter_mut().enumerate() {
            let sum: f64 = (0..taps).map(|j| proto[p + j * up]).sum();
            for (j, v) in c.iter_mut().enumerate() {
                *v = (proto[p + j * up] / sum) as f32;
            }
        }
        let cap = taps + max_in + 1;
        Self { up, down, taps, coeffs, hist: vec![0.0; cap], hist_len: taps - 1, pos: (taps - 1) * up, max_in }
    }

    pub fn ratio(&self) -> (usize, usize) {
        (self.up, self.down)
    }

    /// Max output samples produced for `n` input samples.
    pub fn max_out(&self, n: usize) -> usize {
        (n * self.up).div_ceil(self.down) + 1
    }

    /// Group delay in input samples.
    pub fn delay_in(&self) -> f32 {
        (self.taps as f32 - 1.0) / 2.0
    }

    pub fn reset(&mut self) {
        self.hist.fill(0.0);
        self.hist_len = self.taps - 1;
        self.pos = (self.taps - 1) * self.up;
    }

    /// Resample `input` (≤ max_in samples) into `out`; returns samples written.
    pub fn process(&mut self, input: &[f32], out: &mut [f32]) -> usize {
        assert!(input.len() <= self.max_in, "resampler chunk too large");
        if self.up == self.down {
            out[..input.len()].copy_from_slice(input);
            return input.len();
        }
        self.hist[self.hist_len..self.hist_len + input.len()].copy_from_slice(input);
        self.hist_len += input.len();
        let mut written = 0;
        // Output at upsampled index `pos` needs input index floor(pos/up) to exist.
        while self.pos / self.up < self.hist_len && written < out.len() {
            let base = self.pos / self.up;
            let phase = self.pos % self.up;
            let c = &self.coeffs[phase];
            let mut acc = 0.0f32;
            for (j, &h) in c.iter().enumerate() {
                acc += h * self.hist[base - j];
            }
            out[written] = acc;
            written += 1;
            self.pos += self.down;
        }
        // Keep the last taps-1 samples needed by future outputs.
        let consumed = (self.pos / self.up).saturating_sub(self.taps - 1).min(self.hist_len);
        self.hist.copy_within(consumed..self.hist_len, 0);
        self.hist_len -= consumed;
        self.pos -= consumed * self.up;
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(f: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn run(r: &mut Resampler, x: &[f32], chunk: usize) -> Vec<f32> {
        let mut out = Vec::new();
        let mut buf = vec![0f32; r.max_out(chunk)];
        for c in x.chunks(chunk) {
            let n = r.process(c, &mut buf);
            out.extend_from_slice(&buf[..n]);
        }
        out
    }

    #[test]
    fn rates_and_lengths() {
        for (a, b) in [(48000, 16000), (32000, 48000), (40000, 48000), (48000, 48000)] {
            let mut r = Resampler::new(a, b, 480, 48);
            let y = run(&mut r, &vec![0.0; a as usize], 480);
            let expect = b as usize;
            assert!((y.len() as i64 - expect as i64).abs() <= 2, "{a}->{b}: {}", y.len());
        }
    }

    #[test]
    fn passband_tone_preserved() {
        let mut r = Resampler::new(40000, 48000, 400, 48);
        let x = tone(1000.0, 40000.0, 40000);
        let y = run(&mut r, &x, 400);
        let tail = &y[2000..y.len() - 2000];
        assert!((rms(tail) - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.01, "rms {}", rms(tail));
    }

    #[test]
    fn decimation_rejects_alias() {
        // 12 kHz at 48 kHz would alias to 4 kHz at 16 kHz; it must be strongly attenuated.
        let mut r = Resampler::new(48000, 16000, 480, 64);
        let y = run(&mut r, &tone(12000.0, 48000.0, 48000), 480);
        assert!(rms(&y[1000..]) < 0.01, "alias rms {}", rms(&y[1000..]));
    }

    #[test]
    fn odd_chunks_match_one_shot() {
        let x = tone(440.0, 48000.0, 9000);
        let mut a = Resampler::new(48000, 16000, 9000, 48);
        let mut b = Resampler::new(48000, 16000, 9000, 48);
        let one = run(&mut a, &x, 9000);
        let mut many = Vec::new();
        let mut buf = vec![0f32; b.max_out(9000)];
        let mut i = 0;
        for (k, step) in [1usize, 7, 160, 333, 480].iter().cycle().enumerate() {
            if i >= x.len() || k > 10_000 {
                break;
            }
            let end = (i + step).min(x.len());
            let n = b.process(&x[i..end], &mut buf);
            many.extend_from_slice(&buf[..n]);
            i = end;
        }
        assert_eq!(one.len(), many.len());
        for (p, q) in one.iter().zip(&many) {
            assert!((p - q).abs() < 1e-5);
        }
    }
}
