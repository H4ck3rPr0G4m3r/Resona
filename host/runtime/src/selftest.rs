//! Latency self-test: measured mouth-to-output delay through the whole chain.
//!
//! The capture thread replaces the microphone with a few short voiced bursts (a 150 Hz
//! vowel-like buzz, which the voice model turns into voice) and records the QPC time of each
//! burst's first sample. A second capture stream listens to the output, by default as WASAPI
//! loopback of the output device, or on any recording endpoint (e.g. the virtual cable's
//! "output" side, to include its buffering). Both packet clocks are QPC, so the delay is
//! simply `onset time on the listen stream - burst time on the mic stream`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::Serialize;

use crate::pipeline::RATE;

pub const BURSTS: usize = 5;
pub const PERIOD: usize = RATE; // one burst per second
pub const LEAD_IN: usize = 600 * RATE / 1000;
const BURST_LEN: usize = 300 * RATE / 1000;
const ATTACK: usize = 5 * RATE / 1000;
const RELEASE: usize = 20 * RATE / 1000;
const F0_PERIOD: usize = 320; // 150 Hz
const HNS_PER_SAMPLE: f64 = 1e7 / RATE as f64;

pub struct SelfTestHub {
    pub active: AtomicBool,
    pub done: AtomicBool,
    pub burst_qpc: [AtomicU64; BURSTS],
}

impl Default for SelfTestHub {
    fn default() -> Self {
        Self { active: AtomicBool::new(false), done: AtomicBool::new(false), burst_qpc: Default::default() }
    }
}

impl SelfTestHub {
    pub fn arm(&self) {
        for t in &self.burst_qpc {
            t.store(0, Ordering::Relaxed);
        }
        self.done.store(false, Ordering::Relaxed);
        self.active.store(true, Ordering::Release);
    }
}

/// One period of the test vowel: harmonics with a falling spectrum and two formant bumps.
fn vowel_table() -> Vec<f32> {
    let mut t = vec![0.0f32; F0_PERIOD];
    for k in 1..=30usize {
        let f = 150.0 * k as f32;
        let formant = |fc: f32, bw: f32| 1.0 / (1.0 + ((f - fc) / bw).powi(2));
        let a = (1.0 / k as f32) * (0.3 + 2.0 * formant(700.0, 150.0) + 1.2 * formant(1200.0, 200.0));
        for (i, v) in t.iter_mut().enumerate() {
            *v += a * (2.0 * std::f32::consts::PI * k as f32 * i as f32 / F0_PERIOD as f32).sin();
        }
    }
    let peak = t.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    t.iter_mut().for_each(|v| *v *= 0.35 / peak);
    t
}

/// Capture-thread half: generates the bursts.
pub struct Injector {
    table: Vec<f32>,
    pos: Option<usize>,
}

impl Default for Injector {
    fn default() -> Self {
        Self { table: vowel_table(), pos: None }
    }
}

impl Injector {
    /// Overwrite `buf` (captured at QPC `qpc`) with the test signal while a test is armed.
    pub fn process(&mut self, hub: &SelfTestHub, buf: &mut [f32], qpc: u64) {
        if !hub.active.load(Ordering::Acquire) {
            self.pos = None;
            return;
        }
        let pos = self.pos.get_or_insert(0);
        for (i, v) in buf.iter_mut().enumerate() {
            // Silence first, so audio already in flight drains before the first burst.
            let Some(t) = (*pos + i).checked_sub(LEAD_IN) else {
                *v = 0.0;
                continue;
            };
            let (b, w) = (t / PERIOD, t % PERIOD);
            *v = if b < BURSTS && w < BURST_LEN {
                if w == 0 && qpc != 0 {
                    hub.burst_qpc[b].store(qpc + (i as f64 * HNS_PER_SAMPLE) as u64, Ordering::Relaxed);
                }
                let env = (w as f32 / ATTACK as f32).min(1.0).min((BURST_LEN - w) as f32 / RELEASE as f32);
                self.table[w % F0_PERIOD] * env
            } else {
                0.0 // the mic stays muted for the whole test so room noise can't trigger onsets
            };
        }
        *pos += buf.len();
        if *pos >= LEAD_IN + BURSTS * PERIOD {
            hub.active.store(false, Ordering::Release);
            hub.done.store(true, Ordering::Release);
        }
    }
}

/// What the listen stream recorded: samples plus the QPC time of each packet's first sample.
#[derive(Default)]
pub struct Capture {
    pub samples: Vec<f32>,
    pub packets: Vec<(usize, u64)>,
}

impl Capture {
    pub fn with_capacity(n: usize) -> Self {
        Self { samples: Vec::with_capacity(n), packets: Vec::with_capacity(n / 64) }
    }

    /// Append a packet without reallocating (drops what does not fit).
    pub fn push(&mut self, x: &[f32], qpc: u64) {
        let room = self.samples.capacity() - self.samples.len();
        let n = x.len().min(room);
        if n == 0 || self.packets.len() == self.packets.capacity() {
            return;
        }
        self.packets.push((self.samples.len(), qpc));
        self.samples.extend_from_slice(&x[..n]);
    }

    fn qpc_at(&self, idx: usize) -> Option<f64> {
        let p = self.packets.partition_point(|&(off, _)| off <= idx).checked_sub(1)?;
        let (off, q) = self.packets[p];
        (q != 0).then(|| q as f64 + (idx - off) as f64 * HNS_PER_SAMPLE)
    }

    fn index_at(&self, qpc: f64) -> Option<usize> {
        let p = self.packets.partition_point(|&(_, q)| (q as f64) <= qpc).checked_sub(1)?;
        let (off, q) = self.packets[p];
        Some(off + ((qpc - q as f64) / HNS_PER_SAMPLE).max(0.0) as usize)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SelfTestResult {
    /// Median measured delay, burst start → output onset.
    pub measured_ms: f32,
    pub runs_ms: Vec<f32>,
    pub spread_ms: f32,
    /// The latency bar's estimate at the time of the test, for comparison.
    pub estimate_ms: Option<f32>,
    pub listen: String,
}

/// Find each burst's onset in the listen capture. Envelope: 1 ms RMS; onset = first crossing of
/// 25 % of that burst's envelope peak, refined to the sample where |x| first passes half of it.
pub fn analyze(cap: &Capture, bursts: &[u64]) -> Result<Vec<f32>, String> {
    const HOP: usize = RATE / 1000;
    let x = &cap.samples;
    let mut runs = Vec::new();
    for &bq in bursts.iter().filter(|&&q| q != 0) {
        let Some(start) = cap.index_at(bq as f64) else { continue };
        let end = (start + 900 * RATE / 1000).min(x.len());
        if end <= start + HOP * 10 {
            continue;
        }
        let env: Vec<f32> = x[start..end].chunks(HOP).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()).collect();
        let peak = env.iter().cloned().fold(0.0, f32::max);
        let floor = {
            let mut e = env.clone();
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            e[e.len() / 10]
        };
        if peak < 1e-3 || peak < floor * 8.0 {
            continue; // nothing came through for this burst
        }
        // Gate on a robust 25 % crossing, then walk back sample by sample to the first loud
        // sample of that contiguous active stretch (quiet gaps under 2 ms are zero crossings).
        // 5x the floor RMS keeps noise peaks from extending the stretch backwards.
        let Some(k) = env.iter().position(|&e| e >= peak * 0.25) else { continue };
        let low = (peak * 0.03).max(floor * 5.0);
        let from = start + (k + 1) * HOP;
        let stop = from.saturating_sub(30 * HOP).max(start);
        let mut onset = from.min(end - 1);
        let mut quiet = 0;
        for i in (stop..from.min(end)).rev() {
            if x[i].abs() >= low {
                onset = i;
                quiet = 0;
            } else {
                quiet += 1;
                if quiet >= 2 * HOP {
                    break;
                }
            }
        }
        if let Some(q) = cap.qpc_at(onset) {
            runs.push(((q - bq as f64) / 1e4) as f32);
        }
    }
    if runs.len() < 3 {
        return Err(format!(
            "only {} of {} test bursts came back on the listen device — is it the right device, and is the output unmuted?",
            runs.len(),
            bursts.len()
        ));
    }
    Ok(runs)
}

pub fn summarize(mut runs: Vec<f32>, estimate_ms: Option<f32>, listen: String) -> SelfTestResult {
    let order = runs.clone();
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = runs[runs.len() / 2];
    SelfTestResult { measured_ms: median, spread_ms: runs[runs.len() - 1] - runs[0], runs_ms: order, estimate_ms, listen }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_a_known_delay_across_packet_clocks() {
        let hub = SelfTestHub::default();
        hub.arm();
        let mut inj = Injector::default();
        // Mic: 480-sample packets with QPC starting at 1 s.
        let mut mic = Vec::new();
        let mut q = 10_000_000u64;
        while hub.active.load(Ordering::Acquire) {
            let mut buf = vec![0.0; 480];
            inj.process(&hub, &mut buf, q);
            mic.extend_from_slice(&buf);
            q += (480.0 * HNS_PER_SAMPLE) as u64;
        }
        assert!(hub.done.load(Ordering::Acquire));
        // Listen: the same signal delayed by 123.4 ms, with noise, in 441-sample packets on its own clock.
        let delay = (0.1234 * RATE as f64) as usize;
        let mut cap = Capture::with_capacity(mic.len() + delay + 48_000);
        let lq0 = 10_000_000u64 - 2_000_000; // listen stream started 200 ms earlier
        let pre = 2 * RATE / 10;
        let mut seed = 1u32;
        let listen: Vec<f32> = (0..pre + delay + mic.len())
            .map(|i| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let n = (seed >> 9) as f32 / (1 << 23) as f32 - 0.5;
                let s = i.checked_sub(pre + delay).map(|j| mic[j] * 0.7).unwrap_or(0.0);
                s + n * 0.002
            })
            .collect();
        for (k, c) in listen.chunks(441).enumerate() {
            cap.push(c, lq0 + (k as f64 * 441.0 * HNS_PER_SAMPLE) as u64);
        }
        let bursts: Vec<u64> = hub.burst_qpc.iter().map(|t| t.load(Ordering::Relaxed)).collect();
        let runs = analyze(&cap, &bursts).unwrap();
        assert_eq!(runs.len(), BURSTS);
        for r in &runs {
            assert!((r - 123.4).abs() < 0.5, "measured {r} ms");
        }
    }

    #[test]
    fn silence_on_the_listen_side_is_an_error() {
        let mut cap = Capture::with_capacity(6 * RATE);
        cap.push(&vec![0.0; 6 * RATE], 1);
        let err = analyze(&cap, &[10_000_000; BURSTS]).unwrap_err();
        assert!(err.contains("0 of 5"), "{err}");
    }
}
