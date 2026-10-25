//! A/B recorder: synchronized `dry.wav` (raw input) and `wet.wav` (final output).
//!
//! The capture thread streams raw samples and notes the capture index where the take started.
//! The DSP thread knows which capture time every output sample came from, so it places wet
//! samples on the dry timeline itself (padding dropped blocks, skipping overlap). Sample `i` of
//! both files therefore belongs to the same moment, and the writer thread only appends.

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::Serialize;

use crate::pipeline::RATE;

const RING: usize = 4 * RATE;
const IDLE: u8 = 0;
const RECORDING: u8 = 1;
const STOPPING: u8 = 2;

#[derive(Debug, Clone, Serialize)]
pub struct Take {
    pub dry: String,
    pub wet: String,
    pub seconds: f32,
}

struct Job {
    thread: JoinHandle<Result<Take, String>>,
}

pub struct RecorderHub {
    state: AtomicU8,
    /// Bumped per take so the audio sides reset their cursors.
    pub take: AtomicU64,
    /// Raw-capture index of the first dry sample, -1 until the capture side starts the take.
    pub dry_start: AtomicI64,
    /// Dry samples captured so far (final once STOPPING).
    pub dry_len: AtomicU64,
    rings: Mutex<Option<(rtrb::Consumer<f32>, rtrb::Consumer<f32>)>>,
    job: Mutex<Option<Job>>,
    started: Mutex<Option<Instant>>,
}

impl Default for RecorderHub {
    fn default() -> Self {
        Self {
            state: AtomicU8::new(IDLE),
            take: AtomicU64::new(0),
            dry_start: AtomicI64::new(-1),
            dry_len: AtomicU64::new(0),
            rings: Mutex::new(None),
            job: Mutex::new(None),
            started: Mutex::new(None),
        }
    }
}

impl RecorderHub {
    pub fn recording(&self) -> bool {
        self.state.load(Ordering::Acquire) != IDLE
    }

    pub fn elapsed_s(&self) -> f32 {
        self.started.lock().map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0)
    }

    /// New rings for a pipeline. A take in progress on the old pipeline is finished first.
    pub fn attach(self: &Arc<Self>) -> (DrySide, WetSide) {
        if self.recording() {
            let _ = self.stop();
        }
        let (dp, dc) = rtrb::RingBuffer::new(RING);
        let (wp, wc) = rtrb::RingBuffer::new(RING);
        *self.rings.lock() = Some((dc, wc));
        (DrySide { tx: dp, take: 0, n: 0 }, WetSide { tx: wp, take: 0, cursor: 0 })
    }

    /// Start a take writing `<stem> dry.wav` / `<stem> wet.wav` into `dir`.
    pub fn start(self: &Arc<Self>, dir: PathBuf, stem: String) -> Result<(), String> {
        let mut job = self.job.lock();
        if job.is_some() || self.recording() {
            return Err("already recording".into());
        }
        let (mut dry_rx, mut wet_rx) = self.rings.lock().take().ok_or("start the voice changer first")?;
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let dry_path = dir.join(format!("{stem} dry.wav"));
        let wet_path = dir.join(format!("{stem} wet.wav"));
        let spec = hound::WavSpec { channels: 1, sample_rate: RATE as u32, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
        let mut dry = hound::WavWriter::create(&dry_path, spec).map_err(|e| e.to_string())?;
        let mut wet = hound::WavWriter::create(&wet_path, spec).map_err(|e| e.to_string())?;
        while dry_rx.pop().is_ok() {}
        while wet_rx.pop().is_ok() {}
        self.dry_start.store(-1, Ordering::Relaxed);
        self.dry_len.store(0, Ordering::Relaxed);
        self.take.fetch_add(1, Ordering::Relaxed);
        self.state.store(RECORDING, Ordering::Release);
        *self.started.lock() = Some(Instant::now());
        let hub = self.clone();
        let thread = std::thread::Builder::new()
            .name("resona-recorder".into())
            .spawn(move || {
                let mut n_dry = 0u64;
                let mut n_wet = 0u64;
                let mut stop_at: Option<Instant> = None;
                let drain = |rx: &mut rtrb::Consumer<f32>, w: &mut hound::WavWriter<_>, n: &mut u64| -> Result<(), String> {
                    let avail = rx.slots();
                    if avail == 0 {
                        return Ok(());
                    }
                    let chunk = rx.read_chunk(avail).map_err(|e| e.to_string())?;
                    let (a, b) = chunk.as_slices();
                    for &v in a.iter().chain(b) {
                        w.write_sample(v).map_err(|e| e.to_string())?;
                    }
                    chunk.commit_all();
                    *n += avail as u64;
                    Ok(())
                };
                let res = (|| -> Result<(), String> {
                    loop {
                        drain(&mut dry_rx, &mut dry, &mut n_dry)?;
                        drain(&mut wet_rx, &mut wet, &mut n_wet)?;
                        if hub.state.load(Ordering::Acquire) == STOPPING {
                            let t = *stop_at.get_or_insert_with(Instant::now);
                            let target = hub.dry_len.load(Ordering::Acquire);
                            // The wet side lags by the conversion latency; give it time to catch up.
                            if (n_dry >= target && n_wet >= target) || t.elapsed() > Duration::from_millis(1500) {
                                break;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    for _ in n_wet..n_dry {
                        wet.write_sample(0.0f32).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                })();
                *hub.rings.lock() = Some((dry_rx, wet_rx));
                hub.state.store(IDLE, Ordering::Release);
                res?;
                dry.finalize().map_err(|e| e.to_string())?;
                wet.finalize().map_err(|e| e.to_string())?;
                Ok(Take {
                    dry: dry_path.display().to_string(),
                    wet: wet_path.display().to_string(),
                    seconds: n_dry as f32 / RATE as f32,
                })
            })
            .map_err(|e| e.to_string())?;
        *job = Some(Job { thread });
        Ok(())
    }

    /// Finish the take; waits for the wet side to catch up (≤ 1.5 s).
    pub fn stop(&self) -> Result<Take, String> {
        let job = self.job.lock().take().ok_or("not recording")?;
        if self.state.load(Ordering::Acquire) == RECORDING {
            self.state.store(STOPPING, Ordering::Release);
        }
        *self.started.lock() = None;
        job.thread.join().map_err(|_| "recorder thread panicked".to_string())?
    }
}

/// Write `n` samples produced by `f(i)` (caller checked there is room).
fn push_with(tx: &mut rtrb::Producer<f32>, n: usize, mut f: impl FnMut(usize) -> f32) {
    let mut chunk = tx.write_chunk(n).expect("slots checked");
    let (a, b) = chunk.as_mut_slices();
    for (i, d) in a.iter_mut().chain(b.iter_mut()).enumerate() {
        *d = f(i);
    }
    chunk.commit_all();
}

/// Capture-thread half: raw samples (after file playback substitution, before effects).
pub struct DrySide {
    tx: rtrb::Producer<f32>,
    take: u64,
    n: u64,
}

impl DrySide {
    /// `t` is the raw-capture index of `x[0]`.
    pub fn push(&mut self, hub: &RecorderHub, t: u64, x: &[f32]) {
        if hub.state.load(Ordering::Acquire) != RECORDING {
            return;
        }
        let take = hub.take.load(Ordering::Relaxed);
        if take != self.take {
            self.take = take;
            self.n = 0;
            hub.dry_start.store(t as i64, Ordering::Release);
        }
        // A full ring means the disk stalled for seconds; drop rather than block.
        let n = x.len().min(self.tx.slots());
        if n > 0 {
            push_with(&mut self.tx, n, |i| x[i]);
        }
        self.n += x.len() as u64;
        hub.dry_len.store(self.n, Ordering::Release);
    }
}

/// DSP-thread half: final output, placed on the dry timeline.
pub struct WetSide {
    tx: rtrb::Producer<f32>,
    take: u64,
    cursor: u64,
}

impl WetSide {
    /// `t0` is the raw-capture time of `y[0]` (already corrected for every delay in the path).
    pub fn push(&mut self, hub: &RecorderHub, t0: i64, y: &[f32]) {
        let st = hub.state.load(Ordering::Acquire);
        if st == IDLE {
            return;
        }
        let start = hub.dry_start.load(Ordering::Acquire);
        if start < 0 {
            return;
        }
        let take = hub.take.load(Ordering::Relaxed);
        if take != self.take {
            self.take = take;
            self.cursor = 0;
        }
        // Wet is derived from dry, so it can never be ahead of it; this also ends it with the take.
        let limit = hub.dry_len.load(Ordering::Acquire);
        let mut fi = t0 - start; // file index of y[0]
        let mut y = y;
        if fi < self.cursor as i64 {
            let skip = ((self.cursor as i64 - fi) as usize).min(y.len());
            y = &y[skip..];
            fi += skip as i64;
        }
        if y.is_empty() {
            return;
        }
        // Gap (dropped blocks, or the take started mid-block): pad with silence.
        let gap = (fi as u64).saturating_sub(self.cursor).min(limit.saturating_sub(self.cursor)) as usize;
        let room = self.tx.slots();
        let gap = gap.min(room);
        if gap > 0 {
            push_with(&mut self.tx, gap, |_| 0.0);
            self.cursor += gap as u64;
        }
        let n = y.len().min(self.tx.slots()).min(limit.saturating_sub(self.cursor) as usize);
        if n > 0 {
            push_with(&mut self.tx, n, |i| y[i]);
            self.cursor += n as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(p: &str) -> Vec<f32> {
        hound::WavReader::open(p).unwrap().samples::<f32>().map(|s| s.unwrap()).collect()
    }

    #[test]
    fn wet_lands_on_the_dry_timeline() {
        let hub = Arc::new(RecorderHub::default());
        let (mut dry, mut wet) = hub.attach();
        let dir = std::env::temp_dir().join(format!("resona-rec-{}", std::process::id()));
        hub.start(dir.clone(), "t".into()).unwrap();
        // Raw capture: a ramp, starting mid-stream at capture index 1000.
        let x: Vec<f32> = (0..9600).map(|i| (1000 + i) as f32).collect();
        for (k, c) in x.chunks(480).enumerate() {
            dry.push(&hub, 1000 + (k * 480) as u64, c);
        }
        // Wet: the same timeline delivered in blocks that start before the take, with one block
        // missing and arbitrary block sizes.
        let mut t = 0i64;
        let mut k = 0;
        while t < 1000 + 9600 + 2000 {
            let n = 1920;
            if k != 3 {
                let y: Vec<f32> = (0..n).map(|j| (t + j as i64) as f32).collect();
                wet.push(&hub, t, &y);
            }
            t += n as i64;
            k += 1;
        }
        let take = hub.stop().unwrap();
        let d = read(&take.dry);
        let w = read(&take.wet);
        assert_eq!(d.len(), 9600);
        assert_eq!(w.len(), 9600);
        for i in 0..9600 {
            let missing = (3 * 1920..4 * 1920).contains(&(1000 + i as i64));
            if missing {
                assert_eq!(w[i], 0.0);
            } else {
                assert_eq!(w[i], d[i], "sample {i}");
            }
        }
        let _ = std::fs::remove_dir_all(dir);
        assert!(!hub.recording());
    }
}
