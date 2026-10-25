//! Real-time audio pipeline.
//!
//! ```text
//! capture cb ──► CaptureSide: file player, self-test, VAD, recorder (dry), input chain,
//!                              meter, 48k→16k, block ──► SHM in-ring ──► engine
//!            └─► dry FIFO (48k) ─────────────┐
//! engine ──► SHM out-ring ──► DSP thread: SOLA, output chain, gate, limiter, recorder (wet), gain ──► out FIFO ──► RenderSide ──► VAC
//!                                         └─ passthrough/mute when engine is down    └► mon FIFO ──► RenderSide ──► headphones
//! ```
//!
//! `CaptureSide` and `RenderSide` run on device threads and never allocate or lock. The DSP
//! thread is woken by the engine's output event.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use resona_dsp::fx::denoise::VadTracker;
use resona_dsp::fx::{live_chain, Effect, Limiter, LiveChain};
use resona_dsp::{AtomicF32, Meter, MeterState, Resampler, Sola};

use crate::fx::FxHub;
use crate::player::{PlayerHub, PlayerSide};
use crate::recorder::{DrySide, RecorderHub, WetSide};
use crate::selftest::{Injector, SelfTestHub};
use resona_shm_ipc::{NamedEvent, ShmRing};
use serde::{Deserialize, Serialize};

pub const RATE: usize = 48_000;
const FADE: usize = 240; // 5 ms at 48 kHz
const CAPTURE_MAX_CHUNK: usize = 4096;
const CAPTURE_TAPS: usize = 64;

/// Engine stream layout (mirrors the Python `StreamLayout`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layout {
    pub model_sr: usize,
    pub block_in: usize,
    pub block_out: usize,
    pub out_len: usize,
    pub crossfade: usize,
    pub search: usize,
    pub nominal_offset: usize,
}

impl Layout {
    pub fn block_ms(&self) -> f32 {
        self.block_in as f32 / 16.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DownMode {
    /// Silence while the engine is not producing audio.
    #[default]
    Mute,
    /// Pass the dry microphone through while the engine is not producing audio.
    Passthrough,
}

/// State shared between the device threads, the DSP thread and the runtime.
pub struct Shared {
    pub in_meter: Meter,
    pub out_meter: Meter,
    pub underruns: AtomicU64,
    pub blocks_sent: AtomicU64,
    pub blocks_received: AtomicU64,
    pub ring_full: AtomicU64,
    pub captured: AtomicU64,
    pub rendered: AtomicU64,
    pub fifo_samples: AtomicUsize,
    /// Current (adaptive) render jitter target in 48 kHz samples.
    pub jitter_samples: AtomicUsize,
    pub engine_live: AtomicBool,
    pub down_mode: AtomicU8,
    pub muted: AtomicBool,
    pub output_gain: AtomicF32,
    pub monitor_gain: AtomicF32,
    pub monitor_tap: AtomicU8, // 0 = processed, 1 = dry
    pub sola_ncc: AtomicF32,
    pub stop: AtomicBool,
    pub fx: FxHub,
    pub player: PlayerHub,
    pub recorder: Arc<RecorderHub>,
    pub selftest: SelfTestHub,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            in_meter: Meter::default(),
            out_meter: Meter::default(),
            underruns: AtomicU64::new(0),
            blocks_sent: AtomicU64::new(0),
            blocks_received: AtomicU64::new(0),
            ring_full: AtomicU64::new(0),
            captured: AtomicU64::new(0),
            rendered: AtomicU64::new(0),
            fifo_samples: AtomicUsize::new(0),
            jitter_samples: AtomicUsize::new(0),
            engine_live: AtomicBool::new(false),
            down_mode: AtomicU8::new(0),
            muted: AtomicBool::new(false),
            output_gain: AtomicF32::new(1.0),
            monitor_gain: AtomicF32::new(1.0),
            monitor_tap: AtomicU8::new(0),
            sola_ncc: AtomicF32::new(1.0),
            stop: AtomicBool::new(false),
            fx: FxHub::default(),
            player: PlayerHub::default(),
            recorder: Arc::default(),
            selftest: SelfTestHub::default(),
        }
    }
}

impl Shared {
    pub fn down_mode(&self) -> DownMode {
        if self.down_mode.load(Ordering::Relaxed) == 1 {
            DownMode::Passthrough
        } else {
            DownMode::Mute
        }
    }
    pub fn set_down_mode(&self, m: DownMode) {
        self.down_mode.store(matches!(m, DownMode::Passthrough) as u8, Ordering::Relaxed);
    }
}

/// Engine-side transport handles (rings + wake events).
pub struct Transport {
    pub ring_in: ShmRing,
    pub ring_out: ShmRing,
    pub ev_in: NamedEvent,
    pub ev_out: NamedEvent,
}

impl Transport {
    pub const IN_SLOT: usize = 1600; // 100 ms at 16 kHz
    pub const OUT_SLOT: usize = 6144; // (100 + 10 + 10) ms at 48 kHz, rounded up
    pub const SLOTS: usize = 16;

    pub fn create(prefix: &str) -> resona_shm_ipc::Result<Self> {
        Ok(Self {
            ring_in: ShmRing::create(&format!("{prefix}-in"), Self::SLOTS, Self::IN_SLOT)?,
            ring_out: ShmRing::create(&format!("{prefix}-out"), Self::SLOTS, Self::OUT_SLOT)?,
            ev_in: NamedEvent::create(&format!("{prefix}-in-ev"))?,
            ev_out: NamedEvent::create(&format!("{prefix}-out-ev"))?,
        })
    }
}

fn push_slice(p: &mut rtrb::Producer<f32>, x: &[f32]) -> usize {
    let n = x.len().min(p.slots());
    if n == 0 {
        return 0;
    }
    let mut chunk = p.write_chunk(n).expect("slots checked");
    let (a, b) = chunk.as_mut_slices();
    a.copy_from_slice(&x[..a.len()]);
    b.copy_from_slice(&x[a.len()..n]);
    chunk.commit_all();
    n
}

fn pop_into(c: &mut rtrb::Consumer<f32>, out: &mut [f32]) -> usize {
    let n = out.len().min(c.slots());
    if n == 0 {
        return 0;
    }
    let chunk = c.read_chunk(n).expect("slots checked");
    let (a, b) = chunk.as_slices();
    out[..a.len()].copy_from_slice(a);
    out[a.len()..n].copy_from_slice(b);
    chunk.commit_all();
    n
}

fn discard(c: &mut rtrb::Consumer<f32>, n: usize) {
    let n = n.min(c.slots());
    if n > 0 {
        c.read_chunk(n).expect("slots checked").commit_all();
    }
}

// ------------------------------------------------------------------------------------------------

/// Device-thread side of capture.
pub struct CaptureSide {
    rs: Resampler,
    tmp: Vec<f32>,
    block: Vec<f32>,
    fill: usize,
    seq: u64,
    transport: Arc<Transport>,
    dry: rtrb::Producer<f32>,
    meter: MeterState,
    shared: Arc<Shared>,
    fx: LiveChain,
    vad: VadTracker,
    work: Vec<f32>,
    player: PlayerSide,
    rec: DrySide,
    injector: Injector,
    /// Raw-capture samples so far (the timeline the gate and recorder use).
    raw_t: u64,
}

impl CaptureSide {
    pub fn process(&mut self, x: &[f32]) {
        self.process_timed(x, 0);
    }

    /// `qpc`: capture time of `x[0]` in 100 ns QPC units (0 = unknown).
    pub fn process_timed(&mut self, x: &[f32], qpc: u64) {
        const HNS: f64 = 1e7 / RATE as f64;
        for (k, raw) in x.chunks(CAPTURE_MAX_CHUNK).enumerate() {
            let n = raw.len();
            let sh = &*self.shared;
            let w = &mut self.work[..n];
            w.copy_from_slice(raw);
            self.player.process(&sh.player, w);
            let q = if qpc == 0 { 0 } else { qpc + ((k * CAPTURE_MAX_CHUNK) as f64 * HNS) as u64 };
            self.injector.process(&sh.selftest, w, q);
            self.vad.feed(w, |p| sh.fx.push_vad(p));
            self.rec.push(&sh.recorder, self.raw_t, w);
            self.raw_t += n as u64;
            self.fx.process(&mut self.work[..n]);
            self.shared.fx.input_latency.store(self.fx.latency(), Ordering::Relaxed);
            let mut buf = std::mem::take(&mut self.work);
            self.process_chunk(&buf[..n]);
            std::mem::swap(&mut self.work, &mut buf);
        }
    }

    fn process_chunk(&mut self, x: &[f32]) {
        self.meter.process(x, &self.shared.in_meter);
        self.shared.captured.fetch_add(x.len() as u64, Ordering::Relaxed);
        push_slice(&mut self.dry, x);
        for chunk in x.chunks(CAPTURE_MAX_CHUNK) {
            let n = self.rs.process(chunk, &mut self.tmp);
            let mut i = 0;
            while i < n {
                let take = (self.block.len() - self.fill).min(n - i);
                self.block[self.fill..self.fill + take].copy_from_slice(&self.tmp[i..i + take]);
                self.fill += take;
                i += take;
                if self.fill == self.block.len() {
                    self.fill = 0;
                    if self.transport.ring_in.push(self.seq, 0, &self.block).is_ok() {
                        self.shared.blocks_sent.fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.shared.ring_full.fetch_add(1, Ordering::Relaxed);
                    }
                    self.seq += 1;
                    self.transport.ev_in.set();
                }
            }
        }
    }
}

/// Device-thread side of rendering: underrun concealment plus a latency controller.
///
/// Blocks arrive in bursts of one engine block, so the FIFO level saw-tooths between a floor
/// and floor + block. The controller watches the floor (minimum level over a 1 s window); if it
/// sits above the jitter target — after a start-up burst or from slow clock drift — the excess
/// is skipped with a short crossfade, so latency never creeps up.
///
/// The target itself adapts: an underrun raises it by `ADAPT_UP` (up to `adapt_max` above the
/// configured minimum), and after `RELAX_AFTER` without one it eases back, so a GPU that spikes now
/// and then trades some latency for no dropouts instead of glitching repeatedly.
pub struct RenderSide {
    fifo: rtrb::Consumer<f32>,
    starved: bool,
    fade_in: usize,
    last: f32,
    prefill: usize,
    base_prefill: usize,
    adapt_max: usize,
    clean: usize,
    window_min: usize,
    window_pos: usize,
    scratch: Vec<f32>,
    pending: [f32; FADE],
    pending_len: usize,
    pending_pos: usize,
    gain: Option<Arc<Shared>>, // monitor sides read monitor_gain
    shared: Arc<Shared>,
    count_underruns: bool,
}

const CONTROL_WINDOW: usize = RATE; // 1 s
const CONTROL_SLACK: usize = 3 * RATE / 1000; // tolerate 3 ms above target
const ADAPT_UP: usize = 5 * RATE / 1000;
/// Least headroom the target may grow by. With larger blocks it may grow by one block: a GPU
/// shared with a game delays blocks by tens of ms, and a 40 ms cap then left a 100 ms block
/// dropping out over and over instead of settling.
const ADAPT_MAX: usize = 40 * RATE / 1000;
const ADAPT_DOWN: usize = RATE / 1000;
const RELAX_AFTER: usize = 20 * RATE; // underrun-free time before easing the target down
const RELAX_EVERY: usize = 3 * RATE;

impl RenderSide {
    fn new(fifo: rtrb::Consumer<f32>, prefill: usize, adapt_max: usize, gain: Option<Arc<Shared>>, shared: Arc<Shared>, count_underruns: bool) -> Self {
        Self {
            fifo,
            starved: true,
            fade_in: 0,
            last: 0.0,
            prefill: prefill.max(1),
            base_prefill: prefill.max(1),
            adapt_max,
            clean: 0,
            window_min: usize::MAX,
            window_pos: 0,
            scratch: vec![0.0; RATE + FADE],
            pending: [0.0; FADE],
            pending_len: 0,
            pending_pos: 0,
            gain,
            shared,
            count_underruns,
        }
    }

    pub fn process(&mut self, out: &mut [f32]) {
        let avail = self.fifo.slots();
        if self.count_underruns {
            self.shared.fifo_samples.store(avail, Ordering::Relaxed);
            self.shared.jitter_samples.store(self.prefill, Ordering::Relaxed);
            self.shared.rendered.fetch_add(out.len() as u64, Ordering::Relaxed);
        }
        if self.starved {
            if avail < self.prefill {
                self.decay_last(out);
                return;
            }
            self.starved = false;
            self.fade_in = 0;
            self.window_min = usize::MAX;
            self.window_pos = 0;
        }
        self.control(avail, out.len());

        // Crossfade remainder from a latency correction goes first.
        let mut w = 0;
        while self.pending_pos < self.pending_len && w < out.len() {
            out[w] = self.pending[self.pending_pos];
            self.pending_pos += 1;
            w += 1;
        }
        let n = w + pop_into(&mut self.fifo, &mut out[w..]);
        if n < out.len() {
            // Underrun: conceal with a short decay from the last good sample, then re-prefill.
            if self.count_underruns {
                self.shared.underruns.fetch_add(1, Ordering::Relaxed);
            }
            self.prefill = (self.prefill + ADAPT_UP).min(self.base_prefill + self.adapt_max);
            self.clean = 0;
            if n > 0 {
                self.last = out[n - 1];
            }
            self.decay_last(&mut out[n..]);
            self.starved = true;
        }
        for v in out[w..n].iter_mut() {
            if self.fade_in < FADE {
                *v *= self.fade_in as f32 / FADE as f32;
                self.fade_in += 1;
            }
        }
        if let Some(s) = &self.gain {
            let g = s.monitor_gain.load();
            out.iter_mut().for_each(|v| *v *= g);
        }
        if n > 0 && !self.starved {
            self.last = out[n - 1];
        }
    }

    /// Track the FIFO floor; once per window, skip any persistent excess above the target.
    fn control(&mut self, avail: usize, period: usize) {
        self.clean += period;
        if self.clean >= RELAX_AFTER {
            self.prefill = self.prefill.saturating_sub(ADAPT_DOWN).max(self.base_prefill);
            self.clean = RELAX_AFTER - RELAX_EVERY;
        }
        self.window_min = self.window_min.min(avail);
        self.window_pos += period;
        if self.window_pos < CONTROL_WINDOW {
            return;
        }
        let floor = self.window_min;
        self.window_min = usize::MAX;
        self.window_pos = 0;
        if floor <= self.prefill + CONTROL_SLACK || self.pending_pos < self.pending_len {
            return;
        }
        let excess = (floor - self.prefill).min(self.scratch.len() - FADE);
        let need = excess + FADE;
        if self.fifo.slots() < need {
            return;
        }
        let got = pop_into(&mut self.fifo, &mut self.scratch[..need]);
        debug_assert_eq!(got, need);
        // Equal-gain crossfade from "where we were" to "where we skip to".
        for i in 0..FADE {
            let t = (i as f32 + 0.5) / FADE as f32;
            let fi = (t * std::f32::consts::FRAC_PI_2).sin().powi(2);
            self.pending[i] = self.scratch[i] * (1.0 - fi) + self.scratch[excess + i] * fi;
        }
        self.pending_len = FADE;
        self.pending_pos = 0;
    }

    fn decay_last(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            self.last *= 0.98; // ~-40 dB in 5 ms
            *v = self.last;
        }
    }
}

// ------------------------------------------------------------------------------------------------

pub enum DspCmd {
    Layout(Layout),
    Prime,
}

struct DspThread {
    transport: Arc<Transport>,
    shared: Arc<Shared>,
    layout: Layout,
    sola: Sola,
    rs: Resampler,
    y: Vec<f32>,
    block: Vec<f32>,
    up: Vec<f32>,
    dry_buf: Vec<f32>,
    out: rtrb::Producer<f32>,
    mon: Option<rtrb::Producer<f32>>,
    dry: rtrb::Consumer<f32>,
    cmds: Receiver<DspCmd>,
    jitter: usize,
    primed: bool,
    mute_gain: f32,
    fx: LiveChain,
    limiter: Limiter,
    gate_g: f32,
    gate_held: usize,
    rec: WetSide,
}

impl DspThread {
    fn new_resampler(layout: &Layout) -> Resampler {
        Resampler::new(layout.model_sr as u32, RATE as u32, layout.block_out.max(1), 48)
    }

    fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
        self.sola = Sola::new(layout.crossfade, layout.search);
        self.rs = Self::new_resampler(&layout);
        self.block = vec![0.0; layout.block_out];
        self.up = vec![0.0; self.rs.max_out(layout.block_out).max(layout.block_out)];
        self.primed = false;
    }

    fn push_out(&mut self, n: usize, dry: bool) {
        let g = self.shared.output_gain.load();
        let target_mute = if self.shared.muted.load(Ordering::Relaxed) { 0.0 } else { 1.0 };
        let step = 1.0 / FADE as f32;
        let buf = if dry { &mut self.dry_buf[..n] } else { &mut self.up[..n] };
        for v in buf.iter_mut() {
            self.mute_gain += (target_mute - self.mute_gain).clamp(-step, step);
            *v *= g * self.mute_gain;
        }
        push_slice(&mut self.out, buf);
        if let Some(m) = &mut self.mon {
            let tap_dry = self.shared.monitor_tap.load(Ordering::Relaxed) == 1;
            if !tap_dry || dry {
                push_slice(m, buf);
            }
        }
    }

    /// Fixed delay (48 kHz samples) between capture and the DSP thread's output, beyond the
    /// block/lookahead framing: the 48→16 kHz filter, SOLA's nominal offset and the upsampler.
    fn path_delay(&self) -> i64 {
        let model = self.layout.model_sr as f32;
        let capture = (CAPTURE_TAPS - 1) as f32 / 2.0;
        let sola = self.layout.nominal_offset as f32 * RATE as f32 / model;
        let up = if self.layout.model_sr == RATE { 0.0 } else { self.rs.delay_in() * RATE as f32 / model };
        (capture + sola + up).round() as i64
    }

    /// Output chain, voice gate and limiter on the freshly rendered block `seq` (`up[..n]`).
    fn post(&mut self, seq: u64, n: usize) {
        let path_delay = self.path_delay();
        let buf = &mut self.up[..n];
        self.fx.process(buf);
        let fx = &self.shared.fx;
        // Raw-capture time of this block's first output sample: block k covers input
        // [k*B - L, (k+1)*B - L); the input chain, both resamplers and SOLA's nominal offset
        // delay it further.
        let t0 = (seq as i64) * (self.layout.block_in * (RATE / 16_000)) as i64
            - fx.lookahead.load(Ordering::Relaxed) as i64
            - fx.input_latency.load(Ordering::Relaxed) as i64
            - path_delay;
        // The self-test's buzz is not speech to the VAD; it measures the path, not the gate.
        if fx.gate_on.load(Ordering::Relaxed) && !self.shared.selftest.active.load(Ordering::Relaxed) {
            let thr = fx.gate_threshold.load();
            let hold = fx.gate_hold.load(Ordering::Relaxed);
            let rel = fx.gate_release.load();
            let range = fx.gate_range.load();
            let att = 1.0 - (-1.0f32 / (0.002 * RATE as f32)).exp();
            let mut open = false;
            for (j, v) in buf.iter_mut().enumerate() {
                if j % 48 == 0 {
                    // ~1 ms resolution; look 20 ms ahead so the gate is open before a word starts.
                    open = fx.vad_at(t0 + j as i64, 2) >= thr;
                }
                if open {
                    self.gate_held = hold;
                } else {
                    self.gate_held = self.gate_held.saturating_sub(1);
                }
                let target = if open || self.gate_held > 0 { 1.0 } else { range };
                self.gate_g += (target - self.gate_g) * if target > self.gate_g { att } else { rel };
                *v *= self.gate_g;
            }
            fx.gate_open.store(self.gate_g > 0.5, Ordering::Relaxed);
        } else {
            self.gate_g = 1.0;
            fx.gate_open.store(true, Ordering::Relaxed);
        }
        self.limiter.enabled = fx.limiter_on.load(Ordering::Relaxed);
        self.limiter.set(0, fx.limiter_ceiling_db.load());
        self.limiter.process(buf);
        let out_lat = self.fx.latency() + Limiter::LATENCY;
        fx.output_latency.store(out_lat, Ordering::Relaxed);
        // Output-chain and limiter delay: what leaves now entered the chain `out_lat` ago.
        self.rec.push(&self.shared.recorder, t0 - out_lat as i64, buf);
    }

    fn run(mut self) {
        let mut out_meter = MeterState::new(RATE as f32);
        let mut quiet_ms = 0u32;
        while !self.shared.stop.load(Ordering::Acquire) {
            while let Ok(cmd) = self.cmds.try_recv() {
                match cmd {
                    DspCmd::Layout(l) => self.set_layout(l),
                    DspCmd::Prime => {
                        self.primed = false;
                        self.sola.reset();
                    }
                }
            }
            let woke = self.transport.ev_out.wait(10);
            let mut got = false;
            while let Some(h) = self.transport.ring_out.pop(&mut self.y) {
                got = true;
                self.shared.blocks_received.fetch_add(1, Ordering::Relaxed);
                if h.n < self.layout.out_len {
                    continue; // stale block from a previous layout
                }
                if !self.primed {
                    // Lead with a jitter cushion of silence so the next block arrives in time.
                    let zeros = vec![0.0; self.jitter];
                    push_slice(&mut self.out, &zeros);
                    if let Some(m) = &mut self.mon {
                        push_slice(m, &zeros);
                    }
                    self.primed = true;
                }
                self.sola.process(&self.y[..h.n], self.layout.block_out, &mut self.block);
                self.shared.sola_ncc.store(self.sola.last_ncc);
                let n = if self.layout.model_sr as usize == RATE {
                    // The engine already delivers 48 kHz (all voices do since M4).
                    self.up[..self.block.len()].copy_from_slice(&self.block);
                    self.block.len()
                } else {
                    self.rs.process(&self.block, &mut self.up)
                };
                self.post(h.seq, n);
                self.push_out(n, false);
                out_meter.process(&self.up[..n], &self.shared.out_meter);
            }
            let live = self.shared.engine_live.load(Ordering::Relaxed);
            if got || woke {
                quiet_ms = 0;
            } else {
                quiet_ms = quiet_ms.saturating_add(10);
            }
            // Dry FIFO: pass through when the engine is down (and configured), else keep it drained.
            let avail = self.dry.slots();
            if !live && self.shared.down_mode() == DownMode::Passthrough && quiet_ms >= 100 {
                let n = pop_into(&mut self.dry, &mut self.dry_buf[..avail.min(4096)]);
                if n > 0 {
                    self.push_out(n, true);
                    out_meter.process(&self.dry_buf[..n], &self.shared.out_meter);
                    self.primed = false;
                }
            } else if self.shared.monitor_tap.load(Ordering::Relaxed) == 1 && self.mon.is_some() {
                let n = pop_into(&mut self.dry, &mut self.dry_buf[..avail.min(4096)]);
                if let Some(m) = &mut self.mon {
                    push_slice(m, &self.dry_buf[..n]);
                }
            } else {
                discard(&mut self.dry, avail);
            }
            if !live && quiet_ms >= 100 && !got {
                out_meter.process(&[0.0; 16], &self.shared.out_meter);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------

pub struct PipelineParts {
    pub capture: CaptureSide,
    pub render: RenderSide,
    pub monitor: Option<RenderSide>,
    pub dsp: JoinHandle<()>,
    pub cmds: Sender<DspCmd>,
}

/// Build the pipeline pieces (device-independent, so tests can drive them directly).
pub fn build(transport: Arc<Transport>, shared: Arc<Shared>, layout: Layout, jitter_ms: f32, with_monitor: bool) -> PipelineParts {
    let fifo_cap = RATE; // 1 s
    let (out_p, out_c) = rtrb::RingBuffer::new(fifo_cap);
    let (mon_p, mon_c) = if with_monitor {
        let (p, c) = rtrb::RingBuffer::new(fifo_cap);
        (Some(p), Some(c))
    } else {
        (None, None)
    };
    let (dry_p, dry_c) = rtrb::RingBuffer::new(fifo_cap / 2);
    let (tx, rx) = crossbeam_channel::unbounded();
    let jitter = (jitter_ms * RATE as f32 / 1000.0) as usize;
    let adapt_max = ADAPT_MAX.max(layout.block_out * RATE / layout.model_sr.max(1));
    let render_side = |fifo, is_monitor: bool| {
        RenderSide::new(fifo, jitter, adapt_max, is_monitor.then(|| shared.clone()), shared.clone(), !is_monitor)
    };
    let render = render_side(out_c, false);
    let monitor = mon_c.map(|c| render_side(c, true));
    let (in_live, in_ctl) = live_chain();
    let (out_live, out_ctl) = live_chain();
    shared.fx.attach(in_ctl, out_ctl);
    let player = shared.player.attach();
    let (dry_rec, wet_rec) = shared.recorder.attach();
    let mut limiter = Limiter::new();
    limiter.reset();
    let capture = CaptureSide {
        rs: Resampler::new(RATE as u32, 16_000, CAPTURE_MAX_CHUNK, CAPTURE_TAPS),
        tmp: vec![0.0; CAPTURE_MAX_CHUNK / 3 + 8],
        block: vec![0.0; layout.block_in],
        fill: 0,
        seq: 0,
        transport: transport.clone(),
        dry: dry_p,
        meter: MeterState::new(RATE as f32),
        shared: shared.clone(),
        fx: in_live,
        vad: VadTracker::new(),
        work: vec![0.0; CAPTURE_MAX_CHUNK],
        player,
        rec: dry_rec,
        injector: Injector::default(),
        raw_t: 0,
    };
    let rs = DspThread::new_resampler(&layout);
    let up = vec![0.0; rs.max_out(layout.block_out).max(layout.block_out)];
    let dsp = DspThread {
        transport,
        shared,
        layout,
        sola: Sola::new(layout.crossfade, layout.search),
        rs,
        y: vec![0.0; Transport::OUT_SLOT],
        block: vec![0.0; layout.block_out],
        up,
        dry_buf: vec![0.0; 4096],
        out: out_p,
        mon: mon_p,
        dry: dry_c,
        cmds: rx,
        jitter,
        primed: false,
        mute_gain: 1.0,
        fx: out_live,
        limiter,
        gate_g: 1.0,
        gate_held: 0,
        rec: wet_rec,
    };
    let handle = std::thread::Builder::new()
        .name("resona-dsp".into())
        .spawn(move || {
            boost_thread_priority();
            dsp.run()
        })
        .expect("spawn dsp thread");
    PipelineParts { capture, render, monitor, dsp: handle, cmds: tx }
}

fn boost_thread_priority() {
    use windows::core::w;
    use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
    let mut idx = 0u32;
    // SAFETY: valid task name/out-pointer; the registration lives until the thread exits.
    let _ = unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut idx) };
}

/// Running pipeline bound to real devices.
pub struct Pipeline {
    capture: Option<resona_audio_io::CaptureStream>,
    render: Option<resona_audio_io::RenderStream>,
    monitor: Option<resona_audio_io::RenderStream>,
    dsp: Option<JoinHandle<()>>,
    pub cmds: Sender<DspCmd>,
    pub shared: Arc<Shared>,
    pub info: PipelineInfo,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineInfo {
    pub input: resona_audio_io::StreamInfo,
    pub output: resona_audio_io::StreamInfo,
    pub monitor: Option<resona_audio_io::StreamInfo>,
    pub jitter_ms: f32,
}

pub struct DeviceConfig {
    pub input: resona_audio_io::StreamConfig,
    pub output: resona_audio_io::StreamConfig,
    pub monitor: Option<resona_audio_io::StreamConfig>,
    pub jitter_ms: f32,
}

impl Pipeline {
    pub fn start<E>(transport: Arc<Transport>, shared: Arc<Shared>, layout: Layout, dev: DeviceConfig, on_device_error: E) -> Result<Self, resona_audio_io::AudioError>
    where
        E: Fn(String) + Send + Sync + Clone + 'static,
    {
        shared.stop.store(false, Ordering::Release);
        let parts = build(transport, shared.clone(), layout, dev.jitter_ms, dev.monitor.is_some());
        let PipelineParts { mut capture, mut render, monitor, dsp, cmds } = parts;
        let stop_all = |shared: &Arc<Shared>, dsp: JoinHandle<()>| {
            shared.stop.store(true, Ordering::Release);
            let _ = dsp.join();
        };
        let e1 = on_device_error.clone();
        let render_stream = match resona_audio_io::RenderStream::start(dev.output, move |out| render.process(out), move |e| e1(format!("output: {e}"))) {
            Ok(s) => s,
            Err(e) => {
                stop_all(&shared, dsp);
                return Err(e);
            }
        };
        let monitor_stream = match (dev.monitor, monitor) {
            (Some(cfg), Some(mut side)) => {
                let e2 = on_device_error.clone();
                match resona_audio_io::RenderStream::start(cfg, move |out| side.process(out), move |e| e2(format!("monitor: {e}"))) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        log::warn!("monitor output unavailable: {e}");
                        None
                    }
                }
            }
            _ => None,
        };
        let e3 = on_device_error.clone();
        let capture_stream = match resona_audio_io::CaptureStream::start_timed(dev.input, move |x, qpc| capture.process_timed(x, qpc), move |e| e3(format!("input: {e}"))) {
            Ok(s) => s,
            Err(e) => {
                drop(render_stream);
                drop(monitor_stream);
                stop_all(&shared, dsp);
                return Err(e);
            }
        };
        let info = PipelineInfo {
            input: capture_stream.info().clone(),
            output: render_stream.info().clone(),
            monitor: monitor_stream.as_ref().map(|m| m.info().clone()),
            jitter_ms: dev.jitter_ms,
        };
        Ok(Self { capture: Some(capture_stream), render: Some(render_stream), monitor: monitor_stream, dsp: Some(dsp), cmds, shared, info })
    }

    pub fn stop(&mut self) {
        if let Some(c) = self.capture.take() {
            c.stop();
        }
        if let Some(r) = self.render.take() {
            r.stop();
        }
        if let Some(m) = self.monitor.take() {
            m.stop();
        }
        self.shared.stop.store(true, Ordering::Release);
        if let Some(d) = self.dsp.take() {
            let _ = d.join();
        }
        self.shared.fx.detach();
        self.shared.player.detach();
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Small helper for waiting in tests and supervisors.
pub fn sleep_ms(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(p: &mut rtrb::Producer<f32>, n: usize) {
        for _ in 0..n {
            p.push(0.1).unwrap();
        }
    }

    #[test]
    fn jitter_target_adapts_to_underruns_and_relaxes() {
        let shared = Arc::new(Shared::default());
        let (mut tx, rx) = rtrb::RingBuffer::new(RATE);
        let base = 10 * RATE / 1000;
        let mut r = RenderSide::new(rx, base, ADAPT_MAX, None, shared.clone(), true);
        let mut out = [0.0f32; 480];
        feed(&mut tx, base + 240);
        r.process(&mut out);
        r.process(&mut out); // runs dry → underrun
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 1);
        assert_eq!(r.prefill, base + ADAPT_UP);
        // Steady supply for 40 s: the target eases back toward the configured minimum.
        feed(&mut tx, base + ADAPT_UP);
        for _ in 0..(40 * RATE / 480) {
            feed(&mut tx, 480);
            r.process(&mut out);
        }
        assert_eq!(shared.underruns.load(Ordering::Relaxed), 1);
        assert_eq!(r.prefill, base);
        // Repeated underruns never push it past the cap.
        for i in 0..20u64 {
            feed(&mut tx, r.prefill);
            while shared.underruns.load(Ordering::Relaxed) < 2 + i {
                r.process(&mut out);
            }
        }
        assert_eq!(r.prefill, base + ADAPT_MAX);
    }
}
