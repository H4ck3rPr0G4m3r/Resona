//! Effect state shared by the runtime and the audio threads.
//!
//! The chain *specs* live here so a pipeline rebuilt after a device loss gets the same chains;
//! each pipeline owns fresh audio-side [`LiveChain`]s whose controls are registered here.
//!
//! Voice-activity gate: RNNoise's VAD runs on the raw capture (one probability per 10 ms frame,
//! written to a timeline ring). The render side reads it for the exact input time each output
//! sample came from, so the gate adds no latency and, looking 20 ms ahead, never clips onsets.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use resona_dsp::fx::{ChainControl, ChainSpec};
use resona_dsp::AtomicF32;
use serde::{Deserialize, Serialize};

pub const VAD_FRAME: usize = 480;
const VAD_RING: usize = 4096; // 41 s of frames

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GateParams {
    pub enabled: bool,
    /// Voice probability needed to open (0..1).
    pub threshold: f32,
    pub hold_ms: f32,
    pub release_ms: f32,
    /// Attenuation while closed.
    pub range_db: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self { enabled: false, threshold: 0.6, hold_ms: 200.0, release_ms: 300.0, range_db: -45.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct LimiterParams {
    pub enabled: bool,
    pub ceiling_db: f32,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self { enabled: true, ceiling_db: -1.0 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FxSettings {
    pub input: ChainSpec,
    pub output: ChainSpec,
    pub gate: GateParams,
    pub limiter: LimiterParams,
    /// Bypasses noise suppression and the voice gate (they mangle sustained notes).
    pub singing: bool,
}

pub struct FxHub {
    settings: Mutex<FxSettings>,
    controls: Mutex<Option<(Arc<ChainControl>, Arc<ChainControl>)>>,
    vad: Box<[AtomicU8]>,
    pub vad_frames: AtomicU64,
    pub gate_on: AtomicBool,
    pub gate_threshold: AtomicF32,
    pub gate_hold: AtomicUsize,
    pub gate_release: AtomicF32, // one-pole coefficient
    pub gate_range: AtomicF32,   // linear
    pub gate_open: AtomicBool,   // for the meters
    pub limiter_on: AtomicBool,
    pub limiter_ceiling_db: AtomicF32,
    /// Engine lookahead in 48 kHz samples (output lags input timeline by this much).
    pub lookahead: AtomicUsize,
    pub input_latency: AtomicUsize,
    pub output_latency: AtomicUsize,
}

impl Default for FxHub {
    fn default() -> Self {
        let hub = Self {
            settings: Mutex::new(FxSettings::default()),
            controls: Mutex::new(None),
            vad: (0..VAD_RING).map(|_| AtomicU8::new(255)).collect(),
            vad_frames: AtomicU64::new(0),
            gate_on: AtomicBool::new(false),
            gate_threshold: AtomicF32::new(0.6),
            gate_hold: AtomicUsize::new(9600),
            gate_release: AtomicF32::new(0.0),
            gate_range: AtomicF32::new(0.0056),
            gate_open: AtomicBool::new(true),
            limiter_on: AtomicBool::new(true),
            limiter_ceiling_db: AtomicF32::new(-1.0),
            lookahead: AtomicUsize::new(40 * 48),
            input_latency: AtomicUsize::new(0),
            output_latency: AtomicUsize::new(0),
        };
        hub.publish(&FxSettings::default());
        hub
    }
}

fn effective_input(s: &FxSettings) -> ChainSpec {
    let mut spec = s.input.clone();
    if s.singing {
        for slot in &mut spec.slots {
            if slot.kind == "denoise" || slot.kind == "gate" {
                slot.enabled = false;
            }
        }
    }
    spec
}

impl FxHub {
    pub fn settings(&self) -> FxSettings {
        self.settings.lock().clone()
    }

    fn publish(&self, s: &FxSettings) {
        let g = &s.gate;
        self.gate_on.store(g.enabled && !s.singing, Ordering::Relaxed);
        self.gate_threshold.store(g.threshold.clamp(0.0, 1.0));
        self.gate_hold.store((g.hold_ms.max(0.0) * 48.0) as usize, Ordering::Relaxed);
        self.gate_release.store(1.0 - (-1.0 / (g.release_ms.max(1.0) * 48.0)).exp());
        self.gate_range.store(resona_dsp::from_db(g.range_db.min(0.0)));
        self.limiter_on.store(s.limiter.enabled, Ordering::Relaxed);
        self.limiter_ceiling_db.store(s.limiter.ceiling_db.clamp(-12.0, 0.0));
    }

    /// Validate and apply new settings (to the running pipeline too, if any).
    pub fn update(&self, new: FxSettings) -> Result<(), String> {
        resona_dsp::fx::Chain::build(&new.input)?;
        resona_dsp::fx::Chain::build(&new.output)?;
        if let Some((i, o)) = self.controls.lock().as_ref() {
            i.set(effective_input(&new))?;
            o.set(new.output.clone())?;
        }
        self.publish(&new);
        *self.settings.lock() = new;
        Ok(())
    }

    /// One parameter of one slot, without touching the chain structure (UI drags).
    pub fn set_param(&self, chain: &str, slot: u64, key: &str, value: f32) -> Result<(), String> {
        let mut s = self.settings.lock();
        let spec = match chain {
            "input" => &mut s.input,
            "output" => &mut s.output,
            _ => return Err("chain must be 'input' or 'output'".into()),
        };
        let sl = spec.slots.iter_mut().find(|x| x.id == slot).ok_or("no such slot")?;
        sl.params.insert(key.to_string(), value);
        if let Some((i, o)) = self.controls.lock().as_ref() {
            if chain == "input" { i } else { o }.set_param(slot, key, value)?;
        }
        Ok(())
    }

    /// Register a new pipeline's chain controls and load the current specs into them.
    pub fn attach(&self, input: Arc<ChainControl>, output: Arc<ChainControl>) {
        let s = self.settings.lock().clone();
        let _ = input.set(effective_input(&s));
        let _ = output.set(s.output.clone());
        self.vad_frames.store(0, Ordering::Relaxed);
        *self.controls.lock() = Some((input, output));
    }

    pub fn detach(&self) {
        *self.controls.lock() = None;
    }

    pub fn push_vad(&self, prob: f32) {
        let f = self.vad_frames.load(Ordering::Relaxed);
        self.vad[(f as usize) % VAD_RING].store((prob.clamp(0.0, 1.0) * 255.0) as u8, Ordering::Relaxed);
        self.vad_frames.store(f + 1, Ordering::Release);
    }

    /// Voice probability around raw-capture sample `t`, looking `ahead` frames into the future.
    /// Unknown time (before capture start, or not yet analysed) counts as voice: fail open.
    pub fn vad_at(&self, t: i64, ahead: usize) -> f32 {
        if t < 0 {
            return 1.0;
        }
        let f0 = t as u64 / VAD_FRAME as u64;
        let have = self.vad_frames.load(Ordering::Acquire);
        let mut best = 0u8;
        for f in f0..=f0 + ahead as u64 {
            if f >= have || have - f > VAD_RING as u64 {
                return 1.0;
            }
            best = best.max(self.vad[(f as usize) % VAD_RING].load(Ordering::Relaxed));
        }
        best as f32 / 255.0
    }
}
