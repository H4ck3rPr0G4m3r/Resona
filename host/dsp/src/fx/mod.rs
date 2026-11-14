//! Effect chains (48 kHz mono, in place).
//!
//! * Every effect exposes a static parameter table ([`KindSpec`]); the UI is generated from it.
//! * Every continuous parameter is smoothed at audio rate (one-pole, ~20 ms), so moving a control
//!   mid-speech never clicks; enabling/disabling a slot fades over 10 ms.
//! * A chain is always driven in whole 10 ms frames ([`MAX_CHUNK`] samples, starting on a frame
//!   boundary). That is what lets a frame-based effect (noise suppression) cost one frame of
//!   latency instead of two.
//! * An effect with latency costs none while switched off: it is left out of the signal path,
//!   and switching it rebuilds the chain like any other structural change.
//! * Besides the built-in kinds of [`CATALOG`], a slot can hold a composite filter loaded from a
//!   TOML file ([`composite`]): its kind is `filter:<file>`, its parameters are its macros.
//! * The audio thread owns a [`LiveChain`]. The control side ([`ChainControl`]) builds new chains
//!   off the audio thread and hands them over through a lock-free ring; the old chain crossfades
//!   out and is sent back to be freed on the control side, so the audio thread never allocates.

pub mod biquad;
pub mod composite;
mod dynamics;
pub mod expr;
mod lofi;
pub mod prims;
mod space;
mod svf;
mod tone;
pub mod denoise;

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

pub use biquad::SR;

/// Processing granularity of a chain, and the frame its callers must keep to.
pub const MAX_CHUNK: usize = denoise::FRAME;
const FADE: usize = 480; // 10 ms enable/disable fade
const SWAP_FADE: usize = 960; // 20 ms chain swap crossfade

/// One-pole parameter smoother, ~20 ms time constant at 48 kHz.
#[derive(Clone, Copy, Debug)]
pub struct Smooth {
    pub cur: f32,
    pub target: f32,
}

impl Smooth {
    const K: f32 = 1.0 - 0.998_959_4; // 1 - exp(-1 / (0.02 * 48000))
    pub fn new(v: f32) -> Self {
        Self { cur: v, target: v }
    }
    #[inline]
    pub fn next(&mut self) -> f32 {
        self.cur += (self.target - self.cur) * Self::K;
        self.cur
    }
    /// Advance `n` samples at once (for per-block coefficient updates).
    pub fn advance(&mut self, n: usize) -> f32 {
        let k = 1.0 - (1.0 - Self::K).powi(n as i32);
        self.cur += (self.target - self.cur) * k;
        self.cur
    }
    pub fn settled(&self) -> bool {
        (self.target - self.cur).abs() <= 1e-5 * (1.0 + self.target.abs())
    }
    pub fn snap(&mut self) {
        self.cur = self.target;
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Scale {
    Linear,
    Log,
    /// Discrete choices; the value is the index.
    Choice(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ParamSpec {
    pub key: &'static str,
    pub name: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub unit: &'static str,
    pub scale: Scale,
}

const fn p(key: &'static str, name: &'static str, min: f32, max: f32, default: f32, unit: &'static str) -> ParamSpec {
    ParamSpec { key, name, min, max, default, unit, scale: Scale::Linear }
}
const fn plog(key: &'static str, name: &'static str, min: f32, max: f32, default: f32, unit: &'static str) -> ParamSpec {
    ParamSpec { key, name, min, max, default, unit, scale: Scale::Log }
}
const fn pchoice(key: &'static str, name: &'static str, choices: &'static [&'static str], default: f32) -> ParamSpec {
    ParamSpec { key, name, min: 0.0, max: (choices.len() - 1) as f32, default, unit: "", scale: Scale::Choice(choices) }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Input,
    Output,
    Both,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct KindSpec {
    pub kind: &'static str,
    pub name: &'static str,
    pub stage: Stage,
    pub params: &'static [ParamSpec],
}

pub static CATALOG: &[KindSpec] = &[
    KindSpec { kind: "gain", name: "Gain", stage: Stage::Both, params: &[p("gain", "Gain", -24.0, 24.0, 0.0, "dB")] },
    KindSpec {
        kind: "filter",
        name: "Filter",
        stage: Stage::Both,
        params: &[
            pchoice("mode", "Type", &["Low-pass", "High-pass"], 0.0),
            plog("freq", "Cutoff", 20.0, 20_000.0, 8_000.0, "Hz"),
            plog("q", "Resonance", 0.5, 12.0, 0.707, ""),
            pchoice("slope", "Slope", &["12 dB/oct", "24 dB/oct"], 0.0),
        ],
    },
    KindSpec {
        kind: "eq",
        name: "Equalizer",
        stage: Stage::Output,
        params: &[
            plog("f1", "Low freq", 20.0, 500.0, 100.0, "Hz"),
            p("g1", "Low gain", -18.0, 18.0, 0.0, "dB"),
            plog("f2", "Low-mid freq", 100.0, 2_000.0, 400.0, "Hz"),
            p("g2", "Low-mid gain", -18.0, 18.0, 0.0, "dB"),
            plog("q2", "Low-mid Q", 0.3, 8.0, 1.0, ""),
            plog("f3", "Mid freq", 300.0, 6_000.0, 1_500.0, "Hz"),
            p("g3", "Mid gain", -18.0, 18.0, 0.0, "dB"),
            plog("q3", "Mid Q", 0.3, 8.0, 1.0, ""),
            plog("f4", "High-mid freq", 1_000.0, 16_000.0, 4_000.0, "Hz"),
            p("g4", "High-mid gain", -18.0, 18.0, 0.0, "dB"),
            plog("q4", "High-mid Q", 0.3, 8.0, 1.0, ""),
            plog("f5", "High freq", 2_000.0, 20_000.0, 10_000.0, "Hz"),
            p("g5", "High gain", -18.0, 18.0, 0.0, "dB"),
        ],
    },
    KindSpec {
        kind: "gate",
        name: "Noise gate",
        stage: Stage::Both,
        params: &[
            p("threshold", "Threshold", -80.0, 0.0, -50.0, "dB"),
            plog("attack", "Attack", 0.1, 50.0, 1.0, "ms"),
            p("hold", "Hold", 0.0, 500.0, 80.0, "ms"),
            plog("release", "Release", 5.0, 1_000.0, 150.0, "ms"),
            p("range", "Range", -80.0, 0.0, -60.0, "dB"),
        ],
    },
    KindSpec {
        kind: "compressor",
        name: "Compressor",
        stage: Stage::Output,
        params: &[
            p("threshold", "Threshold", -60.0, 0.0, -18.0, "dB"),
            plog("ratio", "Ratio", 1.0, 20.0, 3.0, ":1"),
            plog("attack", "Attack", 0.1, 100.0, 5.0, "ms"),
            plog("release", "Release", 10.0, 1_000.0, 120.0, "ms"),
            p("knee", "Knee", 0.0, 24.0, 6.0, "dB"),
            p("makeup", "Makeup", 0.0, 24.0, 0.0, "dB"),
        ],
    },
    KindSpec {
        kind: "deesser",
        name: "De-esser",
        stage: Stage::Output,
        params: &[
            plog("freq", "Frequency", 4_000.0, 10_000.0, 6_000.0, "Hz"),
            p("threshold", "Threshold", -60.0, 0.0, -30.0, "dB"),
            p("reduction", "Max reduction", 0.0, 24.0, 8.0, "dB"),
        ],
    },
    KindSpec {
        kind: "reverb",
        name: "Reverb",
        stage: Stage::Output,
        params: &[
            p("size", "Size", 0.0, 1.0, 0.5, ""),
            p("damping", "Damping", 0.0, 1.0, 0.5, ""),
            p("predelay", "Pre-delay", 0.0, 100.0, 10.0, "ms"),
            p("mix", "Mix", 0.0, 1.0, 0.2, ""),
        ],
    },
    KindSpec {
        kind: "chorus",
        name: "Chorus",
        stage: Stage::Output,
        params: &[
            plog("rate", "Rate", 0.05, 5.0, 0.8, "Hz"),
            p("depth", "Depth", 0.1, 10.0, 3.0, "ms"),
            p("delay", "Delay", 5.0, 30.0, 12.0, "ms"),
            p("feedback", "Feedback", 0.0, 0.9, 0.1, ""),
            p("mix", "Mix", 0.0, 1.0, 0.35, ""),
        ],
    },
    KindSpec {
        kind: "lofi",
        name: "Lo-fi mic",
        stage: Stage::Output,
        params: &[
            plog("low", "Low cut", 50.0, 2_000.0, 300.0, "Hz"),
            plog("high", "High cut", 1_000.0, 16_000.0, 3_400.0, "Hz"),
            p("drive", "Drive", 0.0, 30.0, 6.0, "dB"),
            p("bits", "Bits", 2.0, 16.0, 10.0, ""),
            plog("downsample", "Downsample", 1.0, 32.0, 3.0, "x"),
            p("noise", "Noise", -90.0, -20.0, -62.0, "dB"),
            p("mix", "Mix", 0.0, 1.0, 1.0, ""),
        ],
    },
    KindSpec {
        kind: "denoise",
        name: "Noise suppression",
        stage: Stage::Input,
        params: &[p("amount", "Amount", 0.0, 1.0, 1.0, "")],
    },
];

pub fn spec(kind: &str) -> Option<&'static KindSpec> {
    CATALOG.iter().find(|k| k.kind == kind)
}

/// A processor. Parameters are addressed by their index in the kind's [`ParamSpec`] table.
pub trait Effect: Send {
    fn set(&mut self, idx: usize, value: f32);
    fn process(&mut self, x: &mut [f32]);
    fn reset(&mut self);
    /// Constant latency in samples (the chain delays the dry path to match).
    fn latency(&self) -> usize {
        0
    }
}

pub fn create(kind: &str) -> Option<Box<dyn Effect>> {
    let fx: Box<dyn Effect> = match kind {
        "gain" => Box::new(tone::Gain::new()),
        "filter" => Box::new(tone::Filter::new()),
        "eq" => Box::new(tone::Eq5::new()),
        "gate" => Box::new(dynamics::Gate::new()),
        "compressor" => Box::new(dynamics::Compressor::new()),
        "deesser" => Box::new(dynamics::DeEsser::new()),
        "reverb" => Box::new(space::Reverb::new()),
        "chorus" => Box::new(space::Chorus::new()),
        "lofi" => Box::new(lofi::LoFi::new()),
        "denoise" => Box::new(denoise::Denoise::new()),
        _ => return None,
    };
    let mut fx = fx;
    for (i, ps) in spec(kind)?.params.iter().enumerate() {
        fx.set(i, ps.default);
    }
    fx.reset(); // start settled at the defaults
    Some(fx)
}

/// `create`, for use inside a chain: chains run in whole frames, so frame-based effects can use
/// their low-latency form.
fn create_native(kind: &str) -> Option<Box<dyn Effect>> {
    if kind != "denoise" {
        return create(kind);
    }
    let mut fx = denoise::Denoise::framed();
    fx.set(0, spec(kind)?.params[0].default);
    fx.reset();
    Some(Box::new(fx))
}

/// The effect for a chain slot: a built-in kind or a composite filter.
fn create_in_chain(kind: &str) -> Option<Box<dyn Effect>> {
    if kind.starts_with(composite::PREFIX) {
        return Some(Box::new(composite::Composite::new(kind)));
    }
    create_native(kind)
}

/// Latency (samples) a slot of this kind adds to a chain while enabled.
pub fn chain_latency(kind: &str) -> usize {
    match composite::get(kind) {
        Some(def) => def.stages.iter().map(|s| chain_latency(&s.kind)).sum(),
        None if kind == "denoise" => denoise::LATENCY_FRAMED,
        None => 0,
    }
}

/// What a chain needs to know about one parameter of a slot.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamMeta {
    pub key: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

/// The parameters of a slot kind, in index order: a built-in kind's table, or a filter's
/// macros. A filter whose file is gone has none (it still builds, as a bypass); any other
/// unknown kind is `None`.
pub fn params_of(kind: &str) -> Option<Vec<ParamMeta>> {
    if kind.starts_with(composite::PREFIX) {
        let def = composite::get(kind);
        let macros = def.iter().flat_map(|d| d.macros.iter());
        return Some(macros.map(|m| ParamMeta { key: m.key.clone(), min: m.min, max: m.max, default: m.default }).collect());
    }
    Some(spec(kind)?.params.iter().map(|p| ParamMeta { key: p.key.to_string(), min: p.min, max: p.max, default: p.default }).collect())
}

/// Everything the UI can put in a chain: the built-in kinds, then the loaded filters.
pub fn catalog_json() -> serde_json::Value {
    let mut all: Vec<serde_json::Value> = CATALOG.iter().map(|k| serde_json::to_value(k).unwrap_or_default()).collect();
    all.extend(composite::all().iter().map(|d| d.catalog_entry()));
    serde_json::Value::Array(all)
}

pub use dynamics::Limiter;

// -- chain ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SlotSpec {
    pub id: u64,
    pub kind: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    #[serde(default)]
    pub params: BTreeMap<String, f32>,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ChainSpec {
    #[serde(default)]
    pub slots: Vec<SlotSpec>,
}

struct Slot {
    id: u64,
    fx: Box<dyn Effect>,
    mix: Smooth,
    dry: Vec<f32>,
    /// Built switched off and has latency: out of the signal path (see the module docs).
    bypass: bool,
}

pub struct Chain {
    slots: Vec<Slot>,
}

impl Chain {
    pub fn empty() -> Self {
        Self { slots: Vec::new() }
    }

    pub fn build(spec: &ChainSpec) -> Result<Self, String> {
        let mut slots = Vec::with_capacity(spec.slots.len());
        for s in &spec.slots {
            let params = params_of(&s.kind).ok_or_else(|| format!("unknown effect '{}'", s.kind))?;
            let mut fx = create_in_chain(&s.kind).expect("catalog and create agree");
            // A filter's macros change when its file is edited; values saved for macros it no
            // longer has are skipped, not an error.
            let is_filter = s.kind.starts_with(composite::PREFIX);
            for (k, v) in &s.params {
                match params.iter().position(|p| p.key == *k) {
                    Some(i) => fx.set(i, v.clamp(params[i].min, params[i].max)),
                    None if is_filter => {}
                    None => return Err(format!("{}: unknown parameter '{k}'", s.kind)),
                }
            }
            fx.reset();
            let bypass = !s.enabled && fx.latency() > 0;
            slots.push(Slot { id: s.id, fx, mix: Smooth::new(if s.enabled { 1.0 } else { 0.0 }), dry: vec![0.0; MAX_CHUNK], bypass });
        }
        Ok(Self { slots })
    }

    pub fn latency(&self) -> usize {
        self.slots.iter().filter(|s| !s.bypass).map(|s| s.fx.latency()).sum()
    }

    pub fn set_param(&mut self, id: u64, idx: usize, v: f32) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.id == id) {
            s.fx.set(idx, v);
        }
    }

    /// Fade a slot in or out in place. Not for effects with latency: switching those changes
    /// the chain's delay, so [`ChainControl::set`] rebuilds the chain instead.
    pub fn set_enabled(&mut self, id: u64, on: bool) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.id == id && s.fx.latency() == 0) {
            s.mix.target = if on { 1.0 } else { 0.0 };
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.slots {
            s.fx.reset();
        }
    }

    pub fn process(&mut self, buf: &mut [f32]) {
        for chunk in buf.chunks_mut(MAX_CHUNK) {
            for s in &mut self.slots {
                s.process(chunk);
            }
        }
    }
}

impl Slot {
    fn process(&mut self, x: &mut [f32]) {
        if self.bypass {
            return;
        }
        if self.mix.target == 0.0 && self.mix.cur < 1e-4 {
            self.mix.cur = 0.0;
            return; // switched off and faded out: skip the work entirely
        }
        if self.mix.cur == 1.0 && self.mix.target == 1.0 {
            self.fx.process(x);
            return;
        }
        let n = x.len();
        self.dry[..n].copy_from_slice(x);
        self.fx.process(x);
        let k = 1.0 / FADE as f32;
        for i in 0..n {
            let m = &mut self.mix;
            m.cur = if m.target > m.cur { (m.cur + k).min(m.target) } else { (m.cur - k).max(m.target) };
            x[i] = self.dry[i] + (x[i] - self.dry[i]) * m.cur;
        }
    }
}

// -- live handoff ----------------------------------------------------------------------------------

pub enum ChainMsg {
    Swap(Box<Chain>),
    Param { id: u64, idx: usize, value: f32 },
    Enable { id: u64, on: bool },
}

/// Audio-thread side.
pub struct LiveChain {
    cur: Box<Chain>,
    old: Option<Box<Chain>>,
    /// Samples the new chain still runs unheard before the crossfade: its latency, so what it
    /// fades in with is audio and not the silence in front of it.
    warm: usize,
    fade: usize,
    rx: rtrb::Consumer<ChainMsg>,
    trash: rtrb::Producer<Box<Chain>>,
    tmp: Vec<f32>,
}

/// Control side: build/patch chains from any thread.
pub struct ChainControl {
    tx: Mutex<rtrb::Producer<ChainMsg>>,
    trash: Mutex<rtrb::Consumer<Box<Chain>>>,
    spec: Mutex<ChainSpec>,
}

pub fn live_chain() -> (LiveChain, Arc<ChainControl>) {
    let (tx, rx) = rtrb::RingBuffer::new(256);
    let (ttx, trx) = rtrb::RingBuffer::new(8);
    (
        LiveChain { cur: Box::new(Chain::empty()), old: None, warm: 0, fade: 0, rx, trash: ttx, tmp: vec![0.0; MAX_CHUNK] },
        Arc::new(ChainControl { tx: Mutex::new(tx), trash: Mutex::new(trx), spec: Mutex::new(ChainSpec::default()) }),
    )
}

impl LiveChain {
    /// Delay of what is leaving right now.
    pub fn latency(&self) -> usize {
        match &self.old {
            Some(old) if self.warm > 0 => old.latency(),
            _ => self.cur.latency(),
        }
    }

    /// `buf`: whole frames ([`MAX_CHUNK`] samples each), as for every chain.
    pub fn process(&mut self, buf: &mut [f32]) {
        while let Ok(msg) = self.rx.pop() {
            match msg {
                ChainMsg::Swap(c) => {
                    let prev = std::mem::replace(&mut self.cur, c);
                    if let Some(o) = self.old.replace(prev) {
                        let _ = self.trash.push(o); // a still-fading chain is dropped by the control side
                    }
                    self.warm = self.cur.latency();
                    self.fade = SWAP_FADE;
                }
                ChainMsg::Param { id, idx, value } => self.cur.set_param(id, idx, value),
                ChainMsg::Enable { id, on } => self.cur.set_enabled(id, on),
            }
        }
        if self.old.is_none() {
            self.cur.process(buf);
            return;
        }
        for chunk in buf.chunks_mut(MAX_CHUNK) {
            let Some(old) = self.old.as_mut() else {
                self.cur.process(chunk);
                continue;
            };
            let n = chunk.len();
            let tmp = &mut self.tmp[..n];
            tmp.copy_from_slice(chunk);
            old.process(tmp);
            self.cur.process(chunk);
            for i in 0..n {
                if self.warm > 0 {
                    self.warm -= 1;
                    chunk[i] = tmp[i];
                    continue;
                }
                let w = self.fade as f32 / SWAP_FADE as f32;
                chunk[i] = chunk[i] * (1.0 - w) + tmp[i] * w;
                self.fade = self.fade.saturating_sub(1);
            }
            if self.fade == 0 {
                if let Some(o) = self.old.take() {
                    let _ = self.trash.push(o);
                }
            }
        }
    }
}

impl ChainControl {
    fn send(&self, msg: ChainMsg) {
        // Free chains the audio thread is done with, here, off the audio thread.
        while let Ok(c) = self.trash.lock().pop() {
            drop(c);
        }
        let _ = self.tx.lock().push(msg);
    }

    pub fn spec(&self) -> ChainSpec {
        self.spec.lock().clone()
    }

    /// Replace the whole chain. Structural changes rebuild (with a 20 ms crossfade); a spec that
    /// only differs in parameter values or enable flags is patched in place instead. Switching an
    /// effect that has latency counts as structural: it moves the whole chain in time.
    pub fn set(&self, new: ChainSpec) -> Result<(), String> {
        let mut cur = self.spec.lock();
        let same_shape = cur.slots.len() == new.slots.len()
            && cur.slots.iter().zip(&new.slots).all(|(a, b)| {
                a.id == b.id && a.kind == b.kind && (a.enabled == b.enabled || chain_latency(&b.kind) == 0)
            });
        if same_shape {
            for (a, b) in cur.slots.iter().zip(&new.slots) {
                let params = params_of(&b.kind).ok_or("unknown effect")?;
                for (i, ps) in params.iter().enumerate() {
                    let va = a.params.get(&ps.key).copied().unwrap_or(ps.default);
                    let vb = b.params.get(&ps.key).copied().unwrap_or(ps.default).clamp(ps.min, ps.max);
                    if va != vb {
                        self.send(ChainMsg::Param { id: b.id, idx: i, value: vb });
                    }
                }
                if a.enabled != b.enabled {
                    self.send(ChainMsg::Enable { id: b.id, on: b.enabled });
                }
            }
        } else {
            let chain = Chain::build(&new)?;
            self.send(ChainMsg::Swap(Box::new(chain)));
        }
        *cur = new;
        Ok(())
    }

    pub fn set_param(&self, id: u64, key: &str, value: f32) -> Result<(), String> {
        let mut cur = self.spec.lock();
        let slot = cur.slots.iter_mut().find(|s| s.id == id).ok_or("no such slot")?;
        let params = params_of(&slot.kind).ok_or("unknown effect")?;
        let idx = params.iter().position(|p| p.key == key).ok_or("unknown parameter")?;
        let v = value.clamp(params[idx].min, params[idx].max);
        slot.params.insert(key.to_string(), v);
        self.send(ChainMsg::Param { id, idx, value: v });
        Ok(())
    }

    /// Build the chain again from its current spec (crossfaded like any structural change).
    /// For when what a slot's kind means has changed: a filter file was edited.
    pub fn rebuild(&self) -> Result<(), String> {
        let cur = self.spec.lock();
        let chain = Chain::build(&cur)?;
        self.send(ChainMsg::Swap(Box::new(chain)));
        Ok(())
    }
}

#[cfg(test)]
mod tests;
