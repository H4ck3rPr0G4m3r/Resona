//! Composite filters: an effect described by a TOML file instead of code.
//!
//! ```toml
//! name = "Telephone"
//!
//! [macros]                       # the knobs the editor shows
//! amount = { default = 0.5 }     # 0..1 unless min/max say otherwise; `amount = 0.5` works too
//!
//! [[chain]]                      # stages run top to bottom
//! type = "highpass"
//! freq = "lerp(80, 400, amount)" # a number, or an expression over the macros
//!
//! [[chain]]
//! type = "lowpass"
//! freq = 3400
//! ```
//!
//! A stage's `type` is a primitive ([`prims`](super::prims)) or any chain effect, so everything
//! native is available to filters. A file that does not parse still loads: the filter keeps its
//! place in the chains that use it, passes audio through unchanged and carries the error text
//! for the editor to show.
//!
//! Loaded filters live in a process-wide registry, keyed by `filter:<file stem>`, which is the
//! `kind` a chain slot refers to. Reloading a file replaces its entry; chains pick the new
//! definition up when they are rebuilt.

use std::sync::{Arc, RwLock};

use serde_json::{json, Value};

use super::expr::Expr;
use super::{prims, Effect, KindSpec, ParamSpec, Scale, Stage};

/// Prefix of the `kind` of a chain slot that holds a composite filter.
pub const PREFIX: &str = "filter:";

#[derive(Clone, Debug, PartialEq)]
pub struct MacroDef {
    pub key: String,
    pub name: String,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub unit: String,
}

/// One stage parameter: where its value comes from and the range it is kept in.
#[derive(Clone, Debug)]
struct Bind {
    param: usize,
    expr: Expr,
    min: f32,
    max: f32,
    default: f32,
}

#[derive(Clone, Debug)]
pub struct StageDef {
    pub kind: String,
    binds: Vec<Bind>,
}

#[derive(Clone, Debug)]
pub struct FilterDef {
    /// `filter:<file stem>`.
    pub kind: String,
    pub name: String,
    pub description: String,
    pub stage: Stage,
    pub macros: Vec<MacroDef>,
    pub stages: Vec<StageDef>,
    /// Why the file could not be used. The filter is a bypass while this is set.
    pub error: Option<String>,
    pub file: String,
    pub builtin: bool,
}

/// `kind` for a filter file name without its extension: lower case, `[a-z0-9_-]` only.
pub fn kind_for(stem: &str) -> String {
    let id: String = stem.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_lowercase() } else { '_' }).collect();
    format!("{PREFIX}{id}")
}

fn title(stem: &str) -> String {
    let mut out = String::new();
    for word in stem.split(|c: char| c == '_' || c == '-' || c == ' ').filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        let mut cs = word.chars();
        if let Some(c) = cs.next() {
            out.extend(c.to_uppercase());
            out.push_str(cs.as_str());
        }
    }
    out
}

fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_') && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn number(v: &toml::Value) -> Option<f32> {
    match v {
        toml::Value::Integer(i) => Some(*i as f32),
        toml::Value::Float(f) => Some(*f as f32),
        _ => None,
    }
}

/// A stage type: a primitive first, then any chain effect.
fn stage_spec(kind: &str) -> Option<&'static KindSpec> {
    prims::spec(kind).or_else(|| super::spec(kind))
}

fn stage_types() -> String {
    let mut names: Vec<&str> = prims::PRIMS.iter().map(|k| k.kind).collect();
    names.extend(super::CATALOG.iter().map(|k| k.kind).filter(|k| prims::spec(k).is_none()));
    names.join(", ")
}

/// Find a parameter by its key, forgiving the unit suffix: `threshold_db` finds `threshold`,
/// `gain` finds `gain_db`.
fn find_param(params: &[ParamSpec], key: &str) -> Option<usize> {
    const UNITS: [&str; 3] = ["_db", "_ms", "_hz"];
    let exact = |k: &str| params.iter().position(|p| p.key == k);
    exact(key)
        .or_else(|| UNITS.iter().find_map(|u| key.strip_suffix(u).and_then(exact)))
        .or_else(|| UNITS.iter().find_map(|u| exact(&format!("{key}{u}"))))
}

impl FilterDef {
    /// Read a filter file. Never fails: what is wrong with it ends up in `error`.
    pub fn parse(stem: &str, text: &str) -> Self {
        let mut def = Self {
            kind: kind_for(stem),
            name: title(stem),
            description: String::new(),
            stage: Stage::Both,
            macros: Vec::new(),
            stages: Vec::new(),
            error: None,
            file: String::new(),
            builtin: false,
        };
        if let Err(e) = def.fill(text) {
            def.stages.clear();
            def.error = Some(e);
        }
        def
    }

    fn fill(&mut self, text: &str) -> Result<(), String> {
        let table: toml::Table = text.parse().map_err(|e: toml::de::Error| {
            let line = e.span().map(|s| text[..s.start.min(text.len())].matches('\n').count() + 1);
            match line {
                Some(n) => format!("line {n}: {}", e.message()),
                None => e.message().to_string(),
            }
        })?;
        for (key, value) in &table {
            match (key.as_str(), value) {
                ("name", toml::Value::String(s)) => self.name = s.clone(),
                ("description", toml::Value::String(s)) => self.description = s.clone(),
                ("stage", toml::Value::String(s)) => {
                    self.stage = match s.as_str() {
                        "input" => Stage::Input,
                        "output" => Stage::Output,
                        "both" => Stage::Both,
                        other => return Err(format!("stage = \"{other}\": use \"input\", \"output\" or \"both\"")),
                    }
                }
                ("name" | "description" | "stage", _) => return Err(format!("{key} must be a text in quotes")),
                ("macros", toml::Value::Table(_)) | ("chain", toml::Value::Array(_)) => {}
                ("macros", _) => return Err("macros must be a [macros] table".into()),
                ("chain", _) => return Err("chain must be a list of [[chain]] stages".into()),
                _ => return Err(format!("unknown key '{key}' (a filter has: name, description, stage, [macros], [[chain]])")),
            }
        }
        if let Some(toml::Value::Table(macros)) = table.get("macros") {
            for (key, value) in macros {
                self.macros.push(parse_macro(key, value)?);
            }
        }
        let names: Vec<&str> = self.macros.iter().map(|m| m.key.as_str()).collect();
        let mut stages = Vec::new();
        if let Some(toml::Value::Array(chain)) = table.get("chain") {
            for (i, stage) in chain.iter().enumerate() {
                stages.push(parse_stage(i + 1, stage, &names)?);
            }
        }
        if stages.is_empty() {
            return Err("the filter has no [[chain]] stages".into());
        }
        self.stages = stages;
        Ok(())
    }

    /// The catalog entry the editor is generated from: the shape of a [`KindSpec`], plus a
    /// `filter` object with what only filters have.
    pub fn catalog_entry(&self) -> Value {
        let params: Vec<Value> = self
            .macros
            .iter()
            .map(|m| json!({"key": m.key, "name": m.name, "min": m.min, "max": m.max, "default": m.default, "unit": m.unit, "scale": "linear"}))
            .collect();
        json!({
            "kind": self.kind,
            "name": self.name,
            "stage": self.stage,
            "params": params,
            "filter": {
                "file": self.file,
                "builtin": self.builtin,
                "error": self.error,
                "description": self.description,
                "stages": self.stages.iter().map(|s| s.kind.as_str()).collect::<Vec<_>>(),
            },
        })
    }
}

fn parse_macro(key: &str, value: &toml::Value) -> Result<MacroDef, String> {
    if !is_ident(key) {
        return Err(format!("macro '{key}': a name is letters, digits and _, and does not start with a digit"));
    }
    let mut m = MacroDef { key: key.to_string(), name: title(key), min: 0.0, max: 1.0, default: 0.5, unit: String::new() };
    match value {
        toml::Value::Table(t) => {
            for (k, v) in t {
                match (k.as_str(), number(v), v) {
                    ("default", Some(n), _) => m.default = n,
                    ("min", Some(n), _) => m.min = n,
                    ("max", Some(n), _) => m.max = n,
                    ("name", _, toml::Value::String(s)) => m.name = s.clone(),
                    ("unit", _, toml::Value::String(s)) => m.unit = s.clone(),
                    ("default" | "min" | "max", None, _) => return Err(format!("macro '{key}': {k} must be a number")),
                    ("name" | "unit", _, _) => return Err(format!("macro '{key}': {k} must be a text in quotes")),
                    _ => return Err(format!("macro '{key}': unknown setting '{k}' (a macro has: default, min, max, name, unit)")),
                }
            }
        }
        v => m.default = number(v).ok_or_else(|| format!("macro '{key}' must be a number or {{ default = … }}"))?,
    }
    if !(m.min < m.max) {
        return Err(format!("macro '{key}': min must be below max"));
    }
    m.default = m.default.clamp(m.min, m.max);
    Ok(m)
}

fn parse_stage(n: usize, value: &toml::Value, macros: &[&str]) -> Result<StageDef, String> {
    let toml::Value::Table(t) = value else { return Err(format!("stage {n} must be a [[chain]] table")) };
    let kind = match t.get("type") {
        Some(toml::Value::String(s)) => s.as_str(),
        _ => return Err(format!("stage {n} needs a type, for example type = \"lowpass\"")),
    };
    let ks = stage_spec(kind).ok_or_else(|| format!("stage {n}: there is no '{kind}' (types: {})", stage_types()))?;
    let at = format!("stage {n} ({kind})");
    let mut binds = Vec::new();
    for (key, v) in t.iter().filter(|(k, _)| k.as_str() != "type") {
        let Some(i) = find_param(ks.params, key) else {
            let has: Vec<&str> = ks.params.iter().map(|p| p.key).collect();
            return Err(format!("{at}: no setting '{key}' (it has: {})", has.join(", ")));
        };
        let ps = &ks.params[i];
        let constant = |x: f32| -> Result<Expr, String> {
            if x < ps.min || x > ps.max {
                return Err(format!("{at}: {key} = {x} is outside {} to {}", ps.min, ps.max));
            }
            Ok(Expr::constant(x))
        };
        let expr = match (v, ps.scale) {
            (toml::Value::Boolean(b), _) => constant(*b as u8 as f32)?,
            (toml::Value::String(s), Scale::Choice(choices)) => match choices.iter().position(|c| c.eq_ignore_ascii_case(s)) {
                Some(c) => Expr::constant(c as f32),
                None => return Err(format!("{at}: {key} = \"{s}\" is not one of: {}", choices.join(", "))),
            },
            (toml::Value::String(s), _) => Expr::parse(s, macros).map_err(|e| format!("{at}: {key}: {e}"))?,
            (v, _) => constant(number(v).ok_or_else(|| format!("{at}: {key} must be a number or an expression in quotes"))?)?,
        };
        binds.push(Bind { param: i, expr, min: ps.min, max: ps.max, default: ps.default });
    }
    Ok(StageDef { kind: kind.to_string(), binds })
}

// -- registry ------------------------------------------------------------------------------------

static REGISTRY: RwLock<Vec<Arc<FilterDef>>> = RwLock::new(Vec::new());

/// Replace every loaded filter (the result of scanning the filter folders).
pub fn install(defs: Vec<FilterDef>) {
    *REGISTRY.write().unwrap() = defs.into_iter().map(Arc::new).collect();
}

/// Add one filter, or replace the one of the same kind.
pub fn upsert(def: FilterDef) {
    let mut reg = REGISTRY.write().unwrap();
    match reg.iter_mut().find(|d| d.kind == def.kind) {
        Some(slot) => *slot = Arc::new(def),
        None => reg.push(Arc::new(def)),
    }
}

pub fn get(kind: &str) -> Option<Arc<FilterDef>> {
    REGISTRY.read().unwrap().iter().find(|d| d.kind == kind).cloned()
}

pub fn all() -> Vec<Arc<FilterDef>> {
    REGISTRY.read().unwrap().clone()
}

// -- the effect ----------------------------------------------------------------------------------

struct StageFx {
    fx: Box<dyn Effect>,
    binds: Vec<Bind>,
}

/// A filter in a chain. Its parameters are the filter's macros, by index.
pub struct Composite {
    stages: Vec<StageFx>,
    macros: Vec<f32>,
}

impl Composite {
    /// The filter `kind` as currently registered. An unknown kind (its file is gone) or a file
    /// with an error gives a bypass, so the chains that refer to it keep working.
    pub fn new(kind: &str) -> Self {
        let Some(def) = get(kind) else { return Self { stages: Vec::new(), macros: Vec::new() } };
        let stages = def
            .stages
            .iter()
            .filter_map(|s| {
                let fx = prims::create(&s.kind).or_else(|| super::create_native(&s.kind))?;
                Some(StageFx { fx, binds: s.binds.clone() })
            })
            .collect();
        let mut c = Self { stages, macros: def.macros.iter().map(|m| m.default).collect() };
        c.apply(None);
        c.reset();
        c
    }

    /// Push the parameters that depend on macro `changed` (`None`: all) into their stages.
    fn apply(&mut self, changed: Option<usize>) {
        for s in &mut self.stages {
            for b in &s.binds {
                if changed.is_some_and(|i| !b.expr.uses(i)) {
                    continue;
                }
                let v = b.expr.eval(&self.macros);
                s.fx.set(b.param, if v.is_finite() { v.clamp(b.min, b.max) } else { b.default });
            }
        }
    }
}

impl Effect for Composite {
    fn set(&mut self, idx: usize, value: f32) {
        if let Some(m) = self.macros.get_mut(idx) {
            *m = value;
            self.apply(Some(idx));
        }
    }
    fn process(&mut self, x: &mut [f32]) {
        for s in &mut self.stages {
            s.fx.process(x);
        }
    }
    fn reset(&mut self) {
        for s in &mut self.stages {
            s.fx.reset();
        }
    }
    fn latency(&self) -> usize {
        self.stages.iter().map(|s| s.fx.latency()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BAD_MIC: &str = include_str!("../../../../filters/bad_mic.toml");

    fn tone(n: usize, f: f32, amp: f32) -> Vec<f32> {
        (0..n).map(|i| amp * (std::f32::consts::TAU * f * i as f32 / super::super::SR).sin()).collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn bad_mic_ships_and_parses() {
        let def = FilterDef::parse("bad_mic", BAD_MIC);
        assert_eq!(def.error, None);
        assert_eq!((def.kind.as_str(), def.name.as_str()), ("filter:bad_mic", "Bad Mic"));
        let keys: Vec<&str> = def.macros.iter().map(|m| m.key.as_str()).collect();
        assert_eq!(keys, ["strength", "telephone", "static"], "macros keep the order of the file");
        assert_eq!(def.macros[0].default, 0.6);
        let kinds: Vec<&str> = def.stages.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds, ["highpass", "lowpass", "peak", "downsample", "bitcrush", "saturate", "compressor", "noise", "crackle"]);
    }

    #[test]
    fn a_macro_moves_the_stages_it_feeds() {
        upsert(FilterDef::parse(
            "t_phone",
            r#"
            [macros]
            amount = { default = 0.0 }
            [[chain]]
            type = "lowpass"
            freq = "lerp(18000, 400, amount)"
            "#,
        ));
        let mut fx = Composite::new("filter:t_phone");
        let mut x = tone(24_000, 4000.0, 0.5);
        fx.process(&mut x);
        assert!((rms(&x[12_000..]) - 0.3536).abs() < 0.01, "amount 0: 4 kHz passes, rms {}", rms(&x[12_000..]));
        fx.set(0, 1.0);
        let mut x = tone(24_000, 4000.0, 0.5);
        fx.process(&mut x);
        assert!(rms(&x[12_000..]) < 0.01, "amount 1: 4 kHz is gone, rms {}", rms(&x[12_000..]));
        // The move itself is smoothed by the stage: no step in the output.
        let step = x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
        assert!(step < 0.3, "macro change clicked: {step}");
    }

    #[test]
    fn a_broken_file_is_a_bypass_that_says_why() {
        let cases = [
            ("name = ", "line 1: "),
            ("[[chain]]\ntype = \"lowpas\"", "stage 1: there is no 'lowpas' (types: highpass, lowpass, "),
            ("[[chain]]\ntype = \"peak\"\ngian_db = 3", "stage 1 (peak): no setting 'gian_db' (it has: freq, q, gain_db)"),
            ("[[chain]]\ntype = \"peak\"\ngain_db = \"6 * strenght\"", "stage 1 (peak): gain_db: unknown name 'strenght' (this filter has no macros)"),
            ("[[chain]]\ntype = \"lowpass\"\nfreq = 5", "stage 1 (lowpass): freq = 5 is outside 20 to 20000"),
            ("[[chain]]\ntype = \"saturate\"\nmode = \"fuzz\"", "stage 1 (saturate): mode = \"fuzz\" is not one of: soft, hard, tube"),
            ("[[chain]]\nfreq = 100", "stage 1 needs a type, for example type = \"lowpass\""),
            ("[macros]\n2x = 1\n[[chain]]\ntype = \"gain\"", "macro '2x': a name is letters, digits and _"),
            ("[macros]\na = { min = 1, max = 0 }\n[[chain]]\ntype = \"gain\"", "macro 'a': min must be below max"),
            ("name = \"Empty\"", "the filter has no [[chain]] stages"),
            ("nmae = \"x\"\n[[chain]]\ntype = \"gain\"", "unknown key 'nmae'"),
        ];
        for (i, (text, want)) in cases.iter().enumerate() {
            let def = FilterDef::parse(&format!("t_broken_{i}"), text);
            let err = def.error.clone().unwrap_or_else(|| panic!("{text:?} should not load"));
            assert!(err.starts_with(want), "{text:?}\n  got:  {err}\n  want: {want}…");
            assert!(def.stages.is_empty());
            upsert(def);
            let mut fx = Composite::new(&format!("filter:t_broken_{i}"));
            let src = tone(960, 300.0, 0.2);
            let mut x = src.clone();
            fx.process(&mut x);
            assert_eq!(x, src, "a broken filter passes audio through");
        }
        // So does a filter whose file is gone.
        let mut gone = Composite::new("filter:t_never_existed");
        let mut x = vec![0.25; 480];
        gone.process(&mut x);
        assert_eq!(x, vec![0.25; 480]);
    }

    #[test]
    fn stages_can_be_chain_effects_and_keys_forgive_unit_suffixes() {
        let def = FilterDef::parse(
            "t_native",
            r#"
            [[chain]]
            type = "compressor"
            threshold_db = -20   # the effect calls it "threshold"
            ratio = 4
            [[chain]]
            type = "gain"
            gain = -6            # the primitive calls it "gain_db"
            [[chain]]
            type = "saturate"
            mode = "Hard"
            drive = 2
            [[chain]]
            type = "reverb"
            mix = 0.1
            "#,
        );
        assert_eq!(def.error, None);
        upsert(def);
        let mut fx = Composite::new("filter:t_native");
        let mut x = tone(48_000, 300.0, 0.05);
        fx.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        // Quiet enough to stay under the compressor and the clipper: -6 dB from the gain stage,
        // +6 dB from the drive of 2 into the clipper (and a little reverb).
        let g = rms(&x[24_000..]) / rms(&tone(24_000, 300.0, 0.05));
        assert!((g - 1.0).abs() < 0.1, "gain {g}");
    }

    #[test]
    fn an_expression_that_blows_up_falls_back_and_stays_in_range() {
        upsert(FilterDef::parse(
            "t_div",
            "[macros]\nx = 0\n[[chain]]\ntype = \"lowpass\"\nfreq = \"1000 / x\"\n[[chain]]\ntype = \"gain\"\ngain_db = \"ln(x)\"",
        ));
        let mut fx = Composite::new("filter:t_div"); // x = 0: 1000/0 = inf, ln(0) = -inf
        let mut x = tone(4800, 300.0, 0.2);
        fx.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        fx.set(0, 0.5);
        fx.process(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn bad_mic_sounds_degraded_but_stays_sane() {
        upsert(FilterDef::parse("t_bad_mic", BAD_MIC));
        let mut fx = Composite::new("filter:t_bad_mic");
        for (i, m) in [1.0, 1.0, 1.0].iter().enumerate() {
            fx.set(i, *m);
        }
        // A voice-like buzz with content up to 12 kHz.
        let mut ph = 0.0f32;
        let mut x: Vec<f32> = (0..96_000)
            .map(|_| {
                ph += std::f32::consts::TAU * 150.0 / super::super::SR;
                (1..80).map(|h| (ph * h as f32).sin() / h as f32).sum::<f32>() * 0.15
            })
            .collect();
        let dry = x.clone();
        for frame in x.chunks_mut(480) {
            fx.process(frame);
        }
        assert!(x.iter().all(|v| v.is_finite()));
        let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 2.0, "peak {peak}");
        assert!(rms(&x[48_000..]) > 0.01, "something still comes through");
        // Telephone band: the 150 Hz fundamental is cut hard.
        let level_at = |sig: &[f32], f: f32| -> f32 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (i, v) in sig.iter().enumerate() {
                let w = std::f32::consts::TAU * f * i as f32 / super::super::SR;
                re += v * w.cos();
                im += v * w.sin();
            }
            (re * re + im * im).sqrt() / sig.len() as f32
        };
        let (a, b) = (level_at(&dry[48_000..], 150.0), level_at(&x[48_000..], 150.0));
        assert!(b < a * 0.5, "150 Hz: {a} -> {b}");
    }
}
