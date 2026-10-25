//! Resona host runtime: owns the engine worker, the audio pipeline and the session state machine.
//!
//! ```text
//! Stopped → Starting → Loading → Running
//!              ▲                    │ worker exits
//!              └──── Restarting ◄───┘   (≤ 3 restarts / min, else Failed)
//! DeviceLost (audio device error while running) → retry every 2 s → Running
//! ```

pub mod fx;
pub mod pipeline;
pub mod player;
pub mod recorder;
pub mod rpc;
pub mod selftest;
pub mod worker;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub use pipeline::{DownMode, Layout, PipelineInfo};
use pipeline::{DeviceConfig, DspCmd, Pipeline, Shared, Transport};
pub use resona_audio_io::{list_devices, DeviceInfo, Direction, StreamConfig as DeviceStreamConfig, StreamMode};
use rpc::RpcError;
use worker::{locate_engine, EngineLocation, WorkerProcess};

const T_FAST: Duration = Duration::from_secs(10);
const T_SLOW: Duration = Duration::from_secs(240);
const MAX_RESTARTS_PER_MIN: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineState {
    Stopped,
    Starting,
    Loading,
    Running,
    Restarting,
    DeviceLost,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    State { state: EngineState, message: Option<String> },
    Stats(Value),
    Meters { input_peak: f32, input_rms: f32, output_peak: f32, output_rms: f32 },
    Pitch { input: Vec<f32>, output: Vec<f32> },
    Progress { what: String, done: u64, total: u64 },
    Log { line: String },
    /// A recording finished because the session stopped.
    Take(recorder::Take),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceRef {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub index_path: Option<String>,
}

fn default_jitter() -> f32 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default)]
    pub input: DeviceStreamConfig,
    #[serde(default)]
    pub output: DeviceStreamConfig,
    #[serde(default)]
    pub monitor: Option<DeviceStreamConfig>,
    /// Engine `StreamConfig` fields (block_ms, lookahead_ms, …).
    #[serde(default)]
    pub stream: Value,
    pub voice: VoiceRef,
    /// Engine `LiveParams` fields (pitch, formant, index_rate, …).
    #[serde(default)]
    pub params: Value,
    #[serde(default = "default_jitter")]
    pub jitter_ms: f32,
    #[serde(default)]
    pub down_mode: DownMode,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("engine not found: expected engine/.venv next to the app (set RESONA_ENGINE_DIR)")]
    EngineMissing,
    #[error("could not start engine: {0}")]
    Spawn(String),
    #[error(transparent)]
    Rpc(#[from] RpcError),
    #[error(transparent)]
    Audio(#[from] resona_audio_io::AudioError),
    #[error("shared memory: {0}")]
    Shm(#[from] resona_shm_ipc::ShmError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

struct Session {
    cfg: SessionConfig,
    layout: Layout,
    pipeline: Option<Pipeline>,
}

struct Inner {
    emit: Box<dyn Fn(Event) + Send + Sync>,
    stub: bool,
    prefix: String,
    engine: Option<EngineLocation>,
    transport: Arc<Transport>,
    shared: Arc<Shared>,
    worker: Mutex<Option<Arc<WorkerProcess>>>,
    generation: AtomicU64,
    session: Mutex<Option<Session>>,
    state: Mutex<(EngineState, Option<String>)>,
    restarts: Mutex<VecDeque<Instant>>,
    /// Serializes start/stop/recovery so they never interleave.
    control: Mutex<()>,
    /// The latency bar's last total, for comparing with the self-test.
    last_estimate_ms: Mutex<Option<f32>>,
}

#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}

impl Runtime {
    pub fn new<F>(emit: F, stub: bool) -> Result<Self>
    where
        F: Fn(Event) + Send + Sync + 'static,
    {
        let prefix = format!("Local\\resona-{}", std::process::id());
        let transport = Arc::new(Transport::create(&prefix)?);
        let inner = Arc::new(Inner {
            emit: Box::new(emit),
            stub,
            prefix,
            engine: locate_engine(),
            transport,
            shared: Arc::new(Shared::default()),
            worker: Mutex::new(None),
            generation: AtomicU64::new(0),
            session: Mutex::new(None),
            state: Mutex::new((EngineState::Stopped, None)),
            restarts: Mutex::new(VecDeque::new()),
            control: Mutex::new(()),
            last_estimate_ms: Mutex::new(None),
        });
        let rt = Self { inner };
        rt.spawn_meter_thread();
        Ok(rt)
    }

    pub fn engine_dir(&self) -> Option<String> {
        self.inner.engine.as_ref().map(|e| e.dir.display().to_string())
    }

    pub fn state(&self) -> (EngineState, Option<String>) {
        self.inner.state.lock().clone()
    }

    fn set_state(&self, s: EngineState, msg: Option<String>) {
        *self.inner.state.lock() = (s, msg.clone());
        (self.inner.emit)(Event::State { state: s, message: msg });
    }

    pub fn shared(&self) -> &Shared {
        &self.inner.shared
    }

    // -- worker ---------------------------------------------------------------------------------

    fn worker(&self) -> Result<Arc<WorkerProcess>> {
        if let Some(w) = self.inner.worker.lock().as_ref() {
            if w.is_alive() {
                return Ok(w.clone());
            }
        }
        let loc = self.inner.engine.as_ref().ok_or(RuntimeError::EngineMissing)?;
        let gen = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let rt_n = self.clone();
        let rt_x = self.clone();
        let w = WorkerProcess::spawn(
            loc,
            &self.inner.prefix,
            self.inner.stub,
            move |method, params| rt_n.on_notify(method, params),
            move || rt_x.on_worker_exit(gen),
        )
        .map_err(|e| RuntimeError::Spawn(e.to_string()))?;
        let w = Arc::new(w);
        *self.inner.worker.lock() = Some(w.clone());
        w.rpc.call("hello", json!({}), Duration::from_secs(90))?;
        Ok(w)
    }

    fn call(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        Ok(self.worker()?.rpc.call(method, params, timeout)?)
    }

    fn on_notify(&self, method: &str, params: Value) {
        match method {
            "stats" => (self.inner.emit)(Event::Stats(self.merge_stats(params))),
            "pitch" => {
                let arr = |k: &str| -> Vec<f32> {
                    params.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect()).unwrap_or_default()
                };
                (self.inner.emit)(Event::Pitch { input: arr("input"), output: arr("output") });
            }
            "assets.progress" => (self.inner.emit)(Event::Progress {
                what: params.get("file").and_then(Value::as_str).unwrap_or("").into(),
                done: params.get("done").and_then(Value::as_u64).unwrap_or(0),
                total: params.get("total").and_then(Value::as_u64).unwrap_or(0),
            }),
            "stream.error" => {
                let msg = params.get("message").and_then(Value::as_str).unwrap_or("engine error").to_string();
                log::error!("engine stream error: {msg}");
                (self.inner.emit)(Event::Log { line: msg.clone() });
                // The worker's loop stopped; treat it like a crash so recovery kicks in.
                if let Some(w) = self.inner.worker.lock().take() {
                    w.kill();
                }
            }
            _ => {}
        }
    }

    fn merge_stats(&self, mut engine: Value) -> Value {
        let s = &self.inner.shared;
        let sess = self.inner.session.lock();
        if let (Some(obj), Some(sess)) = (engine.as_object_mut(), sess.as_ref()) {
            let lookahead = sess.cfg.stream.get("lookahead_ms").and_then(Value::as_f64).unwrap_or(40.0) as f32;
            let p50 = obj.get("p50_ms").and_then(Value::as_f64).unwrap_or(0.0) as f32;
            let (input_ms, output_ms) = sess.pipeline.as_ref().map(|p| (p.info.input.latency_ms, p.info.output.latency_ms)).unwrap_or((10.0, 10.0));
            let fifo_ms = s.fifo_samples.load(Ordering::Relaxed) as f32 / 48.0;
            let fx_ms = (s.fx.input_latency.load(Ordering::Relaxed) + s.fx.output_latency.load(Ordering::Relaxed)) as f32 / 48.0;
            let jitter_ms = match s.jitter_samples.load(Ordering::Relaxed) {
                0 => sess.cfg.jitter_ms,
                n => n as f32 / 48.0,
            };
            let total_ms = input_ms + sess.layout.block_ms() + lookahead + p50 + jitter_ms + fx_ms + output_ms + 1.0;
            *self.inner.last_estimate_ms.lock() = Some(total_ms);
            let latency = json!({
                "input_ms": input_ms,
                "block_ms": sess.layout.block_ms(),
                "lookahead_ms": lookahead,
                "compute_ms": p50,
                "buffer_ms": jitter_ms,
                "effects_ms": fx_ms,
                "output_ms": output_ms + 1.0,
                "total_ms": total_ms,
            });
            obj.insert("latency".into(), latency);
            obj.insert("underruns".into(), json!(s.underruns.load(Ordering::Relaxed)));
            obj.insert("fifo_ms".into(), json!(fifo_ms));
            obj.insert("sola_ncc".into(), json!(s.sola_ncc.load()));
            obj.insert("gate_open".into(), json!(s.fx.gate_open.load(Ordering::Relaxed)));
            obj.insert("blocks_sent".into(), json!(s.blocks_sent.load(Ordering::Relaxed)));
            obj.insert("ring_full".into(), json!(s.ring_full.load(Ordering::Relaxed)));
            obj.insert("captured".into(), json!(s.captured.load(Ordering::Relaxed)));
            obj.insert("rendered".into(), json!(s.rendered.load(Ordering::Relaxed)));
            obj.insert("blocks_received".into(), json!(s.blocks_received.load(Ordering::Relaxed)));
            if let Some(p) = &sess.pipeline {
                obj.insert("devices".into(), serde_json::to_value(&p.info).unwrap_or(Value::Null));
            }
        }
        engine
    }

    fn on_worker_exit(&self, gen: u64) {
        if self.inner.generation.load(Ordering::SeqCst) != gen {
            return; // an older worker we already replaced
        }
        self.inner.shared.engine_live.store(false, Ordering::Release);
        let tail = self.inner.worker.lock().take().map(|w| w.log_tail(12)).unwrap_or_default();
        if self.inner.session.lock().is_none() {
            return;
        }
        let now = Instant::now();
        let allowed = {
            let mut r = self.inner.restarts.lock();
            r.retain(|t| now.duration_since(*t) < Duration::from_secs(60));
            r.push_back(now);
            r.len() <= MAX_RESTARTS_PER_MIN
        };
        if !allowed {
            self.set_state(EngineState::Failed, Some(format!("engine keeps crashing:\n{}", tail.join("\n"))));
            return;
        }
        self.set_state(EngineState::Restarting, tail.last().cloned());
        let rt = self.clone();
        std::thread::spawn(move || {
            let _guard = rt.inner.control.lock();
            let cfg = match rt.inner.session.lock().as_ref() {
                Some(s) => s.cfg.clone(),
                None => return,
            };
            match rt.bring_up_engine(&cfg) {
                Ok(layout) => {
                    if let Some(s) = rt.inner.session.lock().as_mut() {
                        if layout != s.layout {
                            if let Some(p) = &s.pipeline {
                                let _ = p.cmds.send(DspCmd::Layout(layout));
                            }
                            s.layout = layout;
                        } else if let Some(p) = &s.pipeline {
                            let _ = p.cmds.send(DspCmd::Prime);
                        }
                    }
                    rt.inner.shared.engine_live.store(true, Ordering::Release);
                    rt.set_state(EngineState::Running, None);
                }
                Err(e) => rt.set_state(EngineState::Failed, Some(e.to_string())),
            }
        });
    }

    /// Load + activate the voice, configure the stream and start the engine loop.
    fn bring_up_engine(&self, cfg: &SessionConfig) -> Result<Layout> {
        self.set_state(EngineState::Loading, Some(format!("loading {}", cfg.voice.id)));
        self.call(
            "model.load",
            json!({"id": cfg.voice.id, "path": cfg.voice.path, "index_path": cfg.voice.index_path}),
            T_SLOW,
        )?;
        self.call("model.activate", json!({"id": cfg.voice.id}), T_SLOW)?;
        if !cfg.params.is_null() {
            self.call("params.set", cfg.params.clone(), T_FAST)?;
        }
        let stream = if cfg.stream.is_null() { json!({}) } else { cfg.stream.clone() };
        let layout: Layout = serde_json::from_value(self.call("stream.configure", json!({"config": stream}), T_SLOW)?)
            .map_err(|e| RuntimeError::Other(format!("bad layout from engine: {e}")))?;
        self.call("stream.start", json!({}), T_FAST)?;
        Ok(layout)
    }

    // -- session control ------------------------------------------------------------------------

    pub fn start(&self, cfg: SessionConfig) -> Result<()> {
        let _guard = self.inner.control.lock();
        self.stop_locked();
        self.inner.restarts.lock().clear();
        self.set_state(EngineState::Starting, None);
        let res = (|| -> Result<()> {
            self.worker()?;
            let layout = self.bring_up_engine(&cfg)?;
            let shared = self.inner.shared.clone();
            shared.set_down_mode(cfg.down_mode);
            let pipeline = self.start_pipeline(&cfg, layout)?;
            shared.engine_live.store(true, Ordering::Release);
            *self.inner.session.lock() = Some(Session { cfg, layout, pipeline: Some(pipeline) });
            Ok(())
        })();
        match &res {
            Ok(()) => self.set_state(EngineState::Running, None),
            Err(e) => {
                let _ = self.call("stream.stop", json!({}), T_FAST);
                self.set_state(EngineState::Failed, Some(e.to_string()));
            }
        }
        res
    }

    fn start_pipeline(&self, cfg: &SessionConfig, layout: Layout) -> Result<Pipeline> {
        let dev = DeviceConfig { input: cfg.input.clone(), output: cfg.output.clone(), monitor: cfg.monitor.clone(), jitter_ms: cfg.jitter_ms };
        let lookahead_ms = cfg.stream.get("lookahead_ms").and_then(Value::as_f64).unwrap_or(40.0);
        self.inner.shared.fx.lookahead.store((lookahead_ms * 48.0) as usize, Ordering::Relaxed);
        let rt = self.clone();
        let on_err = move |msg: String| rt.on_device_error(msg);
        Ok(Pipeline::start(self.inner.transport.clone(), self.inner.shared.clone(), layout, dev, on_err)?)
    }

    fn on_device_error(&self, msg: String) {
        log::warn!("audio device error: {msg}");
        if self.inner.session.lock().is_none() || self.state().0 == EngineState::DeviceLost {
            return;
        }
        self.set_state(EngineState::DeviceLost, Some(msg));
        let rt = self.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(2));
            let _guard = rt.inner.control.lock();
            let mut sess = rt.inner.session.lock();
            let Some(s) = sess.as_mut() else { return };
            if let Some(mut p) = s.pipeline.take() {
                p.stop();
            }
            match rt.start_pipeline(&s.cfg, s.layout) {
                Ok(p) => {
                    s.pipeline = Some(p);
                    drop(sess);
                    let live = rt.inner.shared.engine_live.load(Ordering::Acquire);
                    rt.set_state(if live { EngineState::Running } else { EngineState::Restarting }, None);
                    return;
                }
                Err(e) => {
                    drop(sess);
                    rt.set_state(EngineState::DeviceLost, Some(format!("waiting for device: {e}")));
                }
            }
        });
    }

    fn stop_locked(&self) {
        self.inner.shared.engine_live.store(false, Ordering::Release);
        if self.inner.shared.recorder.recording() {
            // Finish the take while audio still flows, so the wet side catches up.
            match self.inner.shared.recorder.stop() {
                Ok(take) => (self.inner.emit)(Event::Take(take)),
                Err(e) => log::warn!("recording: {e}"),
            }
        }
        let sess = self.inner.session.lock().take();
        if let Some(mut s) = sess {
            if let Some(mut p) = s.pipeline.take() {
                p.stop();
            }
            if let Some(w) = self.inner.worker.lock().clone() {
                let _ = w.rpc.call("stream.stop", json!({}), T_FAST);
            }
        }
        self.set_state(EngineState::Stopped, None);
    }

    pub fn stop(&self) {
        let _guard = self.inner.control.lock();
        self.stop_locked();
    }

    pub fn shutdown(&self) {
        self.stop();
        if let Some(w) = self.inner.worker.lock().take() {
            let _ = w.rpc.call("engine.shutdown", json!({}), Duration::from_secs(2));
            w.kill();
        }
    }

    // -- live controls ------------------------------------------------------------------------

    pub fn set_params(&self, params: Value) -> Result<Value> {
        let merged = self.call("params.set", params.clone(), T_FAST)?;
        if let Some(s) = self.inner.session.lock().as_mut() {
            s.cfg.params = merged.clone();
        }
        Ok(merged)
    }

    pub fn inspect_model(&self, path: &str) -> Result<Value> {
        self.call("model.inspect", json!({"path": path}), T_SLOW)
    }

    /// Time the engine at several block sizes for `voice` and recommend the smallest that keeps
    /// p99 compute within `budget` × block. Only while stopped (it borrows the GPU exclusively).
    pub fn calibrate(&self, voice: &VoiceRef, stream: Value, blocks: &[u32]) -> Result<Value> {
        let _guard = self.inner.control.lock();
        if self.inner.session.lock().is_some() {
            return Err(RuntimeError::Other("stop the session before calibrating".into()));
        }
        self.set_state(EngineState::Loading, Some(format!("calibrating {}", voice.id)));
        let res = (|| -> Result<Value> {
            self.call("model.load", json!({"id": voice.id, "path": voice.path, "index_path": voice.index_path}), T_SLOW)?;
            self.call("model.activate", json!({"id": voice.id}), T_SLOW)?;
            let stream = if stream.is_null() { json!({}) } else { stream };
            self.call("stream.configure", json!({"config": stream}), T_SLOW)?;
            self.call("engine.calibrate", json!({"blocks": blocks, "n": 80, "budget": 0.6}), T_SLOW)
        })();
        self.set_state(EngineState::Stopped, res.as_ref().err().map(|e| e.to_string()));
        res
    }

    pub fn preload_voice(&self, voice: &VoiceRef) -> Result<Value> {
        self.call("model.load", json!({"id": voice.id, "path": voice.path, "index_path": voice.index_path}), T_SLOW)
    }

    /// Switch the active voice of a running session.
    /// Switch the active voice (and optionally its profile params). While running this is
    /// seamless: the engine crossfades old -> new over `xfade_ms` without a layout change.
    pub fn activate_voice(&self, voice: &VoiceRef, params: Value, xfade_ms: f32) -> Result<()> {
        let _guard = self.inner.control.lock();
        self.preload_voice(voice)?;
        let mut req = json!({"id": voice.id, "xfade_ms": xfade_ms});
        if !params.is_null() {
            req["params"] = params.clone();
        }
        let res = self.call("model.activate", req, T_SLOW)?;
        if let Some(s) = self.inner.session.lock().as_mut() {
            s.cfg.voice = voice.clone();
            if !params.is_null() {
                s.cfg.params = params;
            }
        }
        if let Some(layout) = res.get("layout").cloned().and_then(|l| serde_json::from_value::<Layout>(l).ok()) {
            if let Some(s) = self.inner.session.lock().as_mut() {
                if layout != s.layout {
                    if let Some(p) = &s.pipeline {
                        let _ = p.cmds.send(DspCmd::Layout(layout));
                    }
                    s.layout = layout;
                }
            }
        }
        Ok(())
    }

    pub fn set_muted(&self, muted: bool) {
        self.inner.shared.muted.store(muted, Ordering::Relaxed);
    }

    pub fn set_output_gain_db(&self, db: f32) {
        self.inner.shared.output_gain.store(resona_dsp::from_db(db));
    }

    pub fn set_monitor_gain_db(&self, db: f32) {
        self.inner.shared.monitor_gain.store(resona_dsp::from_db(db));
    }

    // -- effects ----------------------------------------------------------------------------

    /// Effect catalog (kinds, parameter tables) for generating the chain editor.
    pub fn fx_catalog() -> Value {
        serde_json::to_value(resona_dsp::fx::CATALOG).unwrap_or(Value::Null)
    }

    pub fn fx_settings(&self) -> fx::FxSettings {
        self.inner.shared.fx.settings()
    }

    /// Replace chains / gate / limiter / singing mode. Structural chain edits crossfade in 20 ms.
    pub fn set_fx(&self, settings: fx::FxSettings) -> Result<()> {
        self.inner.shared.fx.update(settings).map_err(RuntimeError::Other)
    }

    /// Fast path for dragging one control: no chain rebuild, no validation of the rest.
    pub fn set_fx_param(&self, chain: &str, slot: u64, key: &str, value: f32) -> Result<()> {
        self.inner.shared.fx.set_param(chain, slot, key, value).map_err(RuntimeError::Other)
    }

    // -- sound-file input ------------------------------------------------------------------

    /// Decode a WAV/MP3/FLAC file and load it as the input source (paused).
    pub fn player_load(&self, path: &std::path::Path) -> Result<player::PlayerState> {
        let samples = player::decode(path).map_err(RuntimeError::Other)?;
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.inner.shared.player.load(name, samples);
        Ok(self.inner.shared.player.state())
    }

    pub fn player_unload(&self) {
        self.inner.shared.player.unload();
    }

    pub fn player_play(&self, on: bool) {
        self.inner.shared.player.play(on);
    }

    pub fn player_seek(&self, seconds: f32) {
        self.inner.shared.player.seek_seconds(seconds);
    }

    pub fn player_set_loop(&self, on: bool) {
        self.inner.shared.player.looping.store(on, Ordering::Relaxed);
    }

    pub fn player_state(&self) -> player::PlayerState {
        self.inner.shared.player.state()
    }

    // -- A/B recorder ---------------------------------------------------------------------

    /// Start writing `<stem> dry.wav` + `<stem> wet.wav` into `dir` (needs a running session).
    pub fn record_start(&self, dir: std::path::PathBuf, stem: String) -> Result<()> {
        if self.inner.session.lock().as_ref().and_then(|s| s.pipeline.as_ref()).is_none() {
            return Err(RuntimeError::Other("start the voice changer first".into()));
        }
        self.inner.shared.recorder.start(dir, stem).map_err(RuntimeError::Other)
    }

    pub fn record_stop(&self) -> Result<recorder::Take> {
        self.inner.shared.recorder.stop().map_err(RuntimeError::Other)
    }

    /// Seconds recorded so far, or `None` when not recording.
    pub fn recording(&self) -> Option<f32> {
        let r = &self.inner.shared.recorder;
        r.recording().then(|| r.elapsed_s())
    }

    // -- latency self-test ----------------------------------------------------------------

    /// Measure the real delay from input to output with test bursts (about 6 s; the mic is
    /// muted meanwhile). `listen`: a recording device to listen on; `None` = output loopback.
    pub fn latency_self_test(&self, listen: Option<String>) -> Result<selftest::SelfTestResult> {
        use selftest::{Capture, BURSTS, PERIOD};
        let output = {
            let sess = self.inner.session.lock();
            let s = sess
                .as_ref()
                .filter(|s| s.pipeline.is_some())
                .ok_or_else(|| RuntimeError::Other("start the voice changer first".into()))?;
            s.cfg.output.clone()
        };
        if self.state().0 != EngineState::Running {
            return Err(RuntimeError::Other("the voice changer is not running yet".into()));
        }
        let shared = &self.inner.shared;
        if shared.selftest.active.load(Ordering::Acquire) {
            return Err(RuntimeError::Other("a self-test is already running".into()));
        }
        let cfg = match &listen {
            Some(id) => DeviceStreamConfig { device_id: Some(id.clone()), mode: StreamMode::Shared, loopback: false },
            None => DeviceStreamConfig { device_id: output.device_id.clone(), mode: StreamMode::Shared, loopback: true },
        };
        let total = (BURSTS + 3) * PERIOD;
        let cap = Arc::new(Mutex::new(Capture::with_capacity(total)));
        let sink = cap.clone();
        let stream = resona_audio_io::CaptureStream::start_timed(
            cfg,
            move |x, qpc| {
                if let Some(mut c) = sink.try_lock() {
                    c.push(x, qpc);
                }
            },
            |e| log::warn!("self-test listen stream: {e}"),
        )?;
        let listen_name = stream.info().device_name.clone();
        std::thread::sleep(Duration::from_millis(300));
        shared.selftest.arm();
        let deadline = Instant::now() + Duration::from_millis((BURSTS * 1000 + 3000) as u64);
        while !shared.selftest.done.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        shared.selftest.active.store(false, Ordering::Release);
        // The last burst is still on its way through the engine and buffers.
        std::thread::sleep(Duration::from_millis(900));
        stream.stop();
        if !shared.selftest.done.load(Ordering::Acquire) {
            return Err(RuntimeError::Other("the microphone stream stopped during the test".into()));
        }
        let bursts: Vec<u64> = shared.selftest.burst_qpc.iter().map(|t| t.load(Ordering::Relaxed)).collect();
        if bursts.iter().any(|&q| q == 0) {
            return Err(RuntimeError::Other("the input device gives no timestamps, so the self-test cannot run on it".into()));
        }
        let cap = cap.lock();
        let runs = selftest::analyze(&cap, &bursts).map_err(RuntimeError::Other)?;
        let estimate = *self.inner.last_estimate_ms.lock();
        let listen = if listen.is_some() { listen_name } else { format!("{listen_name} (loopback)") };
        Ok(selftest::summarize(runs, estimate, listen))
    }

    pub fn set_down_mode(&self, m: DownMode) {
        self.inner.shared.set_down_mode(m);
    }

    pub fn engine_log(&self, n: usize) -> Vec<String> {
        self.inner.worker.lock().as_ref().map(|w| w.log_tail(n)).unwrap_or_default()
    }

    pub fn assets_status(&self) -> Result<Value> {
        self.call("assets.status", json!({}), T_SLOW)
    }

    pub fn assets_fetch(&self) -> Result<Value> {
        self.call("assets.fetch", json!({}), Duration::from_secs(3600))
    }

    fn spawn_meter_thread(&self) {
        let weak = Arc::downgrade(&self.inner);
        std::thread::Builder::new()
            .name("resona-meters".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_millis(33));
                let Some(inner) = weak.upgrade() else { return };
                if inner.session.lock().is_none() {
                    continue;
                }
                let s = &inner.shared;
                (inner.emit)(Event::Meters {
                    input_peak: s.in_meter.peak.load(),
                    input_rms: s.in_meter.rms.load(),
                    output_peak: s.out_meter.peak.load(),
                    output_rms: s.out_meter.rms.load(),
                });
            })
            .expect("spawn meter thread");
    }
}
