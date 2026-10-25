//! Tauri shell: exposes the Resona runtime to the web UI.
//!
//! Commands are thin wrappers; every runtime event is forwarded on the `resona` channel.

use resona_runtime::{DeviceInfo, Direction, Event, Runtime, SessionConfig, VoiceRef};
use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State};

struct AppState {
    rt: Runtime,
}

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

#[derive(Serialize)]
struct Devices {
    inputs: Vec<DeviceInfo>,
    outputs: Vec<DeviceInfo>,
}

#[tauri::command]
async fn list_devices() -> CmdResult<Devices> {
    tauri::async_runtime::spawn_blocking(|| {
        Ok(Devices {
            inputs: resona_runtime::list_devices(Direction::Capture).map_err(err)?,
            outputs: resona_runtime::list_devices(Direction::Render).map_err(err)?,
        })
    })
    .await
    .map_err(err)?
}

#[derive(Serialize)]
struct Status {
    state: resona_runtime::EngineState,
    message: Option<String>,
    engine_dir: Option<String>,
}

#[tauri::command]
fn get_status(st: State<'_, AppState>) -> Status {
    let (state, message) = st.rt.state();
    Status { state, message, engine_dir: st.rt.engine_dir() }
}

/// Run a blocking runtime call off the async executor.
async fn blocking<T, F>(st: &State<'_, AppState>, f: F) -> CmdResult<T>
where
    T: Send + 'static,
    F: FnOnce(Runtime) -> CmdResult<T> + Send + 'static,
{
    let rt = st.rt.clone();
    tauri::async_runtime::spawn_blocking(move || f(rt)).await.map_err(err)?
}

#[tauri::command]
async fn start_session(st: State<'_, AppState>, cfg: SessionConfig) -> CmdResult<()> {
    blocking(&st, move |rt| rt.start(cfg).map_err(err)).await
}

#[tauri::command]
async fn stop_session(st: State<'_, AppState>) -> CmdResult<()> {
    blocking(&st, |rt| {
        rt.stop();
        Ok(())
    })
    .await
}

#[tauri::command]
async fn set_params(st: State<'_, AppState>, params: Value) -> CmdResult<Value> {
    blocking(&st, move |rt| rt.set_params(params).map_err(err)).await
}

#[tauri::command]
async fn inspect_model(st: State<'_, AppState>, path: String) -> CmdResult<Value> {
    blocking(&st, move |rt| rt.inspect_model(&path).map_err(err)).await
}

#[tauri::command]
async fn calibrate(st: State<'_, AppState>, voice: VoiceRef, stream: Option<Value>) -> CmdResult<Value> {
    blocking(&st, move |rt| rt.calibrate(&voice, stream.unwrap_or(Value::Null), &[40, 60, 80, 100]).map_err(err)).await
}

#[tauri::command]
async fn activate_voice(st: State<'_, AppState>, voice: VoiceRef) -> CmdResult<()> {
    blocking(&st, move |rt| rt.activate_voice(&voice).map_err(err)).await
}

#[tauri::command]
async fn preload_voice(st: State<'_, AppState>, voice: VoiceRef) -> CmdResult<Value> {
    blocking(&st, move |rt| rt.preload_voice(&voice).map_err(err)).await
}

#[tauri::command]
fn set_muted(st: State<'_, AppState>, muted: bool) {
    st.rt.set_muted(muted);
}

#[tauri::command]
fn set_output_gain(st: State<'_, AppState>, db: f32) {
    st.rt.set_output_gain_db(db);
}

#[tauri::command]
fn set_monitor_gain(st: State<'_, AppState>, db: f32) {
    st.rt.set_monitor_gain_db(db);
}

#[tauri::command]
fn engine_log(st: State<'_, AppState>, lines: Option<usize>) -> Vec<String> {
    st.rt.engine_log(lines.unwrap_or(50))
}

#[tauri::command]
async fn assets_status(st: State<'_, AppState>) -> CmdResult<Value> {
    blocking(&st, |rt| rt.assets_status().map_err(err)).await
}

fn forward(app: &AppHandle, ev: Event) {
    let _ = app.emit("resona", ev);
}

pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            let stub = std::env::var("RESONA_STUB").is_ok();
            let rt = Runtime::new(move |ev| forward(&handle, ev), stub)?;
            app.manage(AppState { rt });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(st) = window.try_state::<AppState>() {
                    st.rt.shutdown();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            list_devices,
            get_status,
            start_session,
            stop_session,
            set_params,
            inspect_model,
            calibrate,
            activate_voice,
            preload_voice,
            set_muted,
            set_output_gain,
            set_monitor_gain,
            engine_log,
            assets_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Resona");
}
