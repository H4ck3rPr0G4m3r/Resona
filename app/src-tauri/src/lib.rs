//! Tauri shell: exposes the Resona runtime to the web UI.
//!
//! Commands are thin wrappers; every runtime event is forwarded on the `resona` channel.

mod library;
mod shell;
mod store;

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use resona_runtime::{DeviceInfo, Direction, Event, Runtime, SessionConfig, VoiceRef};
use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_autostart::{MacosLauncher, ManagerExt};

pub(crate) struct AppState {
    rt: Runtime,
    shell: shell::Shell,
    launched_minimized: bool,
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
async fn activate_voice(st: State<'_, AppState>, voice: VoiceRef, params: Option<Value>, xfade_ms: Option<f32>) -> CmdResult<()> {
    blocking(&st, move |rt| rt.activate_voice(&voice, params.unwrap_or(Value::Null), xfade_ms.unwrap_or(50.0)).map_err(err)).await
}

#[tauri::command]
async fn preload_voice(st: State<'_, AppState>, voice: VoiceRef) -> CmdResult<Value> {
    blocking(&st, move |rt| rt.preload_voice(&voice).map_err(err)).await
}

#[tauri::command]
fn set_muted(app: AppHandle, muted: bool) {
    shell::set_muted(&app, muted);
}

#[tauri::command]
fn get_muted(st: State<'_, AppState>) -> bool {
    st.shell.muted.load(Ordering::Relaxed)
}

// -- persistence ------------------------------------------------------------------------------

#[tauri::command]
fn settings_load() -> Value {
    store::load_settings()
}

#[tauri::command]
fn settings_save(settings: Value) -> CmdResult<()> {
    store::write_json(&store::settings_path(), &settings).map_err(err)
}

#[tauri::command]
fn profile_load(hash: String) -> Option<Value> {
    store::profile_path(&hash).and_then(|p| store::read_json(&p))
}

#[tauri::command]
fn profile_save(hash: String, profile: Value) -> CmdResult<()> {
    let p = store::profile_path(&hash).ok_or("bad voice id")?;
    store::write_json(&p, &profile).map_err(err)
}

// -- library ----------------------------------------------------------------------------------

#[tauri::command]
fn library_list() -> Vec<Value> {
    library::list()
}

#[tauri::command]
fn library_update(id: String, patch: Value) -> CmdResult<Value> {
    library::update(&id, &patch).map_err(err)
}

#[tauri::command]
fn library_remove(id: String) -> CmdResult<()> {
    library::remove(&id).map_err(err)
}

/// Import files, folders or zips. Each model is inspected by the engine (which rejects unsupported
/// architectures with a reason), then copied into the library with its paired index.
#[tauri::command]
async fn library_import(app: AppHandle, st: State<'_, AppState>, paths: Vec<String>) -> CmdResult<library::ImportReport> {
    blocking(&st, move |rt| {
        let scratch = store::data_root().join("cache").join(format!("import-{}", std::process::id()));
        let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
        let (cands, failed) = library::collect(&paths, &scratch);
        let mut rep = library::ImportReport { failed, ..Default::default() };
        let total = cands.len();
        for (i, c) in cands.iter().enumerate() {
            let name = c.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let _ = app.emit("import", json!({"done": i, "total": total, "name": name}));
            match rt.inspect_model(&c.model.to_string_lossy()) {
                Err(e) => rep.failed.push(json!({"path": c.model, "error": e.to_string()})),
                Ok(info) => match library::install(c, &info) {
                    Ok(Some(v)) => rep.added.push(v),
                    Ok(None) => rep.skipped.push(json!({"path": c.model, "reason": "already in the library"})),
                    Err(e) => rep.failed.push(json!({"path": c.model, "error": e.to_string()})),
                },
            }
        }
        let _ = app.emit("import", json!({"done": total, "total": total, "name": null}));
        if scratch.starts_with(store::data_root()) {
            let _ = std::fs::remove_dir_all(&scratch);
        }
        Ok(rep)
    })
    .await
}

// -- shell ------------------------------------------------------------------------------------

#[tauri::command]
fn apply_hotkeys(app: AppHandle, bindings: Vec<shell::Binding>) -> Vec<shell::BindingError> {
    shell::apply(&app, bindings)
}

#[tauri::command]
fn set_shell_prefs(app: AppHandle, st: State<'_, AppState>, close_to_tray: bool, autostart: bool) -> CmdResult<()> {
    st.shell.close_to_tray.store(close_to_tray, Ordering::Relaxed);
    let al = app.autolaunch();
    let on = al.is_enabled().unwrap_or(false);
    if autostart != on {
        if autostart { al.enable() } else { al.disable() }.map_err(err)?;
    }
    Ok(())
}

#[tauri::command]
fn launched_minimized(st: State<'_, AppState>) -> bool {
    st.launched_minimized
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
    let minimized = std::env::args().any(|a| a == "--minimized");
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| shell::show_main(app)))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(MacosLauncher::LaunchAgent, Some(vec!["--minimized"])))
        .plugin(tauri_plugin_global_shortcut::Builder::new().with_handler(shell::on_shortcut).build())
        .setup(move |app| {
            let handle = app.handle().clone();
            let stub = std::env::var("RESONA_STUB").is_ok();
            let rt = Runtime::new(move |ev| forward(&handle, ev), stub)?;
            app.manage(AppState { rt, shell: shell::Shell::default(), launched_minimized: minimized });
            shell::build_tray(app.handle())?;
            if minimized {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.state::<AppState>().shell.close_to_tray.load(Ordering::Relaxed) {
                    api.prevent_close();
                    let _ = window.hide();
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
            get_muted,
            set_output_gain,
            set_monitor_gain,
            engine_log,
            assets_status,
            settings_load,
            settings_save,
            profile_load,
            profile_save,
            library_list,
            library_update,
            library_remove,
            library_import,
            apply_hotkeys,
            set_shell_prefs,
            launched_minimized,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Resona");
    app.run(|app, ev| {
        if let tauri::RunEvent::Exit = ev {
            if let Some(st) = app.try_state::<AppState>() {
                st.rt.shutdown();
            }
        }
    });
}
