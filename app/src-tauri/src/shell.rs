//! Desktop integration: tray icon, global hotkeys, close-to-tray.
//!
//! Mute and push-to-talk are handled here, in Rust, so they work while the window is hidden;
//! voice hotkeys are forwarded to the UI (it owns profiles and the library order).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutEvent, ShortcutState};

use crate::AppState;

#[derive(Default)]
pub struct Shell {
    pub muted: AtomicBool,
    pub close_to_tray: AtomicBool,
    bindings: Mutex<HashMap<u32, String>>,
    mute_item: Mutex<Option<MenuItem<tauri::Wry>>>,
}

#[derive(Deserialize)]
pub struct Binding {
    pub accel: String,
    pub action: String,
}

#[derive(Serialize)]
pub struct BindingError {
    pub accel: String,
    pub action: String,
    pub error: String,
}

pub fn set_muted(app: &AppHandle, muted: bool) {
    let st = app.state::<AppState>();
    st.shell.muted.store(muted, Ordering::Relaxed);
    st.rt.set_muted(muted);
    if let Some(item) = st.shell.mute_item.lock().as_ref() {
        let _ = item.set_text(if muted { "Unmute" } else { "Mute" });
    }
    let _ = app.emit("muted", json!({ "muted": muted }));
}

pub fn show_main(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

pub fn on_shortcut(app: &AppHandle, shortcut: &Shortcut, ev: ShortcutEvent) {
    let action = app.state::<AppState>().shell.bindings.lock().get(&shortcut.id()).cloned();
    let Some(action) = action else { return };
    let pressed = ev.state() == ShortcutState::Pressed;
    match action.as_str() {
        "mute" if pressed => {
            let muted = app.state::<AppState>().shell.muted.load(Ordering::Relaxed);
            set_muted(app, !muted);
        }
        "ptt" => set_muted(app, !pressed), // talk while held
        _ if pressed => {
            let _ = app.emit("hotkey", json!({ "action": action }));
        }
        _ => {}
    }
}

/// Replace all global hotkeys. Returns the bindings that could not be registered (bad syntax or
/// already taken by another app) so the UI can flag them.
pub fn apply(app: &AppHandle, bindings: Vec<Binding>) -> Vec<BindingError> {
    let gs = app.global_shortcut();
    let _ = gs.unregister_all();
    let mut map = HashMap::new();
    let mut errors = Vec::new();
    for b in bindings {
        if b.accel.trim().is_empty() {
            continue;
        }
        let res = b.accel.parse::<Shortcut>().map_err(|e| e.to_string()).and_then(|s| {
            if map.contains_key(&s.id()) {
                return Err("used by another action".to_string());
            }
            gs.register(s).map_err(|e| e.to_string()).map(|_| s)
        });
        match res {
            Ok(s) => {
                map.insert(s.id(), b.action);
            }
            Err(error) => errors.push(BindingError { accel: b.accel, action: b.action, error }),
        }
    }
    *app.state::<AppState>().shell.bindings.lock() = map;
    errors
}

pub fn build_tray(handle: &AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(handle, "show", "Show Resona", true, None::<&str>)?;
    let mute = MenuItem::with_id(handle, "mute", "Mute", true, None::<&str>)?;
    let quit = MenuItem::with_id(handle, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(handle, &[&show, &mute, &quit])?;
    *handle.state::<AppState>().shell.mute_item.lock() = Some(mute);
    let mut tray = TrayIconBuilder::with_id("main")
        .tooltip("Resona")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, ev| match ev.id.as_ref() {
            "show" => show_main(app),
            "mute" => {
                let muted = app.state::<AppState>().shell.muted.load(Ordering::Relaxed);
                set_muted(app, !muted);
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, ev| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = ev {
                show_main(tray.app_handle());
            }
        });
    if let Some(icon) = handle.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(handle)?;
    Ok(())
}
