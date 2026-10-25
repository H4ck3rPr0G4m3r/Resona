//! Persistence under `%APPDATA%\Resona` (override: `RESONA_APPDATA`).
//!
//! `settings.json` and `profiles\<model_hash>.json` are owned by the UI (their shape lives in
//! `app/src/lib/settings.ts`); the shell only reads the few keys it acts on. Every file carries a
//! `schema` field. Writes go through a temp file + rename so a crash never leaves half a file.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

pub const SCHEMA: u64 = 1;

pub fn data_root() -> PathBuf {
    if let Ok(p) = std::env::var("RESONA_APPDATA") {
        return PathBuf::from(p);
    }
    let base = std::env::var("APPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
    base.join("Resona")
}

pub fn read_json(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => Some(v),
        Err(e) => {
            log::warn!("{}: unreadable ({e}); keeping a .bad copy", path.display());
            let _ = fs::copy(path, path.with_extension("json.bad"));
            None
        }
    }
}

pub fn write_json(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let mut v = value.clone();
    if let Some(obj) = v.as_object_mut() {
        obj.entry("schema").or_insert(json!(SCHEMA));
    }
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&v)?)?;
    fs::rename(&tmp, path)
}

pub fn settings_path() -> PathBuf {
    data_root().join("settings.json")
}

pub fn load_settings() -> Value {
    read_json(&settings_path()).unwrap_or_else(|| json!({ "schema": SCHEMA }))
}

pub fn profile_path(hash: &str) -> Option<PathBuf> {
    // Hashes are hex; refuse anything that could escape the profiles directory.
    if hash.is_empty() || !hash.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(data_root().join("profiles").join(format!("{hash}.json")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_adds_schema_and_rejects_bad_hash() {
        let dir = std::env::temp_dir().join(format!("resona-store-{}", std::process::id()));
        let p = dir.join("x.json");
        write_json(&p, &json!({"a": 1})).unwrap();
        let v = read_json(&p).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["schema"], SCHEMA);
        fs::write(&p, "{not json").unwrap();
        assert!(read_json(&p).is_none());
        assert!(dir.join("x.json.bad").exists());
        assert!(profile_path("..\\evil").is_none());
        assert!(profile_path("c997625737d4fcf6").is_some());
        let _ = fs::remove_dir_all(dir);
    }
}
