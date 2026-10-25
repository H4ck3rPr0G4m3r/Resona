//! Voice library: `%APPDATA%\Resona\voices\<model_hash>\` holds a copy of the model, its paired
//! index and `voice.json` (metadata + user fields). The library owns its files, so moving or
//! deleting the originals never breaks a voice.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};

use crate::store::{data_root, read_json, write_json};

pub fn voices_dir() -> PathBuf {
    data_root().join("voices")
}

fn voice_dir(id: &str) -> Option<PathBuf> {
    (!id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric())).then(|| voices_dir().join(id))
}

/// Everything the UI shows for one voice (voice.json plus resolved absolute paths).
pub fn list() -> Vec<Value> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(voices_dir()) else { return out };
    for e in rd.flatten() {
        let dir = e.path();
        let Some(mut v) = read_json(&dir.join("voice.json")) else { continue };
        let file = |k: &str| v.get(k).and_then(Value::as_str).map(|f| dir.join(f));
        let model = file("model_file");
        if !model.as_ref().is_some_and(|p| p.is_file()) {
            continue; // incomplete import
        }
        let index = file("index_file").filter(|p| p.is_file());
        v["path"] = json!(model.map(|p| p.to_string_lossy().into_owned()));
        v["index_path"] = json!(index.map(|p| p.to_string_lossy().into_owned()));
        out.push(v);
    }
    out.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_lowercase());
    out
}

pub fn update(id: &str, patch: &Value) -> io::Result<Value> {
    let dir = voice_dir(id).ok_or_else(|| io::Error::other("bad voice id"))?;
    let path = dir.join("voice.json");
    let mut v = read_json(&path).ok_or_else(|| io::Error::other("voice not found"))?;
    for key in ["favorite", "hotkey", "name"] {
        if let Some(x) = patch.get(key) {
            v[key] = x.clone();
        }
    }
    write_json(&path, &v)?;
    Ok(v)
}

pub fn remove(id: &str) -> io::Result<()> {
    let dir = voice_dir(id).ok_or_else(|| io::Error::other("bad voice id"))?;
    if dir.join("voice.json").is_file() {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

// -- import ----------------------------------------------------------------------------------

#[derive(Serialize, Default, Debug)]
pub struct ImportReport {
    pub added: Vec<Value>,
    pub skipped: Vec<Value>, // {path, reason}
    pub failed: Vec<Value>,  // {path, error}
}

/// One model to import, with the index files that sit next to it.
#[derive(Debug)]
pub struct Candidate {
    pub model: PathBuf,
    pub index: Option<PathBuf>,
}

fn ext_is(p: &Path, ext: &str) -> bool {
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, depth + 1, out);
        } else {
            out.push(p);
        }
    }
}

/// Lowercase alphanumerics of the model's base name, without RVC's `_475e_8075s` / `_e900_s122400`
/// training suffixes, so `DonaldTrump_475e_8075s.pth` matches `added_IVF…_DonaldTrump_v2.index`.
fn base_key(stem: &str) -> String {
    let parts: Vec<&str> = stem.split(['_', ' ', '-']).collect();
    let is_suffix = |s: &str| {
        let s = s.to_ascii_lowercase();
        let digits = |t: &str| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit());
        (s.ends_with('e') && digits(&s[..s.len() - 1]))
            || (s.ends_with('s') && digits(&s[..s.len() - 1]))
            || (s.starts_with('e') && digits(&s[1..]))
            || (s.starts_with('s') && digits(&s[1..]))
    };
    let mut keep = parts.len();
    while keep > 1 && is_suffix(parts[keep - 1]) {
        keep -= 1;
    }
    parts[..keep].concat().chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

/// Pick the index for `model` among `indexes` (same folder). RVC writes both `trained_*`
/// (an empty IVF) and `added_*` (the real vectors); only the latter is useful. `sole_model`: the
/// model is the only one in its folder, so a lone index there is its own whatever the names say.
pub fn pair_index(model: &Path, indexes: &[PathBuf], sole_model: bool) -> Option<PathBuf> {
    let usable: Vec<&PathBuf> = indexes
        .iter()
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("trained_")))
        .collect();
    if usable.len() == 1 && sole_model {
        return Some(usable[0].clone());
    }
    let key = base_key(&model.file_stem()?.to_string_lossy());
    if key.is_empty() {
        return None;
    }
    let flat = |p: &Path| -> String {
        p.file_stem().map(|s| s.to_string_lossy().chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()).unwrap_or_default()
    };
    let mut hits: Vec<&PathBuf> = usable.into_iter().filter(|p| flat(p).contains(&key)).collect();
    hits.sort_by_key(|p| !flat(p).starts_with("added"));
    hits.first().map(|p| (*p).clone())
}

/// Expand dropped/picked paths (files, folders, zips extracted to `scratch`) into candidates.
pub fn collect(paths: &[PathBuf], scratch: &Path) -> (Vec<Candidate>, Vec<Value>) {
    let mut files = Vec::new();
    let mut failed = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        if p.is_dir() {
            walk(p, 0, &mut files);
        } else if ext_is(p, "zip") {
            let dest = scratch.join(format!("zip{i}"));
            match extract_zip(p, &dest) {
                Ok(()) => walk(&dest, 0, &mut files),
                Err(e) => failed.push(json!({"path": p, "error": format!("could not open archive: {e}")})),
            }
        } else {
            files.push(p.clone());
            // A dropped .pth also picks up the indexes beside it.
            if ext_is(p, "pth") {
                if let Some(dir) = p.parent() {
                    if let Ok(rd) = fs::read_dir(dir) {
                        files.extend(rd.flatten().map(|e| e.path()).filter(|q| ext_is(q, "index")));
                    }
                }
            }
        }
    }
    let mut seen = HashSet::new();
    files.retain(|f| seen.insert(f.clone()));
    let models: Vec<&PathBuf> = files.iter().filter(|f| ext_is(f, "pth")).collect();
    let mut out = Vec::new();
    for m in &models {
        let dir = m.parent().map(Path::to_path_buf).unwrap_or_default();
        let indexes: Vec<PathBuf> = files.iter().filter(|f| ext_is(f, "index") && f.parent() == Some(dir.as_path())).cloned().collect();
        // A folder of several models with one index: that index belongs to at most one of them.
        let sole = models.iter().filter(|o| o.parent() == m.parent()).count() == 1;
        out.push(Candidate { model: (*m).clone(), index: pair_index(m, &indexes, sole) });
    }
    (out, failed)
}

fn extract_zip(path: &Path, dest: &Path) -> io::Result<()> {
    let mut zip = zip::ZipArchive::new(fs::File::open(path)?).map_err(io::Error::other)?;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(io::Error::other)?;
        let Some(rel) = f.enclosed_name() else { continue }; // skip path-traversal entries
        let lower = rel.to_string_lossy().to_lowercase();
        if f.is_dir() || !(lower.ends_with(".pth") || lower.ends_with(".index")) {
            continue;
        }
        let out = dest.join(rel);
        if let Some(d) = out.parent() {
            fs::create_dir_all(d)?;
        }
        io::copy(&mut f, &mut fs::File::create(&out)?)?;
    }
    Ok(())
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Copy one inspected model (+ index) into the library. `info` is the engine's `model.inspect`.
pub fn install(c: &Candidate, info: &Value) -> io::Result<Option<Value>> {
    let hash = info["hash"].as_str().ok_or_else(|| io::Error::other("inspect returned no hash"))?;
    let dir = voice_dir(hash).ok_or_else(|| io::Error::other("bad hash"))?;
    if dir.join("voice.json").is_file() {
        return Ok(None); // already in the library
    }
    fs::create_dir_all(&dir)?;
    let name_of = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let model_file = name_of(&c.model);
    fs::copy(&c.model, dir.join(&model_file))?;
    let index_file = match &c.index {
        Some(ix) => {
            let n = name_of(ix);
            fs::copy(ix, dir.join(&n))?;
            Some(n)
        }
        None => None,
    };
    let display = info["name"].as_str().map(str::to_string).unwrap_or_else(|| model_file.trim_end_matches(".pth").to_string());
    let v = json!({
        "id": hash,
        "name": display,
        "model_file": model_file,
        "index_file": index_file,
        "source": c.model,
        "info": info,
        "favorite": false,
        "hotkey": null,
        "added_at": now_secs(),
    });
    write_json(&dir.join("voice.json"), &v)?;
    Ok(Some(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn base_key_strips_training_suffixes() {
        assert_eq!(base_key("DonaldTrump_475e_8075s"), "donaldtrump");
        assert_eq!(base_key("CyreneAidenDawnHSR_e900_s122400"), "cyreneaidendawnhsr");
        assert_eq!(base_key("AnniePants (asmr)_860e_182320s"), "anniepantsasmr");
        assert_eq!(base_key("Irene"), "irene");
    }

    #[test]
    fn pairs_added_index_by_name() {
        let m = p("x/DonaldTrump_475e_8075s.pth");
        let ix = vec![
            p("x/trained_IVF1_Flat_nprobe_1_DonaldTrump_v2.index"),
            p("x/added_IVF1_Flat_nprobe_1_DonaldTrump_v2.index"),
            p("x/added_IVF9_Flat_nprobe_1_Other_v2.index"),
        ];
        assert_eq!(pair_index(&m, &ix, true), Some(ix[1].clone()));
        // A lone usable index is taken whatever its name, if the model is alone in its folder.
        assert_eq!(pair_index(&m, &[p("x/model.index")], true), Some(p("x/model.index")));
        // Next to other models it has to match by name: it may well be one of theirs.
        assert_eq!(pair_index(&m, &[p("x/added_IVF9_Other_v2.index")], false), None);
        assert_eq!(pair_index(&m, &[p("x/added_IVF1_DonaldTrump_v2.index")], false), Some(p("x/added_IVF1_DonaldTrump_v2.index")));
        // Several unrelated ones: no guess.
        assert_eq!(pair_index(&m, &[p("x/a.index"), p("x/b.index")], true), None);
        assert_eq!(pair_index(&m, &[p("x/trained_IVF1_DonaldTrump.index")], true), None);
    }
}

/// End-to-end import against the real engine and reference models (needs the engine venv):
/// `cargo test -p resona -- --ignored import_reference_models`
#[cfg(test)]
mod e2e {
    use super::*;

    #[test]
    #[ignore]
    fn import_reference_models() {
        let root = std::env::temp_dir().join(format!("resona-import-{}", std::process::id()));
        std::env::set_var("RESONA_APPDATA", &root);
        let src = PathBuf::from(r"C:\Pedro\AiVoiceChanger\Vonovox-1.6.9\models\Trump");
        if !src.is_dir() {
            return;
        }
        let rt = resona_runtime::Runtime::new(|_| {}, false).unwrap();
        let (cands, failed) = collect(&[src.clone()], &root.join("scratch"));
        assert!(failed.is_empty());
        assert_eq!(cands.len(), 1);
        assert!(cands[0].index.is_some(), "index should pair automatically");
        let info = rt.inspect_model(&cands[0].model.to_string_lossy()).unwrap();
        let v = install(&cands[0], &info).unwrap().expect("first install adds the voice");
        assert!(install(&cands[0], &info).unwrap().is_none(), "second install is a duplicate");
        let listed = list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["id"], v["id"]);
        assert!(listed[0]["index_path"].as_str().is_some_and(|p| PathBuf::from(p).is_file()));
        remove(v["id"].as_str().unwrap()).unwrap();
        assert!(list().is_empty());
        rt.shutdown();
        let _ = fs::remove_dir_all(root);
    }
}
