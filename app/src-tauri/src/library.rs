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
    mark_shared_indexes(&mut out);
    out
}

/// `index_shared_with`: the other voices whose index is the same file (same name and size).
/// An older import paired a folder's only index with every model picked from that folder, which
/// leaves exactly this behind; the index belongs to at most one of them. `index_owner`: this
/// voice's name is in the index's file name, so it is the one the index was made for.
fn mark_shared_indexes(voices: &mut [Value]) {
    let key = |v: &Value| -> Option<(String, u64)> {
        let path = PathBuf::from(v["index_path"].as_str()?);
        Some((path.file_name()?.to_string_lossy().to_lowercase(), fs::metadata(&path).ok()?.len()))
    };
    let keys: Vec<Option<(String, u64)>> = voices.iter().map(key).collect();
    let names: Vec<Value> = voices.iter().map(|v| v["name"].clone()).collect();
    for (i, v) in voices.iter_mut().enumerate() {
        let others: Vec<Value> = match &keys[i] {
            Some(k) => (0..keys.len()).filter(|&j| j != i && keys[j].as_ref() == Some(k)).map(|j| names[j].clone()).collect(),
            None => Vec::new(),
        };
        let stem = v["model_file"].as_str().map(|f| Path::new(f).file_stem().unwrap_or_default().to_string_lossy().into_owned());
        let model_key = stem.map(|s| base_key(&s)).unwrap_or_default();
        let owner = !model_key.is_empty()
            && keys[i].as_ref().is_some_and(|(name, _)| name.chars().filter(|c| c.is_alphanumeric()).collect::<String>().contains(&model_key));
        v["index_shared_with"] = json!(others);
        v["index_owner"] = json!(owner);
    }
}

/// Give a voice a copy of another index, or none. The old copy is deleted.
pub fn set_index(id: &str, index: Option<&Path>) -> io::Result<Value> {
    let dir = voice_dir(id).ok_or_else(|| io::Error::other("bad voice id"))?;
    let path = dir.join("voice.json");
    let mut v = read_json(&path).ok_or_else(|| io::Error::other("voice not found"))?;
    let old = v.get("index_file").and_then(Value::as_str).map(str::to_string);
    let new = match index {
        Some(src) => {
            if !ext_is(src, "index") {
                return Err(io::Error::other("pick an .index file"));
            }
            let name = src.file_name().ok_or_else(|| io::Error::other("bad index path"))?.to_string_lossy().into_owned();
            let dest = dir.join(&name);
            if fs::canonicalize(src).ok() != fs::canonicalize(&dest).ok() {
                fs::copy(src, &dest)?;
            }
            Some(name)
        }
        None => None,
    };
    v["index_file"] = json!(new);
    write_json(&path, &v)?;
    if let Some(o) = old.filter(|o| Some(o) != new.as_ref()) {
        let _ = fs::remove_file(dir.join(o));
    }
    Ok(v)
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
/// (an empty IVF) and `added_*` (the real vectors); only the latter is useful. `trust_lone`: a
/// single usable index is this model's whatever the names say (it has no other model to belong to).
pub fn pair_index(model: &Path, indexes: &[PathBuf], trust_lone: bool) -> Option<PathBuf> {
    let usable: Vec<&PathBuf> = indexes
        .iter()
        .filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().starts_with("trained_")))
        .collect();
    if usable.len() == 1 && trust_lone {
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

/// Number of models in `dir` on disk (not only the ones being imported).
fn models_in(dir: &Path) -> usize {
    fs::read_dir(dir).map(|rd| rd.flatten().filter(|e| ext_is(&e.path(), "pth")).count()).unwrap_or(0)
}

/// Expand dropped/picked paths (files, folders, zips extracted to `scratch`) into candidates.
pub fn collect(paths: &[PathBuf], scratch: &Path) -> (Vec<Candidate>, Vec<Value>) {
    let mut files = Vec::new();
    let mut picked = HashSet::new(); // files the user chose themselves, not found inside a folder
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
            picked.insert(p.clone());
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
        let picked_ix: Vec<PathBuf> = indexes.iter().filter(|f| picked.contains(*f)).cloned().collect();
        // Indexes picked together with this model are meant for it: choose among those only, and
        // take a lone one if it came with a lone model. Otherwise a lone index found in the folder
        // is trusted only if the folder holds no other model *on disk*: a .pth picked out of a
        // Downloads folder must not take another voice's index that happens to sit next to it.
        let (cands, trust_lone) = if picked.contains(*m) && !picked_ix.is_empty() {
            let picked_models = models.iter().filter(|o| o.parent() == m.parent() && picked.contains(**o)).count();
            (picked_ix, picked_models == 1)
        } else {
            (indexes, models_in(&dir) == 1)
        };
        out.push(Candidate { model: (*m).clone(), index: pair_index(m, &cands, trust_lone) });
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

    #[test]
    fn picked_model_never_takes_a_neighbours_index() {
        let root = std::env::temp_dir().join(format!("resona-pair-{}", std::process::id()));
        let downloads = root.join("Downloads");
        let own = root.join("Mia");
        let shared = root.join("Voices");
        fs::create_dir_all(&downloads).unwrap();
        fs::create_dir_all(&own).unwrap();
        fs::create_dir_all(&shared).unwrap();
        for f in ["AnniePants (asmr)_860e.pth", "added_IVF512_Flat_nprobe_1_AnniePants (asmr)_v2.index", "Lena.pth"] {
            fs::write(shared.join(f), b"").unwrap();
        }
        for f in ["AnniePants (asmr)_860e.pth", "added_IVF512_Flat_nprobe_1_AnniePants (asmr)_v2.index", "Mia_300e.pth", "Lena.pth", "added_IVF256_Flat_v2.index"] {
            fs::write(downloads.join(f), b"").unwrap();
        }
        fs::write(own.join("Mia_300e.pth"), b"").unwrap();
        fs::write(own.join("added_IVF256_Flat_v2.index"), b"").unwrap();
        let scratch = root.join("scratch");
        let index_of = |paths: &[PathBuf], model: &str| -> Option<String> {
            let (c, _) = collect(paths, &scratch);
            let c = c.into_iter().find(|c| c.model.file_name().unwrap() == model).expect("candidate");
            c.index.map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        };
        let anniepants = Some("added_IVF512_Flat_nprobe_1_AnniePants (asmr)_v2.index".to_string());
        let generic = Some("added_IVF256_Flat_v2.index".to_string());

        // One index in a folder of several voices: picking any other voice must not take it.
        assert_eq!(index_of(&[shared.join("Lena.pth")], "Lena.pth"), None);
        assert_eq!(index_of(&[shared.clone()], "Lena.pth"), None);
        assert_eq!(index_of(&[shared.join("AnniePants (asmr)_860e.pth")], "AnniePants (asmr)_860e.pth"), anniepants);
        // Picked on their own out of a shared folder: only a name match pairs.
        assert_eq!(index_of(&[downloads.join("AnniePants (asmr)_860e.pth")], "AnniePants (asmr)_860e.pth"), anniepants);
        assert_eq!(index_of(&[downloads.join("Lena.pth")], "Lena.pth"), None);
        // Picked together with the index the user chose for it: paired whatever the names.
        assert_eq!(index_of(&[downloads.join("Lena.pth"), downloads.join("added_IVF256_Flat_v2.index")], "Lena.pth"), generic);
        // The whole folder: names decide, nobody gets AnniePants' index by elimination.
        let all = [downloads.clone()];
        assert_eq!(index_of(&all, "AnniePants (asmr)_860e.pth"), anniepants);
        assert_eq!(index_of(&all, "Mia_300e.pth"), None);
        // A model alone in its folder keeps its lone, generically named index.
        assert_eq!(index_of(&[own.join("Mia_300e.pth")], "Mia_300e.pth"), generic);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn shared_index_files_are_flagged() {
        let root = std::env::temp_dir().join(format!("resona-shared-{}", std::process::id()));
        let at = |voice: &str, file: &str, bytes: usize| -> Value {
            let p = root.join(voice).join(file);
            fs::create_dir_all(p.parent().unwrap()).unwrap();
            fs::write(&p, vec![0u8; bytes]).unwrap();
            json!({"name": voice, "model_file": format!("{voice}_300e.pth"), "index_path": p.to_string_lossy()})
        };
        let mut v = vec![
            at("AnniePants", "added_IVF512_Flat_AnniePants_v2.index", 64),
            at("Lena", "added_IVF512_Flat_AnniePants_v2.index", 64), // the wrongly paired copy
            at("Mia", "added_IVF256_Flat_v2.index", 64),             // same size, other file
            at("Zoe", "added_IVF256_Flat_v2.index", 32),             // same name, other content
            json!({"name": "Noindex", "model_file": "Noindex.pth", "index_path": null}),
        ];
        mark_shared_indexes(&mut v);
        assert_eq!(v[0]["index_shared_with"], json!(["Lena"]));
        assert_eq!(v[1]["index_shared_with"], json!(["AnniePants"]));
        assert_eq!((v[0]["index_owner"].clone(), v[1]["index_owner"].clone()), (json!(true), json!(false)));
        for i in 2..5 {
            assert_eq!(v[i]["index_shared_with"], json!([]), "{}", v[i]["name"]);
            assert_eq!(v[i]["index_owner"], json!(false), "{}", v[i]["name"]);
        }
        let _ = fs::remove_dir_all(root);
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
