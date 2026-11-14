//! Filter files on disk: where they live, and what is in the folders right now.
//!
//! Built-in filters ship in `filters/` next to the app; the user's own live in
//! `%APPDATA%\Resona\filters`. A user file replaces the built-in filter of the same file name,
//! which is how a built-in one is customised: copy it over and edit the copy.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use resona_dsp::fx::composite::FilterDef;

#[derive(Clone, Debug, Default)]
pub struct FilterDirs {
    pub builtin: Option<PathBuf>,
    pub user: Option<PathBuf>,
}

/// `filters/` with the built-in filters: $RESONA_FILTERS_DIR, then walking up from the
/// executable and the current directory (as the engine is found).
pub fn locate_builtin() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("RESONA_FILTERS_DIR") {
        return Some(PathBuf::from(d));
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().skip(1).map(Path::to_path_buf));
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.extend(cwd.ancestors().map(Path::to_path_buf));
    }
    roots.into_iter().map(|r| r.join("filters")).find(|d| !toml_files(d).is_empty())
}

fn toml_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x.eq_ignore_ascii_case("toml")))
        .collect();
    files.sort();
    files
}

/// What the watcher compares: every filter file with its size and modification time.
pub type Signature = Vec<(PathBuf, u64, Option<SystemTime>)>;

impl FilterDirs {
    fn dirs(&self) -> impl Iterator<Item = (&PathBuf, bool)> {
        self.builtin.iter().map(|d| (d, true)).chain(self.user.iter().map(|d| (d, false)))
    }

    pub fn signature(&self) -> Signature {
        self.dirs()
            .flat_map(|(d, _)| toml_files(d))
            .map(|p| {
                let meta = std::fs::metadata(&p).ok();
                (p, meta.as_ref().map_or(0, |m| m.len()), meta.and_then(|m| m.modified().ok()))
            })
            .collect()
    }

    /// Read every filter file. Sorted by name, built-in ones first among equals.
    pub fn scan(&self) -> Vec<FilterDef> {
        let mut defs: Vec<FilterDef> = Vec::new();
        for (dir, builtin) in self.dirs() {
            for path in toml_files(dir) {
                let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let mut def = match std::fs::read_to_string(&path) {
                    Ok(text) => FilterDef::parse(&stem, &text),
                    Err(e) => {
                        let mut d = FilterDef::parse(&stem, "");
                        d.error = Some(format!("could not read the file: {e}"));
                        d
                    }
                };
                def.file = path.display().to_string();
                def.builtin = builtin;
                // The user's folder is read last, so its file wins.
                match defs.iter_mut().find(|d| d.kind == def.kind) {
                    Some(slot) => *slot = def,
                    None => defs.push(def),
                }
            }
        }
        defs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        defs
    }
}

/// A starting point for a new filter: small, working, and explaining itself.
pub fn template(name: &str) -> String {
    format!(
        r#"# {name}: a Resona filter. Save this file and Resona reloads it.
#
# [macros] are the knobs the editor shows (0 to 1 unless min and max say otherwise).
# [[chain]] stages run top to bottom. A setting is a number, or an expression in quotes:
#   + - * / ( )   lerp(a, b, t)   clamp(x, lo, hi)   min max pow sqrt exp ln abs db
#
# Stage types:
#   highpass lowpass bandpass      freq, q
#   peak lowshelf highshelf        freq, q, gain_db
#   gain                           gain_db
#   saturate                       mode ("soft", "hard", "tube"), drive (1 = clean)
#   bitcrush                       bits
#   downsample                     rate (Hz)
#   noise                          level_db
#   crackle                        rate (events per second), level
#   delay                          time_ms, feedback, mix
#   and every chain effect: compressor, gate, eq, filter, deesser, reverb, chorus, lofi

name = "{name}"

[macros]
amount = {{ default = 0.5 }}

[[chain]]
type = "highpass"
freq = "lerp(40, 500, amount)"

[[chain]]
type = "lowpass"
freq = "lerp(16000, 3000, amount)"

[[chain]]
type = "saturate"
mode = "soft"
drive = "lerp(1, 6, amount)"
"#
    )
}

/// File name for a filter called `name`: lower case, `[a-z0-9_-]`.
pub fn file_stem(name: &str) -> String {
    let stem: String = name.trim().chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c.to_ascii_lowercase() } else { '_' }).collect();
    let stem = stem.trim_matches('_').to_string();
    if stem.is_empty() {
        "filter".into()
    } else {
        stem
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_files_replace_builtins_and_the_template_loads() {
        let root = std::env::temp_dir().join(format!("resona-filters-{}", std::process::id()));
        let dirs = FilterDirs { builtin: Some(root.join("builtin")), user: Some(root.join("user")) };
        for d in [&dirs.builtin, &dirs.user] {
            std::fs::create_dir_all(d.as_ref().unwrap()).unwrap();
        }
        let gain = |db: i32| format!("name = \"Zed\"\n[[chain]]\ntype = \"gain\"\ngain_db = {db}");
        std::fs::write(root.join("builtin/zed.toml"), gain(-3)).unwrap();
        std::fs::write(root.join("builtin/broken.toml"), "name = ").unwrap();
        std::fs::write(root.join("builtin/notes.txt"), "not a filter").unwrap();
        let defs = dirs.scan();
        assert_eq!(defs.iter().map(|d| d.kind.as_str()).collect::<Vec<_>>(), ["filter:broken", "filter:zed"]);
        assert!(defs[0].error.is_some() && defs[0].builtin);
        assert!(defs[1].error.is_none() && defs[1].file.ends_with("zed.toml"));

        let before = dirs.signature();
        std::fs::write(root.join("user/zed.toml"), gain(-9)).unwrap();
        std::fs::write(root.join("user/My Filter.toml"), template("My Filter")).unwrap();
        assert_ne!(dirs.signature(), before, "a new file changes the signature");
        let defs = dirs.scan();
        let zed = defs.iter().find(|d| d.kind == "filter:zed").unwrap();
        assert!(!zed.builtin && zed.file.contains("user"), "the user's file wins");
        let mine = defs.iter().find(|d| d.kind == "filter:my_filter").expect("the template is a valid filter");
        assert_eq!((mine.error.clone(), mine.name.as_str(), mine.macros.len()), (None, "My Filter", 1));
        assert_eq!(file_stem("  My Filter! "), "my_filter");
        assert_eq!(file_stem("***"), "filter");
        let _ = std::fs::remove_dir_all(root);
    }
}
