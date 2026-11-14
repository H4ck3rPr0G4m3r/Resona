//! Filter hot reload, end to end: a `.toml` saved in the user's folder is in the catalog and in
//! the running output chain half a second later, without a restart.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use resona_runtime::{Event, Runtime};
use serde_json::json;

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let t = Instant::now();
    while !ok() {
        assert!(t.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn a_saved_filter_file_is_live_within_a_moment() {
    let dir = std::env::temp_dir().join(format!("resona-hot-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let events = Arc::new(AtomicUsize::new(0));
    let seen = events.clone();
    let rt = Runtime::new(
        move |ev| {
            if matches!(ev, Event::Filters { .. }) {
                seen.fetch_add(1, Ordering::SeqCst);
            }
        },
        true,
    )
    .unwrap();
    rt.watch_filters(Some(dir.clone()));
    let entry = |kind: &str| Runtime::fx_catalog().as_array().unwrap().iter().find(|k| k["kind"] == kind).cloned();
    assert!(entry("filter:bad_mic").is_some(), "the built-in Bad Mic is found next to the app");
    assert!(entry("filter:hot").is_none());

    // Drop a file in: it appears.
    let path = rt.filter_new("Hot").unwrap();
    assert_eq!(path, dir.join("hot.toml"));
    wait_for("the new filter", || entry("filter:hot").is_some());
    assert_eq!(entry("filter:hot").unwrap()["params"][0]["key"], "amount");
    assert!(rt.filter_new("Hot").is_err(), "an existing file is never overwritten");

    // A chain that uses it is accepted, including a value for a macro the file will lose.
    let fx = json!({"output": {"slots": [{"id": 1, "kind": "filter:hot", "params": {"amount": 0.3}}]}});
    rt.set_fx(serde_json::from_value(fx).unwrap()).unwrap();

    // Break the file: the catalog carries the error, the chain still validates (bypass).
    let n = events.load(Ordering::SeqCst);
    std::fs::write(&path, "name = \"Hot\"\n[[chain]]\ntype = \"lowpas\"\n").unwrap();
    wait_for("the error", || entry("filter:hot").is_some_and(|e| e["filter"]["error"].as_str().is_some_and(|m| m.contains("lowpas"))));
    assert!(events.load(Ordering::SeqCst) > n, "the UI is told");
    rt.set_fx(rt.fx_settings()).unwrap();

    // Fix it with different macros: they replace the old ones.
    std::fs::write(&path, "[macros]\ndepth = 0.2\n[[chain]]\ntype = \"lowpass\"\nfreq = \"lerp(8000, 500, depth)\"\n").unwrap();
    wait_for("the fix", || entry("filter:hot").is_some_and(|e| e["filter"]["error"].is_null() && e["params"][0]["key"] == "depth"));
    rt.set_fx(rt.fx_settings()).unwrap();
    rt.set_fx_param("output", 1, "depth", 0.8).unwrap();

    // Editing a built-in filter edits a copy in the user's folder, which then takes its place.
    assert_eq!(entry("filter:bad_mic").unwrap()["filter"]["builtin"], true);
    let copy = rt.filter_file("filter:bad_mic").unwrap();
    assert_eq!(copy, dir.join("bad_mic.toml"));
    wait_for("the copy to take over", || entry("filter:bad_mic").is_some_and(|e| e["filter"]["builtin"] == false));
    assert_eq!(rt.filter_file("filter:bad_mic").unwrap(), copy);

    // Delete a file: gone from the catalog, and the chain that used it still builds.
    std::fs::remove_file(&path).unwrap();
    wait_for("the removal", || entry("filter:hot").is_none());
    rt.set_fx(rt.fx_settings()).unwrap();
    let _ = std::fs::remove_dir_all(dir);
}
