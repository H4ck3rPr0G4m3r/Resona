//! Pipeline soak test: real stub engine process + shared memory, simulated device clocks.
//!
//! `cargo test -p resona-runtime --test soak` (20 s). Set RESONA_SOAK_SECS=1800 for the 30-min soak.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use resona_runtime::pipeline::{build, Layout, Shared, Transport};
use resona_runtime::worker::{locate_engine, WorkerProcess};
use serde_json::json;

fn spin_until(t: Instant) {
    while Instant::now() < t {
        if t - Instant::now() > Duration::from_millis(2) {
            std::thread::sleep(Duration::from_millis(1));
        } else {
            std::hint::spin_loop();
        }
    }
}

#[test]
fn stub_engine_soak_has_no_underruns() {
    let Some(loc) = locate_engine() else {
        eprintln!("engine venv not found; skipping");
        return;
    };
    let secs: u64 = std::env::var("RESONA_SOAK_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let prefix = format!("Local\\resona-soak-{}", std::process::id());
    let transport = Arc::new(Transport::create(&prefix).unwrap());
    let worker = WorkerProcess::spawn(&loc, &prefix, true, |_, _| {}, || {}).unwrap();
    worker.rpc.call("hello", json!({}), Duration::from_secs(30)).unwrap();
    let layout: Layout = serde_json::from_value(
        worker.rpc.call("stream.configure", json!({"config": {"block_ms": 60}}), Duration::from_secs(30)).unwrap(),
    )
    .unwrap();
    worker.rpc.call("stream.start", json!({}), Duration::from_secs(10)).unwrap();

    let shared = Arc::new(Shared::default());
    shared.engine_live.store(true, Ordering::Release);
    let parts = build(transport, shared.clone(), layout, 15.0, false);
    let (mut cap, mut ren) = (parts.capture, parts.render);

    // 10 ms device periods; capture and render share one clock here.
    let period = 480;
    let mut phase = 0f32;
    let mut inp = vec![0f32; period];
    let mut out = vec![0f32; period];
    let mut rendered: Vec<f32> = Vec::new();
    let start = Instant::now();
    let mut next = start;
    let mut underruns_after_warmup = 0;
    let warmup = Duration::from_secs(1);
    while start.elapsed() < Duration::from_secs(secs) {
        next += Duration::from_millis(10);
        spin_until(next);
        for v in inp.iter_mut() {
            *v = 0.3 * phase.sin();
            phase += 2.0 * std::f32::consts::PI * 440.0 / 48000.0;
        }
        cap.process(&inp);
        let before = shared.underruns.load(Ordering::Relaxed);
        ren.process(&mut out);
        if start.elapsed() > warmup && shared.underruns.load(Ordering::Relaxed) > before {
            underruns_after_warmup += 1;
        }
        if start.elapsed() > Duration::from_secs(2) && rendered.len() < 48000 {
            rendered.extend_from_slice(&out);
        }
    }
    shared.stop.store(true, Ordering::Release);
    parts.dsp.join().unwrap();
    worker.kill();

    assert_eq!(underruns_after_warmup, 0, "underruns during soak");
    let rms = (rendered.iter().map(|v| v * v).sum::<f32>() / rendered.len() as f32).sqrt();
    assert!((rms - 0.3 / 2f32.sqrt()).abs() < 0.03, "tone should pass through the echo engine, rms={rms}");
    // No clicks: the largest sample-to-sample step stays near that of a clean 440 Hz sine.
    let max_step = rendered.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
    let clean_step = 0.3 * 2.0 * std::f32::consts::PI * 440.0 / 48000.0;
    assert!(max_step < clean_step * 1.5, "discontinuity in output: {max_step} vs {clean_step}");
}
