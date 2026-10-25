//! End-to-end: sound-file input + A/B recorder through the real pipeline and the stub (echo)
//! engine. The dry take must be the file, and the wet take must line up with it sample-exactly
//! even though the output chain (limiter lookahead, etc.) delays the signal.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use resona_runtime::pipeline::{build, Layout, Shared, Transport};
use resona_runtime::worker::{locate_engine, WorkerProcess};
use serde_json::json;

fn read(p: &str) -> Vec<f32> {
    hound::WavReader::open(p).unwrap().samples::<f32>().map(|s| s.unwrap()).collect()
}

/// Normalized cross-correlation of `b` against `a` at `lag` (b[i + lag] vs a[i]).
fn ncc(a: &[f32], b: &[f32], lag: i64) -> f32 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() as i64 {
        let j = i + lag;
        if j < 0 || j >= b.len() as i64 {
            continue;
        }
        let (x, y) = (a[i as usize] as f64, b[j as usize] as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    (ab / (aa.sqrt() * bb.sqrt() + 1e-12)) as f32
}

#[test]
fn file_input_and_ab_takes_are_aligned() {
    let Some(loc) = locate_engine() else {
        eprintln!("engine venv not found; skipping");
        return;
    };
    let prefix = format!("Local\\resona-ab-{}", std::process::id());
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

    // The "file": 6 s of amplitude-modulated noise (aperiodic, so correlation has one peak),
    // low-passed well under the stub engine's 8 kHz Nyquist so the round trip keeps it intact.
    let mut seed = 7u32;
    let mut lp = [0f32; 4];
    let a = 1.0 - (-2.0 * std::f32::consts::PI * 1500.0 / 48_000.0).exp();
    let clip: Vec<f32> = (0..6 * 48_000)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let mut v = (seed >> 8) as f32 / (1 << 24) as f32 - 0.5;
            for s in lp.iter_mut() {
                *s += a * (v - *s);
                v = *s;
            }
            let env = 0.5 + 0.5 * (i as f32 * 2.0 * std::f32::consts::PI * 3.0 / 48_000.0).sin();
            3.0 * v * env
        })
        .collect();
    shared.player.load("noise".into(), clip.clone());

    let dir = std::env::temp_dir().join(format!("resona-ab-{}", std::process::id()));
    let mic = vec![0.05f32; 480]; // must never reach the take while the file plays
    let mut out = vec![0f32; 480];
    let start = Instant::now();
    let mut next = start;
    let mut playing = false;
    let mut recording = false;
    let mut take = None;
    while start.elapsed() < Duration::from_secs(6) {
        next += Duration::from_millis(10);
        while Instant::now() < next {
            std::hint::spin_loop();
        }
        let t = start.elapsed();
        if t > Duration::from_millis(500) && !playing {
            playing = true;
            shared.player.play(true);
        }
        if t > Duration::from_millis(1500) && !recording {
            recording = true;
            shared.recorder.start(dir.clone(), "ab".into()).unwrap();
        }
        if t > Duration::from_millis(4500) && take.is_none() {
            take = Some(shared.recorder.stop().unwrap());
        }
        cap.process(&mic);
        ren.process(&mut out);
    }
    shared.stop.store(true, Ordering::Release);
    parts.dsp.join().unwrap();
    worker.kill();

    let take = take.unwrap();
    let dry = read(&take.dry);
    let wet = read(&take.wet);
    let _ = std::fs::remove_dir_all(&dir);
    assert!((2.8..3.2).contains(&take.seconds), "take length {}", take.seconds);
    assert_eq!(dry.len(), wet.len());

    // Dry is the file itself: find where in the clip the take started.
    let probe = &dry[..4800];
    let off = (0..clip.len() - probe.len()).find(|&o| clip[o..o + 32] == probe[..32]).expect("dry take is not the file");
    assert_eq!(&clip[off..off + dry.len()], &dry[..], "dry take differs from the file");

    // Wet lines up with dry: the correlation peak sits at lag 0 (± 2 samples).
    let a = &dry[24_000..120_000];
    let w = &wet[24_000..120_000];
    let (best, lag) = (-480..=480).map(|l| (ncc(a, w, l), l)).fold((f32::MIN, 0), |m, x| if x.0 > m.0 { x } else { m });
    assert!(best > 0.9, "wet does not resemble dry (ncc {best})");
    assert!(lag.abs() <= 2, "wet is misaligned by {lag} samples (ncc {best})");
}
