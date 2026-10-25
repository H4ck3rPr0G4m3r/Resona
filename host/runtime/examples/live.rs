//! Live end-to-end check: mic → engine → output device, printing stats; optionally kills the
//! engine mid-run to exercise crash recovery.
//!
//! cargo run -p resona-runtime --example live -- <model.pth> [index] [seconds] [--kill-at N] [--out "Line 1"]
//!     [--block 60 | --block auto]   (auto = run the engine calibration first)

use std::time::{Duration, Instant};

use resona_runtime::{list_devices, Direction, Event, Runtime, SessionConfig};
use serde_json::json;

fn main() {
    env_logger_lite();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().expect("model path").clone();
    let index = args.get(1).filter(|s| !s.starts_with("--") && !s.is_empty()).cloned();
    let secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let kill_at: Option<u64> = flag("--kill-at").and_then(|s| s.parse().ok());
    let out_name = flag("--out").unwrap_or_else(|| "Line 1".into());
    let out_mode = flag("--out-mode").unwrap_or_else(|| "shared".into());
    let out_id = list_devices(Direction::Render).unwrap().into_iter().find(|d| d.name.contains(&out_name)).map(|d| d.id);

    let rt = Runtime::new(
        |ev| match ev {
            Event::State { state, message } => println!("[state] {state:?} {}", message.unwrap_or_default()),
            Event::Stats(s) => println!(
                "[stats] cap={} ren={} sent={} recv={} p50={:.1} p99={:.1} lat={:.0}ms underruns={} ncc={:.3} dropped={} fifo={:.1}ms vram={}",
                s["captured"], s["rendered"], s["blocks_sent"], s["blocks_received"],
                s["p50_ms"].as_f64().unwrap_or(0.0),
                s["p99_ms"].as_f64().unwrap_or(0.0),
                s["latency"]["total_ms"].as_f64().unwrap_or(0.0),
                s["underruns"],
                s["sola_ncc"].as_f64().unwrap_or(0.0),
                s["dropped"],
                s["fifo_ms"].as_f64().unwrap_or(0.0),
                s["vram_mb"]
            ),
            _ => {}
        },
        false,
    )
    .unwrap();
    let voice = json!({"id": "v1", "path": model, "index_path": index});
    let block: u64 = match flag("--block").as_deref() {
        Some("auto") => {
            let vr = serde_json::from_value(voice.clone()).unwrap();
            let res = rt.calibrate(&vr, json!({}), &[40, 60, 80, 100]).expect("calibrate");
            println!("[calibrate] {res}");
            res["recommended"].as_u64().unwrap_or(60)
        }
        Some(b) => b.parse().expect("--block"),
        None => 60,
    };
    let cfg: SessionConfig = serde_json::from_value(json!({
        "output": {"device_id": out_id, "mode": out_mode},
        "voice": voice,
        "params": {"pitch": 0},
        "stream": {"block_ms": block},
        "jitter_ms": 10
    }))
    .unwrap();
    let t0 = Instant::now();
    rt.start(cfg).expect("start");
    println!("started in {:.1}s", t0.elapsed().as_secs_f32());
    let t1 = Instant::now();
    let mut killed = false;
    while t1.elapsed() < Duration::from_secs(secs) {
        std::thread::sleep(Duration::from_millis(250));
        if let Some(k) = kill_at {
            if !killed && t1.elapsed() >= Duration::from_secs(k) {
                println!(">>> killing engine worker");
                kill_engine();
                killed = true;
            }
        }
    }
    println!("final state: {:?}", rt.state());
    rt.shutdown();
}

fn kill_engine() {
    // Kill only our own worker: python processes whose command line mentions resona_engine.worker.
    let ps = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*resona_engine.worker*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
    let _ = std::process::Command::new("powershell").args(["-NoProfile", "-Command", ps]).output();
}

fn env_logger_lite() {
    struct L;
    impl log::Log for L {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Info
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("{} {}: {}", r.level(), r.target(), r.args());
            }
        }
        fn flush(&self) {}
    }
    let _ = log::set_logger(&L).map(|_| log::set_max_level(log::LevelFilter::Info));
}
