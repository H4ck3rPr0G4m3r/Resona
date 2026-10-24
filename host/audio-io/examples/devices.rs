//! cargo run -p resona-audio-io --example devices [-- passthrough <seconds> <render-name-substring>]
use resona_audio_io::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

fn main() {
    for dir in [Direction::Capture, Direction::Render] {
        println!("{dir:?}:");
        for d in list_devices(dir).unwrap() {
            let mark = if d.is_default { "*" } else { " " };
            println!("  {mark} {} [{} Hz, {} ch]", d.name, d.mix_rate, d.mix_channels);
        }
    }
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) != Some("passthrough") {
        return;
    }
    let secs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let render_id = args.get(3).and_then(|needle| {
        list_devices(Direction::Render).unwrap().into_iter().find(|d| d.name.contains(needle.as_str())).map(|d| d.id)
    });
    let fifo = Arc::new(Mutex::new(VecDeque::<f32>::new()));
    let (f_in, f_out) = (fifo.clone(), fifo.clone());
    let cap = CaptureStream::start(StreamConfig::default(), move |x| f_in.lock().unwrap().extend(x), |e| eprintln!("capture error: {e}")).unwrap();
    let ren = RenderStream::start(
        StreamConfig { device_id: render_id, ..Default::default() },
        move |out| {
            let mut q = f_out.lock().unwrap();
            for v in out.iter_mut() {
                *v = q.pop_front().unwrap_or(0.0);
            }
        },
        |e| eprintln!("render error: {e}"),
    )
    .unwrap();
    println!("capture: {:?}\nrender:  {:?}", cap.info(), ren.info());
    std::thread::sleep(std::time::Duration::from_secs(secs));
    println!("fifo level at end: {} samples", fifo.lock().unwrap().len());
}
