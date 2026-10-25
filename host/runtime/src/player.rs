//! Sound-file input: a decoded clip that replaces the microphone while it plays.
//!
//! Files are decoded (symphonia) and resampled to 48 kHz mono on the control side. The audio
//! side receives clips over a ring and hands replaced clips back over another, so the capture
//! thread never frees memory.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use resona_dsp::Resampler;
use serde::Serialize;

use crate::pipeline::RATE;

pub type Samples = Arc<[f32]>;

const MAX_SECONDS: usize = 60 * 60;

/// Decode any WAV/MP3/FLAC file to 48 kHz mono f32.
pub fn decode(path: &Path) -> Result<Vec<f32>, String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::errors::Error as SErr;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("unsupported or damaged audio file: {e}"))?;
    let mut format = probed.format;
    let track = format.default_track().ok_or("the file has no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("unsupported codec: {e}"))?;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);

    let mut mono: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SErr::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SErr::ResetRequired) => break,
            Err(e) => return Err(format!("read error: {e}")),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SErr::DecodeError(_)) => continue, // skip a corrupt frame
            Err(e) => return Err(format!("decode error: {e}")),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let ch = spec.channels.count().max(1);
        let sb = match &mut buf {
            Some(b) if b.capacity() >= decoded.capacity() * ch => b,
            _ => buf.insert(SampleBuffer::new(decoded.capacity() as u64, spec)),
        };
        sb.copy_interleaved_ref(decoded);
        let s = sb.samples();
        mono.extend(s.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        if rate > 0 && mono.len() > MAX_SECONDS * rate as usize {
            return Err("files longer than an hour are not supported".into());
        }
    }
    if mono.is_empty() || rate == 0 {
        return Err("the file contains no audio".into());
    }
    Ok(resample_to_48k(&mono, rate))
}

fn resample_to_48k(x: &[f32], rate: u32) -> Vec<f32> {
    if rate as usize == RATE {
        return x.to_vec();
    }
    const CHUNK: usize = 8192;
    let mut rs = Resampler::new(rate, RATE as u32, CHUNK, 48);
    let mut out = Vec::with_capacity(x.len() * RATE / rate as usize + 64);
    let mut tmp = vec![0.0; rs.max_out(CHUNK)];
    let delay = (rs.delay_in() * RATE as f32 / rate as f32).round() as usize;
    // Flush the filter with a tail of silence, then drop its group delay from the front.
    let tail = vec![0.0; 64];
    for c in x.chunks(CHUNK).chain(std::iter::once(&tail[..])) {
        let n = rs.process(c, &mut tmp);
        out.extend_from_slice(&tmp[..n]);
    }
    out.drain(..delay.min(out.len()));
    out.truncate(x.len() * RATE / rate as usize);
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct PlayerState {
    pub name: Option<String>,
    pub playing: bool,
    pub looping: bool,
    pub position_s: f32,
    pub duration_s: f32,
}

struct Link {
    tx: rtrb::Producer<Option<Samples>>,
    trash: rtrb::Consumer<Samples>,
}

pub struct PlayerHub {
    clip: Mutex<Option<(String, Samples)>>,
    link: Mutex<Option<Link>>,
    pub playing: AtomicBool,
    pub looping: AtomicBool,
    pub pos: AtomicUsize,
    len: AtomicUsize,
    seek: AtomicI64,
}

impl Default for PlayerHub {
    fn default() -> Self {
        Self {
            clip: Mutex::new(None),
            link: Mutex::new(None),
            playing: AtomicBool::new(false),
            looping: AtomicBool::new(false),
            pos: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            seek: AtomicI64::new(-1),
        }
    }
}

impl PlayerHub {
    fn send(&self, clip: Option<Samples>) {
        if let Some(l) = self.link.lock().as_mut() {
            while l.trash.pop().is_ok() {}
            let _ = l.tx.push(clip);
        }
    }

    /// Load a decoded clip (paused, at the start).
    pub fn load(&self, name: String, samples: Vec<f32>) {
        let s: Samples = samples.into();
        self.playing.store(false, Ordering::Relaxed);
        self.len.store(s.len(), Ordering::Relaxed);
        self.pos.store(0, Ordering::Relaxed);
        self.seek.store(-1, Ordering::Relaxed);
        self.send(Some(s.clone()));
        *self.clip.lock() = Some((name, s));
    }

    pub fn unload(&self) {
        self.playing.store(false, Ordering::Relaxed);
        self.len.store(0, Ordering::Relaxed);
        self.pos.store(0, Ordering::Relaxed);
        self.send(None);
        *self.clip.lock() = None;
    }

    pub fn play(&self, on: bool) {
        if on && self.pos.load(Ordering::Relaxed) >= self.len.load(Ordering::Relaxed) {
            self.seek.store(0, Ordering::Relaxed);
        }
        self.playing.store(on && self.clip.lock().is_some(), Ordering::Relaxed);
    }

    pub fn seek_seconds(&self, s: f32) {
        let n = ((s.max(0.0) * RATE as f32) as usize).min(self.len.load(Ordering::Relaxed));
        self.pos.store(n, Ordering::Relaxed);
        self.seek.store(n as i64, Ordering::Relaxed);
    }

    pub fn state(&self) -> PlayerState {
        let name = self.clip.lock().as_ref().map(|c| c.0.clone());
        PlayerState {
            name,
            playing: self.playing.load(Ordering::Relaxed),
            looping: self.looping.load(Ordering::Relaxed),
            position_s: self.pos.load(Ordering::Relaxed) as f32 / RATE as f32,
            duration_s: self.len.load(Ordering::Relaxed) as f32 / RATE as f32,
        }
    }

    /// Wire a new pipeline's audio side and hand it the current clip.
    pub fn attach(&self) -> PlayerSide {
        let (mut tx, rx) = rtrb::RingBuffer::new(8);
        let (trash_tx, trash) = rtrb::RingBuffer::new(8);
        if let Some((_, s)) = self.clip.lock().as_ref() {
            let _ = tx.push(Some(s.clone()));
        }
        *self.link.lock() = Some(Link { tx, trash });
        PlayerSide { rx, trash: trash_tx, clip: None, pos: self.pos.load(Ordering::Relaxed) }
    }

    pub fn detach(&self) {
        *self.link.lock() = None;
    }
}

/// Capture-thread half.
pub struct PlayerSide {
    rx: rtrb::Consumer<Option<Samples>>,
    trash: rtrb::Producer<Samples>,
    clip: Option<Samples>,
    pos: usize,
}

impl PlayerSide {
    /// Replace `buf` (the microphone) with the clip while it plays. Never allocates or frees.
    pub fn process(&mut self, hub: &PlayerHub, buf: &mut [f32]) {
        while let Ok(msg) = self.rx.pop() {
            if let Some(old) = std::mem::replace(&mut self.clip, msg) {
                if let Err(rtrb::PushError::Full(old)) = self.trash.push(old) {
                    std::mem::forget(old); // never happens in practice; leaking beats freeing here
                }
            }
            self.pos = 0;
        }
        let s = hub.seek.swap(-1, Ordering::Relaxed);
        if s >= 0 {
            self.pos = s as usize;
        }
        let Some(clip) = &self.clip else { return };
        if !hub.playing.load(Ordering::Relaxed) {
            return;
        }
        let mut w = 0;
        while w < buf.len() {
            if self.pos >= clip.len() {
                if hub.looping.load(Ordering::Relaxed) && !clip.is_empty() {
                    self.pos = 0;
                } else {
                    buf[w..].fill(0.0);
                    hub.playing.store(false, Ordering::Relaxed);
                    break;
                }
            }
            let n = (buf.len() - w).min(clip.len() - self.pos);
            buf[w..w + n].copy_from_slice(&clip[self.pos..self.pos + n]);
            self.pos += n;
            w += n;
        }
        hub.pos.store(self.pos.min(clip.len()), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_over_the_mic_then_stops_and_hands_back_old_clips() {
        let hub = PlayerHub::default();
        let mut side = hub.attach();
        hub.load("a".into(), vec![0.5; 1000]);
        let mut buf = vec![1.0f32; 400];
        side.process(&hub, &mut buf);
        assert!(buf.iter().all(|&v| v == 1.0), "paused: mic passes through");
        hub.play(true);
        side.process(&hub, &mut buf);
        assert!(buf.iter().all(|&v| v == 0.5));
        buf.fill(1.0);
        side.process(&hub, &mut buf);
        buf.fill(1.0);
        side.process(&hub, &mut buf);
        assert!(buf[..200].iter().all(|&v| v == 0.5) && buf[200..].iter().all(|&v| v == 0.0), "end: rest zeroed");
        assert!(!hub.playing.load(Ordering::Relaxed));
        hub.load("b".into(), vec![0.25; 10]);
        side.process(&hub, &mut buf);
        // The replaced clip went back to the control side, which drops it on the next send.
        assert_eq!(hub.link.lock().as_ref().unwrap().trash.slots(), 1);
    }

    #[test]
    fn seek_and_loop() {
        let hub = PlayerHub::default();
        let mut side = hub.attach();
        hub.load("a".into(), (0..100).map(|i| i as f32).collect());
        hub.looping.store(true, Ordering::Relaxed);
        hub.seek_seconds(90.0 / RATE as f32);
        hub.play(true);
        let mut buf = vec![0.0; 20];
        side.process(&hub, &mut buf);
        assert_eq!(buf[0], 90.0);
        assert_eq!(buf[10], 0.0);
        assert_eq!(buf[19], 9.0);
    }

    #[test]
    fn resampling_keeps_duration_and_alignment() {
        let rate = 44_100;
        let mut x = vec![0.0; rate as usize];
        x[22_050] = 1.0; // impulse at 0.5 s
        let y = resample_to_48k(&x, rate);
        assert_eq!(y.len(), RATE);
        let peak = y.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!((peak as i64 - 24_000).abs() <= 1, "impulse moved to {peak}");
    }
}
