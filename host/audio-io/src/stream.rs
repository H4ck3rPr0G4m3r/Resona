use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

use serde::{Deserialize, Serialize};
use wasapi::{AudioClient, Handle, SampleType, ShareMode, TimingMode, WaveFormat};

use crate::convert::{from_mono, to_mono, SampleFormat};
use crate::devices::open_device;
use crate::{AudioError, Direction, Result, SAMPLE_RATE};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamMode {
    /// Regular event-driven shared mode (10 ms engine period). The safe default.
    #[default]
    Shared,
    /// IAudioClient3 low-latency shared mode when the driver supports it (falls back to Shared).
    /// Opt-in: some virtual drivers (e.g. VAC Lite) run their clock wrong in this mode.
    LowLatency,
    /// Exclusive mode at the device's minimum period.
    Exclusive,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamConfig {
    pub device_id: Option<String>,
    #[serde(default)]
    pub mode: StreamMode,
    /// Capture only: record what a *render* device (`device_id`) is playing (WASAPI loopback).
    #[serde(default)]
    pub loopback: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamInfo {
    pub device_name: String,
    pub mode: String,
    pub period_frames: u32,
    pub buffer_frames: u32,
    /// Nominal one-way device latency contribution in ms.
    pub latency_ms: f32,
}

struct Opened {
    client: AudioClient,
    event: Handle,
    fmt: SampleFormat,
    channels: usize,
    info: StreamInfo,
}

const HNS_PER_MS: i64 = 10_000;
const OPEN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

fn open(direction: Direction, cfg: &StreamConfig) -> Result<Opened> {
    // Loopback opens the render endpoint and initializes it for capture (shared mode only).
    let loopback = cfg.loopback && direction == Direction::Capture;
    let device = open_device(if loopback { Direction::Render } else { direction }, cfg.device_id.as_deref())?;
    let device_name = device.get_friendlyname().unwrap_or_default();
    let wdir: wasapi::Direction = direction.into();
    let mix = device.get_iaudioclient()?.get_mixformat()?;
    let mix_fmt = SampleFormat::from_wave(&mix);

    if cfg.mode == StreamMode::Exclusive && !loopback {
        let mut client = device.get_iaudioclient()?;
        let channels = mix.get_nchannels() as usize;
        let candidates = [
            (WaveFormat::new(32, 32, &SampleType::Float, SAMPLE_RATE as usize, channels, None), SampleFormat::F32),
            (WaveFormat::new(32, 24, &SampleType::Int, SAMPLE_RATE as usize, channels, None), SampleFormat::I24In32),
            (WaveFormat::new(16, 16, &SampleType::Int, SAMPLE_RATE as usize, channels, None), SampleFormat::I16),
        ];
        let (wf, fmt) = candidates
            .into_iter()
            .find(|(wf, _)| client.is_supported(wf, &ShareMode::Exclusive).is_ok())
            .ok_or_else(|| AudioError::Format("no 48 kHz exclusive format supported".into()))?;
        let (_, min_period) = client.get_device_period()?;
        let period = client.calculate_aligned_period_near(min_period.max(3 * HNS_PER_MS), Some(128), &wf)?;
        client.initialize_client(&wf, &wdir, &wasapi::StreamMode::EventsExclusive { period_hns: period })?;
        let event = client.set_get_eventhandle()?;
        let buffer = client.get_buffer_size()?;
        let info = StreamInfo {
            device_name,
            mode: "exclusive".into(),
            period_frames: buffer,
            buffer_frames: buffer,
            latency_ms: buffer as f32 * 1000.0 / SAMPLE_RATE as f32,
        };
        return Ok(Opened { client, event, fmt, channels, info });
    }

    if cfg.mode == StreamMode::LowLatency && !loopback && mix.get_samplespersec() == SAMPLE_RATE {
        if let Some(fmt) = mix_fmt {
            let attempt = (|| -> Result<Opened> {
                let mut client = device.get_iaudioclient()?;
                let periods = client.get_shared_mode_engine_period(&mix)?;
                if periods.min >= periods.default {
                    return Err(AudioError::Format("driver has no low-latency periods".into()));
                }
                client.initialize_shared_audio_stream(&mix, TimingMode::Events, periods.min)?;
                let event = client.set_get_eventhandle()?;
                let buffer = client.get_buffer_size()?;
                let info = StreamInfo {
                    device_name: device_name.clone(),
                    mode: "low-latency shared".into(),
                    period_frames: periods.min,
                    buffer_frames: buffer,
                    latency_ms: periods.min as f32 * 1000.0 / SAMPLE_RATE as f32,
                };
                Ok(Opened { client, event, fmt, channels: mix.get_nchannels() as usize, info })
            })();
            match attempt {
                Ok(o) => return Ok(o),
                Err(e) => log::info!("{device_name}: low-latency shared mode unavailable ({e}), using shared"),
            }
        }
    }

    // Regular shared mode. Use the mix format directly when it is already 48 kHz (no Windows
    // conversion in the path); otherwise ask the engine to autoconvert to 48 kHz mono f32.
    let mut client = device.get_iaudioclient()?;
    let (default_period, _) = client.get_device_period()?;
    let native = mix.get_samplespersec() == SAMPLE_RATE && mix_fmt.is_some();
    let (wf, fmt, channels) = if native {
        (mix.clone(), mix_fmt.unwrap(), mix.get_nchannels() as usize)
    } else {
        (WaveFormat::new(32, 32, &SampleType::Float, SAMPLE_RATE as usize, 1, None), SampleFormat::F32, 1)
    };
    client.initialize_client(
        &wf,
        &wdir,
        &wasapi::StreamMode::EventsShared { autoconvert: !native, buffer_duration_hns: default_period },
    )?;
    let event = client.set_get_eventhandle()?;
    let buffer = client.get_buffer_size()?;
    let period_frames = (default_period * SAMPLE_RATE as i64 / 10_000_000) as u32;
    let info = StreamInfo {
        device_name,
        mode: match (loopback, native) {
            (true, _) => "loopback".into(),
            (false, true) => "shared".into(),
            (false, false) => "shared (converted)".into(),
        },
        period_frames,
        buffer_frames: buffer,
        latency_ms: period_frames as f32 * 1000.0 / SAMPLE_RATE as f32,
    };
    Ok(Opened { client, event, fmt, channels, info })
}

/// Registers the calling thread with MMCSS for the lifetime of the guard.
struct MmcssGuard(Option<windows::Win32::Foundation::HANDLE>);

impl MmcssGuard {
    fn pro_audio() -> Self {
        use windows::core::w;
        use windows::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
        let mut idx = 0u32;
        // SAFETY: valid task name and out-pointer.
        let h = unsafe { AvSetMmThreadCharacteristicsW(w!("Pro Audio"), &mut idx) }.ok();
        Self(h)
    }
}

impl Drop for MmcssGuard {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            // SAFETY: handle came from AvSetMmThreadCharacteristicsW on this thread.
            unsafe {
                let _ = windows::Win32::System::Threading::AvRevertMmThreadCharacteristics(h);
            }
        }
    }
}

struct Running {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    info: StreamInfo,
}

impl Running {
    fn spawn<B>(name: &str, direction: Direction, cfg: StreamConfig, body: B) -> Result<Self>
    where
        B: FnOnce(Opened, Arc<AtomicBool>) + Send + 'static,
    {
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<Result<StreamInfo>>();
        let stop2 = stop.clone();
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let _ = wasapi::initialize_mta();
                let _mmcss = MmcssGuard::pro_audio();
                match open(direction, &cfg) {
                    Ok(opened) => {
                        let _ = tx.send(Ok(opened.info.clone()));
                        body(opened, stop2);
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                    }
                }
            })
            .map_err(|e| AudioError::Start(e.to_string()))?;
        match rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(info) => Ok(Self { stop, thread: Some(thread), info: info? }),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // A hung Initialize is almost always security software (e.g. antivirus microphone
                // protection) holding the request. Detach the thread; if it ever opens, it sees
                // `stop` and exits immediately.
                stop.store(true, Ordering::Release);
                Err(AudioError::Timeout(direction))
            }
            Err(e) => Err(AudioError::Start(e.to_string())),
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Event-driven capture. `on_data` receives 48 kHz mono f32 packets on the audio thread.
pub struct CaptureStream(Running);

impl CaptureStream {
    pub fn start<F, E>(cfg: StreamConfig, mut on_data: F, on_error: E) -> Result<Self>
    where
        F: FnMut(&[f32]) + Send + 'static,
        E: FnOnce(String) + Send + 'static,
    {
        Self::start_timed(cfg, move |x, _| on_data(x), on_error)
    }

    /// Like [`start`](Self::start), plus the QPC time (100 ns units) of each packet's first frame.
    /// Times from different streams share one clock, so they can be compared directly.
    pub fn start_timed<F, E>(cfg: StreamConfig, mut on_data: F, on_error: E) -> Result<Self>
    where
        F: FnMut(&[f32], u64) + Send + 'static,
        E: FnOnce(String) + Send + 'static,
    {
        let r = Running::spawn("resona-capture", Direction::Capture, cfg, move |o, stop| {
            let res = (|| -> std::result::Result<(), String> {
                let capture = o.client.get_audiocaptureclient().map_err(|e| e.to_string())?;
                let frames_max = (o.info.buffer_frames as usize).max(4096) * 2;
                let mut bytes = vec![0u8; frames_max * o.fmt.bytes() * o.channels];
                let mut mono = vec![0f32; frames_max];
                o.client.start_stream().map_err(|e| e.to_string())?;
                let mut idle = 0;
                while !stop.load(Ordering::Acquire) {
                    if o.event.wait_for_event(250).is_err() {
                        idle += 1;
                        if idle >= 8 {
                            return Err("capture device stopped delivering audio".into());
                        }
                        continue;
                    }
                    idle = 0;
                    loop {
                        match capture.get_next_packet_size().map_err(|e| e.to_string())? {
                            Some(n) if n > 0 => {}
                            _ => break,
                        }
                        let (frames, info) = capture.read_from_device(&mut bytes).map_err(|e| e.to_string())?;
                        let frames = frames as usize;
                        if frames == 0 {
                            break;
                        }
                        let byte_len = frames * o.fmt.bytes() * o.channels;
                        if info.flags.silent {
                            mono[..frames].fill(0.0);
                        } else {
                            to_mono(&bytes[..byte_len], o.fmt, o.channels, &mut mono[..frames]);
                        }
                        on_data(&mono[..frames], info.timestamp);
                    }
                }
                let _ = o.client.stop_stream();
                Ok(())
            })();
            if let Err(e) = res {
                on_error(e);
            }
        })?;
        Ok(Self(r))
    }

    pub fn info(&self) -> &StreamInfo {
        &self.0.info
    }

    pub fn stop(mut self) {
        self.0.stop();
    }
}

impl Drop for CaptureStream {
    fn drop(&mut self) {
        self.0.stop();
    }
}

/// Event-driven render. `fill` must write 48 kHz mono f32 into the slice on the audio thread.
pub struct RenderStream(Running);

impl RenderStream {
    pub fn start<F, E>(cfg: StreamConfig, mut fill: F, on_error: E) -> Result<Self>
    where
        F: FnMut(&mut [f32]) + Send + 'static,
        E: FnOnce(String) + Send + 'static,
    {
        let r = Running::spawn("resona-render", Direction::Render, cfg, move |o, stop| {
            let res = (|| -> std::result::Result<(), String> {
                let render = o.client.get_audiorenderclient().map_err(|e| e.to_string())?;
                let frames_max = o.info.buffer_frames as usize;
                let frame_bytes = o.fmt.bytes() * o.channels;
                let mut bytes = vec![0u8; frames_max * frame_bytes];
                let mut mono = vec![0f32; frames_max];
                // Prime with silence so the stream starts without an underrun.
                let avail = o.client.get_available_space_in_frames().map_err(|e| e.to_string())? as usize;
                render
                    .write_to_device(avail, &bytes[..avail * frame_bytes], None)
                    .map_err(|e| e.to_string())?;
                o.client.start_stream().map_err(|e| e.to_string())?;
                let mut idle = 0;
                while !stop.load(Ordering::Acquire) {
                    if o.event.wait_for_event(250).is_err() {
                        idle += 1;
                        if idle >= 8 {
                            return Err("render device stopped requesting audio".into());
                        }
                        continue;
                    }
                    idle = 0;
                    let n = (o.client.get_available_space_in_frames().map_err(|e| e.to_string())? as usize).min(frames_max);
                    if n == 0 {
                        continue;
                    }
                    fill(&mut mono[..n]);
                    from_mono(&mono[..n], o.fmt, o.channels, &mut bytes[..n * frame_bytes]);
                    render.write_to_device(n, &bytes[..n * frame_bytes], None).map_err(|e| e.to_string())?;
                }
                let _ = o.client.stop_stream();
                Ok(())
            })();
            if let Err(e) = res {
                on_error(e);
            }
        })?;
        Ok(Self(r))
    }

    pub fn info(&self) -> &StreamInfo {
        &self.0.info
    }

    pub fn stop(mut self) {
        self.0.stop();
    }
}

impl Drop for RenderStream {
    fn drop(&mut self) {
        self.0.stop();
    }
}
