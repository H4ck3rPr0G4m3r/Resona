//! WASAPI audio I/O for Resona: device enumeration and event-driven capture/render streams.
//!
//! Every stream presents **48 kHz mono f32** to its callback regardless of the device format:
//! - low-latency shared mode (IAudioClient3, ~3 ms periods) when the device mix rate is 48 kHz,
//! - otherwise regular event-driven shared mode with Windows' format autoconversion,
//! - or exclusive mode when requested and supported.
//!
//! Stream threads register with MMCSS ("Pro Audio") and never allocate after start.

mod convert;
mod devices;
mod stream;

pub use devices::{list_devices, DeviceInfo};
pub use stream::{CaptureStream, RenderStream, StreamConfig, StreamInfo, StreamMode};

pub const SAMPLE_RATE: u32 = 48_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Direction {
    Capture,
    Render,
}

impl From<Direction> for wasapi::Direction {
    fn from(d: Direction) -> Self {
        match d {
            Direction::Capture => wasapi::Direction::Capture,
            Direction::Render => wasapi::Direction::Render,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("audio device error: {0}")]
    Wasapi(#[from] wasapi::WasapiError),
    #[error("device not found: {0}")]
    NotFound(String),
    #[error("unsupported device format: {0}")]
    Format(String),
    #[error("stream thread failed to start: {0}")]
    Start(String),
}

pub type Result<T> = std::result::Result<T, AudioError>;
