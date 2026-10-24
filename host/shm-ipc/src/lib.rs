//! Shared-memory audio transport between the Resona host and the Python engine.
//!
//! A [`ShmRing`] is a single-producer / single-consumer ring of fixed-size slots living in a
//! named Windows file mapping. Each slot carries one audio block plus a small header. Wakeups
//! use named auto-reset events ([`NamedEvent`]), so neither side polls.
//!
//! Memory layout (little-endian, all offsets in bytes):
//!
//! ```text
//! 0    magic u32 | version u32 | slots u32 | slot_samples u32
//! 64   write_idx u64   (producer-owned, monotonically increasing)
//! 128  read_idx  u64   (consumer-owned, monotonically increasing)
//! 192  slot[0] .. slot[slots-1], each `stride` bytes:
//!        seq u64 | n u32 | flags u32 | f32 × slot_samples
//! ```
//!
//! The Python twin lives in `engine/resona_engine/ipc/shm.py` and must match this layout.

mod event;
mod ring;

pub use event::NamedEvent;
pub use ring::{BlockHeader, ShmRing};

#[derive(Debug, thiserror::Error)]
pub enum ShmError {
    #[error("windows error: {0}")]
    Win(#[from] windows::core::Error),
    #[error("shared memory '{0}' has an incompatible layout")]
    Layout(String),
    #[error("block of {got} samples exceeds slot capacity {cap}")]
    TooLarge { got: usize, cap: usize },
    #[error("ring is full")]
    Full,
}

pub type Result<T> = std::result::Result<T, ShmError>;

pub(crate) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}
