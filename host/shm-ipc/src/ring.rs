use std::sync::atomic::{AtomicU64, Ordering};

use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
    MEMORY_MAPPED_VIEW_ADDRESS, PAGE_READWRITE,
};

use crate::{wide, Result, ShmError};

pub const MAGIC: u32 = 0x524E_5341; // "ASNR" little-endian → reads "RSNA"
pub const VERSION: u32 = 1;
const OFF_WRITE: usize = 64;
const OFF_READ: usize = 128;
const OFF_SLOTS: usize = 192;
const SLOT_HEADER: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockHeader {
    pub seq: u64,
    pub n: usize,
    pub flags: u32,
}

/// SPSC ring of audio blocks in a named file mapping.
///
/// Exactly one thread (in one process) may call [`push`](Self::push) and exactly one may call
/// [`pop`](Self::pop)/[`clear`](Self::clear). `push` and `pop` never allocate or lock.
pub struct ShmRing {
    handle: HANDLE,
    base: *mut u8,
    slots: usize,
    slot_samples: usize,
    stride: usize,
    name: String,
}

// SAFETY: the mapping is process-wide memory; the SPSC contract above is what makes concurrent
// access sound, and all cross-thread index traffic goes through atomics.
unsafe impl Send for ShmRing {}
unsafe impl Sync for ShmRing {}

pub fn stride_for(slot_samples: usize) -> usize {
    (SLOT_HEADER + 4 * slot_samples).div_ceil(64) * 64
}

pub fn size_for(slots: usize, slot_samples: usize) -> usize {
    OFF_SLOTS + slots * stride_for(slot_samples)
}

impl ShmRing {
    /// Create (or re-initialize) a named ring. `name` should be unique per session.
    pub fn create(name: &str, slots: usize, slot_samples: usize) -> Result<Self> {
        let size = size_for(slots, slot_samples);
        let w = wide(name);
        // SAFETY: plain Win32 calls with valid arguments; the view is checked for null.
        unsafe {
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                None,
                PAGE_READWRITE,
                (size as u64 >> 32) as u32,
                size as u32,
                PCWSTR(w.as_ptr()),
            )?;
            let mut ring = Self::map(handle, name, size)?;
            ring.slots = slots;
            ring.slot_samples = slot_samples;
            ring.stride = stride_for(slot_samples);
            std::ptr::write_bytes(ring.base, 0, size);
            let hdr = ring.base as *mut u32;
            hdr.add(1).write(VERSION);
            hdr.add(2).write(slots as u32);
            hdr.add(3).write(slot_samples as u32);
            std::sync::atomic::fence(Ordering::Release);
            hdr.write(MAGIC);
            Ok(ring)
        }
    }

    /// Open an existing ring created by another process or thread.
    pub fn open(name: &str) -> Result<Self> {
        let w = wide(name);
        // SAFETY: as above; layout is validated before use.
        unsafe {
            let handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS.0, false, PCWSTR(w.as_ptr()))?;
            let probe = Self::map(handle, name, OFF_SLOTS)?;
            let hdr = probe.base as *const u32;
            let (magic, version) = (hdr.read_volatile(), hdr.add(1).read());
            let (slots, slot_samples) = (hdr.add(2).read() as usize, hdr.add(3).read() as usize);
            if magic != MAGIC || version != VERSION || slots == 0 {
                return Err(ShmError::Layout(name.into()));
            }
            // Re-map with the full size (probe's Drop unmaps and closes its own duplicate).
            let handle2 = OpenFileMappingW(FILE_MAP_ALL_ACCESS.0, false, PCWSTR(w.as_ptr()))?;
            drop(probe);
            let mut full = Self::map(handle2, name, size_for(slots, slot_samples))?;
            full.slots = slots;
            full.slot_samples = slot_samples;
            full.stride = stride_for(slot_samples);
            Ok(full)
        }
    }

    unsafe fn map(handle: HANDLE, name: &str, size: usize) -> Result<Self> {
        let view = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if view.Value.is_null() {
            let err = windows::core::Error::from_thread();
            let _ = CloseHandle(handle);
            return Err(err.into());
        }
        Ok(Self { handle, base: view.Value as *mut u8, slots: 0, slot_samples: 0, stride: 0, name: name.into() })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn slot_samples(&self) -> usize {
        self.slot_samples
    }

    pub fn capacity(&self) -> usize {
        self.slots
    }

    fn idx(&self, off: usize) -> &AtomicU64 {
        // SAFETY: offsets are 64-byte aligned inside the mapping, which outlives `self`.
        unsafe { &*(self.base.add(off) as *const AtomicU64) }
    }

    /// Number of blocks waiting to be popped.
    pub fn len(&self) -> usize {
        let w = self.idx(OFF_WRITE).load(Ordering::Acquire);
        let r = self.idx(OFF_READ).load(Ordering::Acquire);
        w.saturating_sub(r) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Producer: append one block. Fails with [`ShmError::Full`] instead of overwriting.
    pub fn push(&self, seq: u64, flags: u32, data: &[f32]) -> Result<()> {
        if data.len() > self.slot_samples {
            return Err(ShmError::TooLarge { got: data.len(), cap: self.slot_samples });
        }
        let w = self.idx(OFF_WRITE).load(Ordering::Relaxed);
        let r = self.idx(OFF_READ).load(Ordering::Acquire);
        if w - r >= self.slots as u64 {
            return Err(ShmError::Full);
        }
        // SAFETY: slot index is in range; the consumer does not touch this slot until write_idx
        // is published below.
        unsafe {
            let slot = self.base.add(OFF_SLOTS + (w as usize % self.slots) * self.stride);
            (slot as *mut u64).write(seq);
            (slot.add(8) as *mut u32).write(data.len() as u32);
            (slot.add(12) as *mut u32).write(flags);
            std::ptr::copy_nonoverlapping(data.as_ptr(), slot.add(SLOT_HEADER) as *mut f32, data.len());
        }
        self.idx(OFF_WRITE).store(w + 1, Ordering::Release);
        Ok(())
    }

    /// Consumer: pop the oldest block into `out` (must hold `slot_samples`).
    pub fn pop(&self, out: &mut [f32]) -> Option<BlockHeader> {
        let r = self.idx(OFF_READ).load(Ordering::Relaxed);
        let w = self.idx(OFF_WRITE).load(Ordering::Acquire);
        if r >= w {
            return None;
        }
        // SAFETY: the producer published this slot via the Release store of write_idx.
        let hdr = unsafe {
            let slot = self.base.add(OFF_SLOTS + (r as usize % self.slots) * self.stride);
            let seq = (slot as *const u64).read();
            let n = ((slot.add(8) as *const u32).read() as usize).min(self.slot_samples);
            let flags = (slot.add(12) as *const u32).read();
            let n_copy = n.min(out.len());
            std::ptr::copy_nonoverlapping(slot.add(SLOT_HEADER) as *const f32, out.as_mut_ptr(), n_copy);
            BlockHeader { seq, n: n_copy, flags }
        };
        self.idx(OFF_READ).store(r + 1, Ordering::Release);
        Some(hdr)
    }

    /// Consumer: drop everything currently queued.
    pub fn clear(&self) {
        let w = self.idx(OFF_WRITE).load(Ordering::Acquire);
        self.idx(OFF_READ).store(w, Ordering::Release);
    }
}

impl Drop for ShmRing {
    fn drop(&mut self) {
        // SAFETY: base/handle came from MapViewOfFile/CreateFileMappingW and are released once.
        unsafe {
            let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: self.base as _ });
            let _ = CloseHandle(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique(tag: &str) -> String {
        format!("Local\\resona-test-{tag}-{}", std::process::id())
    }

    #[test]
    fn push_pop_roundtrip() {
        let ring = ShmRing::create(&unique("rt"), 4, 8).unwrap();
        let mut out = [0f32; 8];
        assert!(ring.pop(&mut out).is_none());
        ring.push(7, 1, &[1.0, 2.0, 3.0]).unwrap();
        let h = ring.pop(&mut out).unwrap();
        assert_eq!(h, BlockHeader { seq: 7, n: 3, flags: 1 });
        assert_eq!(&out[..3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn full_and_too_large() {
        let ring = ShmRing::create(&unique("full"), 2, 4).unwrap();
        ring.push(0, 0, &[0.0]).unwrap();
        ring.push(1, 0, &[0.0]).unwrap();
        assert!(matches!(ring.push(2, 0, &[0.0]), Err(ShmError::Full)));
        assert!(matches!(ring.push(2, 0, &[0.0; 5]), Err(ShmError::TooLarge { .. })));
        ring.clear();
        assert!(ring.is_empty());
    }

    #[test]
    fn open_sees_same_memory() {
        let name = unique("open");
        let a = ShmRing::create(&name, 8, 16).unwrap();
        let b = ShmRing::open(&name).unwrap();
        assert_eq!(b.slot_samples(), 16);
        a.push(42, 0, &[0.5; 16]).unwrap();
        let mut out = [0f32; 16];
        assert_eq!(b.pop(&mut out).unwrap().seq, 42);
        assert!(a.is_empty());
    }

    #[test]
    fn spsc_stress_preserves_order_and_content() {
        let name = unique("stress");
        let prod = ShmRing::create(&name, 16, 64).unwrap();
        let cons = ShmRing::open(&name).unwrap();
        const N: u64 = 200_000;
        let t = std::thread::spawn(move || {
            let mut seq = 0u64;
            let mut buf = [0f32; 64];
            while seq < N {
                let n = 1 + (seq as usize % 64);
                for (i, v) in buf[..n].iter_mut().enumerate() {
                    *v = (seq as f32) + i as f32;
                }
                if prod.push(seq, 0, &buf[..n]).is_ok() {
                    seq += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });
        let mut expect = 0u64;
        let mut out = [0f32; 64];
        while expect < N {
            if let Some(h) = cons.pop(&mut out) {
                assert_eq!(h.seq, expect);
                assert_eq!(h.n, 1 + (expect as usize % 64));
                assert_eq!(out[h.n - 1], expect as f32 + (h.n - 1) as f32);
                expect += 1;
            }
        }
        t.join().unwrap();
    }
}
