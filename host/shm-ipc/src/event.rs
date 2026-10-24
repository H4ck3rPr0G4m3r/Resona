use windows::core::PCWSTR;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::System::Threading::{
    CreateEventW, OpenEventW, SetEvent, WaitForSingleObject, EVENT_MODIFY_STATE, SYNCHRONIZATION_SYNCHRONIZE,
};

use crate::{wide, Result};

/// Named auto-reset event used to wake the other side of a ring.
pub struct NamedEvent {
    handle: HANDLE,
}

// SAFETY: event handles may be used from any thread.
unsafe impl Send for NamedEvent {}
unsafe impl Sync for NamedEvent {}

impl NamedEvent {
    pub fn create(name: &str) -> Result<Self> {
        let w = wide(name);
        // SAFETY: valid null-terminated name; auto-reset, initially non-signaled.
        let handle = unsafe { CreateEventW(None, false, false, PCWSTR(w.as_ptr()))? };
        Ok(Self { handle })
    }

    pub fn open(name: &str) -> Result<Self> {
        let w = wide(name);
        // SAFETY: as above.
        let handle = unsafe {
            OpenEventW(SYNCHRONIZATION_SYNCHRONIZE | EVENT_MODIFY_STATE, false, PCWSTR(w.as_ptr()))?
        };
        Ok(Self { handle })
    }

    pub fn set(&self) {
        // SAFETY: handle is valid for the lifetime of self.
        unsafe {
            let _ = SetEvent(self.handle);
        }
    }

    /// Wait up to `timeout_ms`; returns true if signaled.
    pub fn wait(&self, timeout_ms: u32) -> bool {
        // SAFETY: handle is valid for the lifetime of self.
        unsafe { WaitForSingleObject(self.handle, timeout_ms) == WAIT_OBJECT_0 }
    }
}

impl Drop for NamedEvent {
    fn drop(&mut self) {
        // SAFETY: handle came from CreateEventW/OpenEventW and is closed once.
        unsafe {
            let _ = CloseHandle(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_and_wait() {
        let name = format!("Local\\resona-test-ev-{}", std::process::id());
        let a = NamedEvent::create(&name).unwrap();
        let b = NamedEvent::open(&name).unwrap();
        assert!(!b.wait(0));
        a.set();
        assert!(b.wait(100));
        assert!(!b.wait(0), "auto-reset");
    }
}
