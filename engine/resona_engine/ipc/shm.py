"""Python twin of `host/shm-ipc` (layout documented in host/shm-ipc/src/lib.rs).

x86-64 gives aligned 8-byte loads/stores atomically and in program order (TSO), and CPython
executes these statements sequentially, so writing slot data before publishing the index is
enough for SPSC correctness.
"""

from __future__ import annotations

import ctypes
import mmap
import struct
from ctypes import wintypes

import numpy as np

MAGIC = 0x524E5341
VERSION = 1
OFF_WRITE, OFF_READ, OFF_SLOTS, SLOT_HEADER = 64, 128, 192, 16

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.OpenEventW.restype = wintypes.HANDLE
_k32.CreateEventW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_k32.CreateEventW.restype = wintypes.HANDLE
_k32.SetEvent.argtypes = [wintypes.HANDLE]
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
SYNCHRONIZE, EVENT_MODIFY_STATE, WAIT_OBJECT_0 = 0x00100000, 0x0002, 0


def stride_for(slot_samples: int) -> int:
    return -(-(SLOT_HEADER + 4 * slot_samples) // 64) * 64


class NamedEvent:
    def __init__(self, handle: int):
        if not handle:
            raise OSError(ctypes.get_last_error(), "event open failed")
        self.handle = handle

    @classmethod
    def open(cls, name: str) -> "NamedEvent":
        return cls(_k32.OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, False, name))

    @classmethod
    def create(cls, name: str) -> "NamedEvent":
        return cls(_k32.CreateEventW(None, False, False, name))

    def set(self) -> None:
        _k32.SetEvent(self.handle)

    def wait(self, timeout_ms: int) -> bool:
        return _k32.WaitForSingleObject(self.handle, timeout_ms) == WAIT_OBJECT_0

    def close(self) -> None:
        if self.handle:
            _k32.CloseHandle(self.handle)
            self.handle = 0


class ShmRing:
    def __init__(self, mm: mmap.mmap, slots: int, slot_samples: int):
        self.mm, self.slots, self.slot_samples = mm, slots, slot_samples
        self.stride = stride_for(slot_samples)
        self._view = memoryview(mm)

    @classmethod
    def open(cls, name: str) -> "ShmRing":
        probe = mmap.mmap(-1, OFF_SLOTS, tagname=name)
        magic, version, slots, slot_samples = struct.unpack_from("<IIII", probe, 0)
        probe.close()
        if magic != MAGIC or version != VERSION or slots == 0:
            raise ValueError(f"shared memory {name!r} has an incompatible layout")
        size = OFF_SLOTS + slots * stride_for(slot_samples)
        return cls(mmap.mmap(-1, size, tagname=name), slots, slot_samples)

    @classmethod
    def create(cls, name: str, slots: int, slot_samples: int) -> "ShmRing":
        """Used by tests (the host normally creates rings)."""
        size = OFF_SLOTS + slots * stride_for(slot_samples)
        mm = mmap.mmap(-1, size, tagname=name)
        mm[:size] = bytes(size)
        struct.pack_into("<III", mm, 4, VERSION, slots, slot_samples)
        struct.pack_into("<I", mm, 0, MAGIC)
        return cls(mm, slots, slot_samples)

    def _get(self, off: int) -> int:
        return struct.unpack_from("<Q", self.mm, off)[0]

    def _set(self, off: int, v: int) -> None:
        struct.pack_into("<Q", self.mm, off, v)

    def __len__(self) -> int:
        return self._get(OFF_WRITE) - self._get(OFF_READ)

    def push(self, seq: int, data: np.ndarray, flags: int = 0) -> bool:
        data = np.ascontiguousarray(data, dtype=np.float32)
        n = data.shape[0]
        if n > self.slot_samples:
            raise ValueError(f"block of {n} samples exceeds slot capacity {self.slot_samples}")
        w, r = self._get(OFF_WRITE), self._get(OFF_READ)
        if w - r >= self.slots:
            return False
        off = OFF_SLOTS + (w % self.slots) * self.stride
        struct.pack_into("<QII", self.mm, off, seq, n, flags)
        self._view[off + SLOT_HEADER : off + SLOT_HEADER + 4 * n] = data.tobytes()
        self._set(OFF_WRITE, w + 1)
        return True

    def pop(self) -> tuple[int, int, np.ndarray] | None:
        r, w = self._get(OFF_READ), self._get(OFF_WRITE)
        if r >= w:
            return None
        off = OFF_SLOTS + (r % self.slots) * self.stride
        seq, n, flags = struct.unpack_from("<QII", self.mm, off)
        n = min(n, self.slot_samples)
        data = np.frombuffer(self.mm, dtype=np.float32, count=n, offset=off + SLOT_HEADER).copy()
        self._set(OFF_READ, r + 1)
        return seq, flags, data

    def close(self) -> None:
        self._view.release()
        self.mm.close()
