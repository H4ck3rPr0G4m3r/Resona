"""Scheduling priority of the engine worker (Windows), so a game doesn't starve it.

At normal priority a game that keeps the GPU busy gets as much of it as the engine, and the work
per block hardly depends on the block size (the context window sets it), so blocks come back late
at any block size. Raising this process's GPU scheduling priority, as game-streaming hosts such as
Sunshine do, lets the engine's short bursts go ahead of the game's frames. The CPU priority keeps
the per-block Python work from queueing behind the game's threads.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

# D3DKMT_SCHEDULINGPRIORITYCLASS, highest first. REALTIME (5) is left out: on NVIDIA it can freeze
# or crash the driver (see Sunshine's display_base.cpp).
GPU_CLASSES = (("high", 4), ("above normal", 3))
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY = 0x0020, 0x0008
SE_PRIVILEGE_ENABLED = 0x00000002


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _LUID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Luid", _LUID), ("Attributes", wintypes.DWORD)]


class _TOKEN_PRIVILEGES(ctypes.Structure):
    _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", _LUID_AND_ATTRIBUTES * 1)]


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def _enable_privilege(k32, name: str) -> None:
    """Turn on a privilege the token holds but has disabled (administrators hold the one
    for raising scheduling priority; for standard users this does nothing)."""
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(_LUID)]
    adv.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(_TOKEN_PRIVILEGES), wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
    ]
    token = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ctypes.byref(token)):
        return
    try:
        tp = _TOKEN_PRIVILEGES(1)
        entry = tp.Privileges[0]
        if adv.LookupPrivilegeValueW(None, name, ctypes.byref(entry.Luid)):
            entry.Attributes = SE_PRIVILEGE_ENABLED
            adv.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None)
    finally:
        k32.CloseHandle(token)


def raise_priority() -> dict[str, str]:
    """Best effort. Returns the levels reached, e.g. {"gpu": "high", "cpu": "above normal"};
    "normal" means the raise was refused (GPU: usually fixed by running as administrator)."""
    got = {"gpu": "normal", "cpu": "normal"}
    try:
        k32 = _kernel32()
        proc = k32.GetCurrentProcess()
        if k32.SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS):
            got["cpu"] = "above normal"
        _enable_privilege(k32, "SeIncreaseBasePriorityPrivilege")
        set_gpu = ctypes.WinDLL("gdi32").D3DKMTSetProcessSchedulingPriorityClass
        set_gpu.argtypes = [wintypes.HANDLE, ctypes.c_int]
        set_gpu.restype = ctypes.c_long  # NTSTATUS, 0 = success
        for name, cls in GPU_CLASSES:
            if set_gpu(proc, cls) == 0:
                got["gpu"] = name
                break
    except (OSError, AttributeError):
        pass  # not Windows, or an old gdi32 without D3DKMT exports
    return got
