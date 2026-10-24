"""Newline-delimited JSON-RPC 2.0 over stdio (requests in on stdin, responses/notifications out on stdout)."""

from __future__ import annotations

import json
import sys
import threading
import traceback
from typing import Any, Callable


class RpcError(Exception):
    def __init__(self, message: str, code: int = -32000, data: Any = None):
        super().__init__(message)
        self.code, self.data = code, data


class RpcServer:
    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout
        self._lock = threading.Lock()
        self.methods: dict[str, Callable[[dict], Any]] = {}

    def method(self, name: str):
        def deco(fn):
            self.methods[name] = fn
            return fn

        return deco

    def _send(self, msg: dict) -> None:
        line = json.dumps(msg, separators=(",", ":"), allow_nan=False, default=_default)
        with self._lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def notify(self, method: str, params: Any) -> None:
        try:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})
        except (OSError, ValueError):
            pass  # host went away; the main loop will notice on stdin EOF

    def handle(self, line: str) -> None:
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}})
            return
        rid, name = req.get("id"), req.get("method")
        fn = self.methods.get(name)
        try:
            if fn is None:
                raise RpcError(f"unknown method {name!r}", -32601)
            result = fn(req.get("params") or {})
            if rid is not None:
                self._send({"jsonrpc": "2.0", "id": rid, "result": result})
        except RpcError as e:
            if rid is not None:
                self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": e.code, "message": str(e), "data": e.data}})
        except Exception as e:  # report, never crash the loop
            traceback.print_exc(file=sys.stderr)
            if rid is not None:
                self._send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}})

    def serve(self) -> None:
        for line in self.stdin:
            line = line.strip()
            if line:
                self.handle(line)


def _default(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return [float(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)
