"""Worker process round-trips over real shared memory + stdio JSON-RPC (stub backend)."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from resona_engine.ipc.shm import NamedEvent, ShmRing


def test_ring_roundtrip_python():
    name = f"Local\\resona-pytest-ring-{os.getpid()}"
    r = ShmRing.create(name, 4, 32)
    other = ShmRing.open(name)
    assert r.push(5, np.arange(10, dtype=np.float32))
    seq, flags, data = other.pop()
    assert seq == 5 and flags == 0 and np.array_equal(data, np.arange(10))
    assert other.pop() is None


class Worker:
    def __init__(self, prefix: str):
        self.p = subprocess.Popen(
            [sys.executable, "-m", "resona_engine.worker", "--shm", prefix, "--stub"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.next_id = 1

    def call(self, method, params=None, timeout=10.0):
        rid = self.next_id
        self.next_id += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = json.loads(self.p.stdout.readline())
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]
        raise TimeoutError(method)

    def close(self):
        try:
            self.call("engine.shutdown")
        except Exception:
            pass
        self.p.wait(timeout=5)


@pytest.fixture
def worker():
    prefix = f"Local\\resona-pytest-{os.getpid()}"
    rings = (ShmRing.create(prefix + "-in", 16, 1600), ShmRing.create(prefix + "-out", 16, 6144))
    events = (NamedEvent.create(prefix + "-in-ev"), NamedEvent.create(prefix + "-out-ev"))
    w = Worker(prefix)
    yield w, rings, events
    w.close()


def test_stub_worker_streams_blocks(worker):
    w, (rin, rout), (ev_in, ev_out) = worker
    assert w.call("hello")["stub"] is True
    layout = w.call("stream.configure", {"config": {"block_ms": 60}})
    assert layout["block_in"] == 960 and layout["out_len"] == 960 + 160 + 160
    w.call("stream.start")
    got = []
    for seq in range(10):
        assert rin.push(seq, np.full(960, seq, dtype=np.float32))
        ev_in.set()
        assert ev_out.wait(2000)
        item = rout.pop()
        assert item is not None
        got.append(item[0])
        assert item[2].shape == (layout["out_len"],)
    assert got == list(range(10))
    w.call("stream.stop")


def test_bad_config_is_an_error(worker):
    w, *_ = worker
    with pytest.raises(RuntimeError):
        w.call("stream.configure", {"config": {"block_ms": 55}})
    with pytest.raises(RuntimeError):
        w.call("no.such.method")


class FakeRpc:
    def __init__(self):
        self.methods, self.notes = {}, []

    def method(self, name):
        def deco(fn):
            self.methods[name] = fn
            return fn
        return deco

    def notify(self, method, params):
        self.notes.append((method, params))


def test_calibrate_and_voice_cache_on_gpu():
    torch = pytest.importorskip("torch")
    from tests.conftest import index_for, reference_models

    models = [p for p in reference_models() if index_for(p)]
    if not torch.cuda.is_available() or len(models) < 2:
        pytest.skip("needs CUDA and reference models")
    from resona_engine.worker import Engine

    rpc = FakeRpc()
    eng = Engine(rpc, None, stub=False)
    call = lambda m, p=None: rpc.methods[m](p or {})  # noqa: E731
    call("model.load", {"id": "a", "path": str(models[0]), "index_path": str(index_for(models[0]))})
    call("model.activate", {"id": "a"})
    call("stream.configure", {"config": {"block_ms": 60}})
    call("model.load", {"id": "b", "path": str(models[1])})
    assert len(eng.backend._g_synth) == 2  # preloaded voice captured ahead of activation
    res = call("engine.calibrate", {"blocks": [40, 100], "n": 20})
    assert [r["block_ms"] for r in res["results"]] == [40, 100]
    assert res["recommended"] in (40, 100)
    assert eng.cfg.block_ms == 60 and eng.layout.block_in == 960  # configuration restored
    assert len(eng.backend._g_synth) == 2
    call("model.unload", {"id": "b"})
    assert len(eng.backend._g_synth) == 1
