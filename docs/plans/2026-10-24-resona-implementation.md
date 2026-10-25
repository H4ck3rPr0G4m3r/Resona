# Resona Implementation Plan (M1–M6)

> **For agentic workers:** executed natively (superpowers:executing-plans) by the session agent, at
> Peter's explicit request ("start the whole implementation, stop only for questions"). Tasks are
> listed as a map with files, interfaces, tests, and exit criteria; code lives in the repo, not here.
> Commit and push after every task.

**Goal:** build Resona per the spec, from the offline engine (M1) through user-definable filters (M6).

**Architecture:** a Rust host (WASAPI audio, DSP, SOLA, supervisor) inside a Tauri 2 app with a
Svelte 5 UI. A Python 3.12 inference worker (PyTorch → TensorRT) talks to the host over shared-memory
audio rings plus JSON-RPC.

**Tech stack:** Rust 1.99 (windows 0.62, wasapi, rubato, nnnoiseless, ort, tauri 2), Python 3.12
(uv, torch cu128, transformers, faiss-cpu, tensorrt-cu12), Svelte 5 + TypeScript + Vite.

**Spec:** `docs/specs/2026-10-24-resona-design.md`

## Global constraints

- Windows 11 x64, NVIDIA CUDA only. Target GPU: RTX 3060 Ti (8 GB).
- Must strictly load all 10 models in `C:\Pedro\AiVoiceChanger\Vonovox-1.6.9\models`.
- Never commit model/asset/audio binaries (see `.gitignore`).
- Real-time threads: no allocation, locks, or I/O.
- Engine I/O contract: 16 kHz mono f32 in, model-rate mono f32 out.
- Block, context, and lookahead are multiples of **20 ms**, because ContentVec runs at 50 fps.
  - This deviates from the spec's 10 ms. Calibration candidates are 40/60/80/100 ms.
- Control channel is JSON-RPC over the worker's **stdio** (newline-delimited); logs go to stderr.
  - The spec named a named pipe; stdio is simpler and has the same semantics.

## Review focus

1. **Silence/very quiet input.** F0 is all zero; the output must be quiet with no NaN, no buzz, no
   DC. Test: `test_pipeline_silence_is_finite_and_quiet` (Task 1.5).
2. **Models without an index, or with `index_rate=0`.** These must convert identically to having no
   index. Test: `test_no_index_path` (Task 1.5).
3. **Old-format checkpoints** (`sr: "40k"`, `info` key, no `vocoder`). Test:
   `test_load_all_reference_models` (Task 1.3).
4. **Worker dies mid-stream.** The host must conceal the gap, restart the worker, and resume. Test:
   the supervisor test kills the stub (Task 2.5).
5. **Device sample rate ≠ 48 kHz, or a device unplugged while running.** Test: the device-open path
   handles a non-48k mix format through autoconvert (Task 2.2). Unplug is a manual check.

---

## M1 — Engine offline

### Task 1.1: Assets + model metadata
- Files: `engine/resona_engine/assets.py`, `engine/resona_engine/rvc/checkpoint.py`, `tests/test_checkpoint.py`
- Produces:
  - `assets.resolve() -> AssetPaths(contentvec_dir, rmvpe, fcpe)`: searches
    `%LOCALAPPDATA%\Resona\assets`, then imports from a Vonovox install (with hash check).
  - `checkpoint.inspect(path) -> ModelInfo(sr:int, version, f0:bool, vocoder, config:list, hash)`.
  - `checkpoint.load_state(path) -> (ModelInfo, state_dict)`. Normalizes `"40k"` → 40000 and drops
    `enc_q.*`.
- Test: inspect all reference models; sr ∈ {32000, 40000, 48000}.

### Task 1.2: RMVPE + ContentVec feature extractors
- Files: `rvc/rmvpe.py` (E2E net, mel, decode), `rvc/contentvec.py` (HF HubertModel, layer-12 output), `tests/test_features.py`
- Produces:
  - `RMVPE(path, device, half).infer(x16k: Tensor[T]) -> Tensor[T//160]` (Hz, 0 = unvoiced).
  - `ContentVec(dir, device, half).extract(x16k) -> Tensor[1, T//320, 768]`.
- Test: strict weight load. A 220 Hz synthetic vowel-ish tone gives F0 ≈ 220 ± 3 Hz. Mel filterbank
  matches librosa (htk, slaney) within 1e-5.

### Task 1.3: Synthesizer (our module definitions)
- Files: `rvc/attentions.py`, `rvc/modules.py`, `rvc/synth.py` (TextEncoder, flow, `GeneratorNSFBody`), `rvc/source.py` (stateful `PhaseContinuousSource`), `tests/test_synth.py`
- Produces: `Synth.from_checkpoint(path, device, half)` with methods
  - `encode_flow(phone, pitch, noise) -> z`
  - `decode(z, excitation) -> wav`
- Test: `test_load_all_reference_models` (strict). Source phase is continuous across a seam
  (max |Δ| below the in-block maximum). Noise is deterministic per absolute index.

### Task 1.4: Index + F0 post-processing
- Files: `rvc/index.py` (faiss → fp16 GPU matrix, cache `.f16.npy`, top-k blend), `pipeline/f0post.py`, `tests/test_index.py`, `tests/test_f0post.py`
- Produces:
  - `GpuIndex.load(path, device)` and `.blend(feats, rate) -> feats`.
  - `F0Post(cfg)` with `.process(abs_frame0, f0_window) -> f0`, `.commit(upto_abs_frame)`, and `.reset()`.
- Test: GPU top-k matches faiss exhaustive results. Octave-jump correction. Hysteresis. Median
  across a seam.

### Task 1.5: Streaming backend + offline reference + CLI
- Files: `backends/base.py` (Protocol, `StreamConfig`, `LiveParams`), `backends/rvc_v2.py` (`RVCv2Backend`), `pipeline/offline.py`, `cli.py`, `tests/test_pipeline.py`
- Produces:
  - `RVCv2Backend.process(x16k_block) -> Tensor[(B+X)·sr]`.
  - `offline.convert(backend_cfg, wav16k) -> wav`.
  - CLI `resona-engine convert in.wav out.wav --model M [--index I] [--pitch N] [--streaming]`.
- Test: silence is finite and quiet; the no-index path; streaming output length matches.

### Task 1.6: Quality harness
- Files: `engine/resona_engine/harness.py`, `tests/test_harness.py`, `tests/clips/*.wav` (synthesized + short recordings generated by script)
- Produces: CLI `resona-engine harness --model M clip.wav` → JSON with `lsd`, `boundary_excess`,
  `f0_rmse_cents`, and `octave_error_rate`.
- Exit (M1): boundary_excess ≤ 1.1 on the reference clips, and all 10 models load.

## M2 — Host + real-time

### Task 2.1: Cargo workspace + shm-ipc crate
- Files: `Cargo.toml` (workspace), `host/shm-ipc/src/{lib,ring,event}.rs`, `engine/resona_engine/ipc/shm.py`
- Produces:
  - Rust: `ShmRing::create(name, capacity)`, `write_block(seq, &[f32])`, `read_block() -> Option<(u64, Vec<f32>)>`.
  - Python: `ShmRing.open(name)`, same semantics. Layout: header {magic, cap, write_idx, read_idx} + slots.
- Test: Rust stress test (producer/consumer threads). A Python↔Rust round-trip test via a spawned
  python process.

### Task 2.2: audio-io crate (WASAPI)
- Files: `host/audio-io/src/{lib,devices,stream}.rs`
- Produces:
  - `list_devices(Direction) -> Vec<DeviceInfo>`.
  - `CaptureStream::start(id, |&[f32]|)` and `RenderStream::start(id, |&mut [f32]|)`.
  - Both run at 48 kHz mono f32 with autoconvert, event-driven, on an MMCSS "Pro Audio" thread,
    with device-loss callbacks.
- Test: enumerate devices; open/close default devices (ignored in CI, run locally).

### Task 2.3: dsp crate core (resampler, SOLA, FIFO, conceal)
- Files: `host/dsp/src/{lib,resample,sola,fade,meter}.rs`
- Produces: `Resampler::new(in_sr, out_sr, max_block)`, `Sola::new(sr, xfade, search).process(block) -> Vec<f32>`,
  `Meter`.
- Test: SOLA aligns shifted sines (no discontinuity). Resampler preserves a 1 kHz tone. Equal-power
  sum is ≈ constant.

### Task 2.4: Engine worker (Python side)
- Files: `engine/resona_engine/worker.py`, `ipc/rpc.py`
- Produces: `python -m resona_engine.worker --shm NAME`, a JSON-RPC loop over stdio (methods per the
  spec §3.3), and a streaming loop thread on SHM + events. Stub mode: `--stub` echoes input.
- Test: pytest spawns the worker in `--stub`, round-trips blocks.

### Task 2.5: Supervisor + runtime
- Files: `host/supervisor/src/lib.rs`, `host/runtime/src/{lib,pipeline,state}.rs`
- Produces:
  - `Runtime::new(cfg)`, `.start()`, `.stop()`, `.set_params()`, `.load_model()`.
  - Events: `State`, `Stats`, `Meters`, `Pitch`.
  - The state machine from spec §3.4, plus the concealment fade on deadline miss.
- Test: the stub worker is killed mid-stream → state goes to `WorkerCrashed`, then `Running`.

### Task 2.6: Tauri app + minimal UI
- Files: `app/` (Vite + Svelte 5), `app/src-tauri/` (commands/events)
- Exit (M2): live conversion into VAC from the minimal UI (devices, model, pitch, start/stop).

## M3 — Speed

### Task 3.1: CUDA graphs + static buffers + GC hygiene
- Static-shape per-StreamConfig buffers. Graphs: `features` (ContentVec + RMVPE on two streams) and
  `synth` (index + enc/flow + decode).
- Test: graph output equals eager (max abs diff < 1e-3); no allocations per block
  (`torch.cuda.memory_stats` stable).

### Task 3.2: TensorRT engines (optional runtime)
- ONNX export per stage with fixed shapes, a TRT build cache keyed per spec §4.3, and fallback to
  graphs on failure. Progress is reported via the `trt.progress` notification.

### Task 3.3: Auto-calibration
- `engine.calibrate` picks the smallest B ∈ {40, 60, 80, 100} ms with p99 ≤ 0.6·B.
- Exit (M3): p99 ≤ 20 ms at B = 60 ms; latency self-test ≤ 150 ms.

## M4 — Full UI (frontend skill)
- 4.1 Voice library (import/copy, metadata, favorites, search) + persistence (`settings.json`,
  `profiles/`).
- 4.2 Profiles + seamless switching (`model.preload`/`model.activate` with crossfade).
- 4.3 Live view: pitch trace canvas, latency bar, meters, all §7 states.
- 4.4 Settings drawer, monitoring output, global hotkeys, tray.

## M5 — Effects
- 5.1 `Effect` trait + parameter smoothing + chain (reorder, enable, export/import).
- 5.2 Effects: gain, gate, compressor, LPF/HPF, 5-band EQ, de-esser, reverb, chorus, lo-fi, limiter.
- 5.3 Input chain: HPF, RNNoise (nnnoiseless), Silero VAD with delayed gate, singing mode.
- 5.4 A/B recorder, sound-file input (symphonia), latency self-test.
- Status (done): all of the above.
  - Deviation: the voice gate uses RNNoise's VAD (already running for noise suppression) rather
    than Silero. This avoids an ONNX runtime in the host, and the gate is still lookahead-aligned
    (20 ms) on the capture timeline.
  - A/B takes are sample-aligned. `tests/ab.rs` puts wet within ±2 samples of dry, through the
    real pipeline, and a real RVC take shows 0 ms envelope lag.
  - Self-test (RTX 3060 Ti, B = 40 ms, VAC): 144 ms measured (spread ~9 ms), meeting the M3 exit
    target of ≤ 150 ms. The latency bar estimates 127 ms; the gap is device and driver buffering
    that the estimate cannot see.

## M6 — Filters
- 6.1 Expression language + TOML composite filters + primitives registry.
- 6.2 Hot-reload, generated UI, Bad Mic.
- Status: deferred (not started).
