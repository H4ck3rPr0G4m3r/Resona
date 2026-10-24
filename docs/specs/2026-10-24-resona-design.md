# Resona — Design Spec

**Date:** 2026-10-24
**Status:** Approved in conversation; pending written-spec review
**Author:** Peter (nagy.peter.zalan@gmail.com), with Claude

Resona is a real-time GPU voice changer for RVC models. It is a clean-room competitor to
Vonovox 1.6.9 (abandoned, closed-source, compiled Cython core). It aims for lower latency,
fewer artifacts, and a better workflow.

---

## 1. Agreed understanding

**What Peter said**
- Build a competitor to Vonovox, similar in spirit but different, with more features, better
  performance, and better voice quality.
- It is a **personal daily driver first**; a public release may come later.
- The pain points that matter most are **voice quality / artifacts** and **latency**. Gaming
  choppiness and general UI workflow are secondary.
- Model scope: **RVC v2 (including Applio-trained variants) as the core**. The engine sits behind a
  backend interface so newer families (Beatrice 2, zero-shot) can be added later.
- Architecture: **hybrid**. A Rust audio host and a Tauri UI, plus a Python inference worker.
- User-definable filters, with a "Bad Mic" filter as the default example (lower priority).

**Assumptions (stated, not contested)**
- Windows 11 only, NVIDIA only. The target machine is an RTX 3060 Ti (8 GB) routing into VAC Line 1.
- Inference only. No model training in v1.
- Existing RVC v2 `.pth`/`.index` files must load without conversion.

**Success criteria**
1. Mouth-to-ear latency **≤ 150 ms**, measured at a 60 ms block on the 3060 Ti. Vonovox needs
   0.25–0.4 s blocks plus lookahead, an estimated 300–500 ms.
2. Audibly fewer block-boundary artifacts than Vonovox (warble, clicks, octave cracks, smeared
   sibilants). This is backed by the objective "streaming ≈ offline" metric in §8.
3. Zero underruns in normal use with an idle-ish GPU. Glitches are concealed (faded), never crackled.
4. All 10 models in `Vonovox-1.6.9/models` load and convert correctly.

### Reference: Peter's model library (inspected 2026-10-24)

| Sample rate | Count | Notes |
|---|---|---|
| 32 kHz | 7 | Applio-format metadata (`vocoder: HiFi-GAN`, `embedder_model: contentvec`) |
| 40 kHz | 2 | Cyrene is old-format (`sr: "40k"`, `info` key) |
| 48 kHz | 1 | egirl, old-format |

`Irene.pth` declares `vocoder_architecture: hifi_refine` (codename-rvc-fork lineage). Its decoder
keys follow the standard NSF-HiFiGAN layout, but loading must use **strict** state-dict matching.
Any mismatch is a clear "unsupported vocoder variant" error, never silent garbage. 9 of the 10
models have a `.index` file.

---

## 2. Non-goals for v1

The following are out of scope: training/fine-tuning, AMD/Intel GPUs, ASIO, macOS/Linux, an
installer and auto-update, zero-shot conversion, the Beatrice 2 backend, bandwidth extension
(AP-BWE), and licensing/monetization. Each of these is a possible later milestone (§10).

---

## 3. System architecture

```
┌──────────────────────── resona.exe (Rust / Tauri 2) ──────────────────────┐
│  UI (webview, Svelte 5 + TS) ◄── Tauri commands/events ──► App core        │
│                                                                            │
│  Mic ─► [Capture thread] ─► input chain ─► SHM ring ──┐                     │
│         WASAPI low-latency                            │                     │
│                                                       ▼                     │
│  VAC ◄─ [Render thread] ◄─ output chain ◄─ SOLA ◄─ SHM ring ◄──┐            │
│  Monitor ◄────────────────┘                                     │            │
│  Supervisor: spawn / heartbeat / auto-restart worker            │            │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
┌──────────── resona-engine (Python 3.12, uv venv, CUDA) ─────────┴──────────┐
│  Streaming pipeline: ContentVec → F0 → index → RVC synth                    │
│  Runtimes: TensorRT fp16 engines (default) │ PyTorch eager (fallback)       │
│  VoiceBackend interface: RVCv2Backend (v1)                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Processes and responsibilities

| Unit | Language | Owns | Depends on |
|---|---|---|---|
| `app` | Svelte/TS | All UI. No audio logic. | Tauri commands/events |
| `host/audio-io` | Rust | WASAPI devices, RT threads, device change handling | `windows` crate |
| `host/dsp` | Rust | Resampling, input/output chains, SOLA, limiter, filters | `rubato`, `nnnoiseless`, `ort` |
| `host/shm-ipc` | Rust | Shared-memory rings and named events; JSON-RPC client | `windows` crate |
| `host/supervisor` | Rust | Worker lifecycle, health, restart, state machine | `shm-ipc` |
| `engine` | Python | Model loading, streaming inference, TRT build/cache | torch, tensorrt, faiss-cpu (load only) |

### 3.2 Audio data path (shared memory)

- **Input ring:** 16 kHz mono float32, written by the host and read by the engine.
- **Output ring:** model-rate mono float32 (32k/40k/48k), written by the engine and read by the host.
- Both rings are single-producer/single-consumer and lock-free with a fixed capacity (≥ 2 s). They
  live in a named file mapping. Each block carries a header: `seq: u64`, `n_samples: u32`,
  `flags: u32`.
- **Wakeups:** a named auto-reset event per direction (`block_ready`, `output_ready`). Nobody polls.
- The engine keeps its own context history. The host sends only new block samples.
- **Deadline handling:**
  - Each output block has a deadline. If block *N* is not ready when the render thread needs it,
    the host plays a 5 ms fade to silence, increments `underruns`, and stays faded until the next
    on-time block. It then fades back in.
  - A late block *N* that arrives after its deadline is dropped and never played out of order.

### 3.3 Control path

- JSON-RPC 2.0 over a named pipe (`\\.\pipe\resona-engine-<pid>`), newline-delimited.
- **Host → engine methods:**
  - `hello`
  - `assets.status`
  - `model.inspect(path)`
  - `model.load(id, path, index_path)`
  - `model.preload(id)`
  - `model.activate(id, crossfade_ms)`
  - `model.unload(id)`
  - `stream.configure(StreamConfig)`
  - `params.set(LiveParams)`
  - `engine.calibrate()`
  - `engine.shutdown()`
- **Engine → host notifications:** `stats` (10 Hz: per-stage ms, p50/p99, VRAM, GPU util),
  `pitch` (input/output F0 frames for the UI trace), `trt.progress`, `log`.
- `LiveParams` changes take effect at the next block boundary. `StreamConfig` changes cause an
  engine rebuild or warm-up. During that time the host passes audio through dry or mutes it (user
  setting).

### 3.4 Supervisor state machine

```
Stopped → StartingWorker → LoadingModel → BuildingEngines* → Running
              ▲                                                  │
              └──────────── WorkerCrashed (auto-restart ≤3/min) ◄┘
DeviceLost (any state with audio) → Reconnecting → previous state
```

*`BuildingEngines` runs the PyTorch runtime in the meantime, so audio already flows.*

Every state is exposed to the UI. After 3 crashes in a minute, the supervisor stops retrying and
shows the worker log tail.

---

## 4. Engine: streaming inference pipeline

### 4.1 Window model

Each step processes the window `[left context C | new block B | lookahead L]`. Defaults are
C ≈ 1.0 s, B = 60 ms, and L = 40 ms. All sizes are multiples of 10 ms (one 100 fps frame).

```
16k window ─┬─► ContentVec (layer 12, 768-d, 50 fps) ─► 2× interp → 100 fps ─► index blend ─┐
            └─► F0: RMVPE | FCPE (100 fps) ─► stateful F0 post ─────────────────────────────┴─► synth
                (parallel CUDA streams, high priority)
synth = enc_p(full window) → flow(reverse) → NSF decoder(tail only: B + crossfade + RF padding)
```

### 4.2 `VoiceBackend` interface

```python
class VoiceBackend(Protocol):
    def load(self, path: Path, opts: dict) -> ModelInfo
    def configure(self, cfg: StreamConfig) -> None   # sizes → build/warm engines
    def process(self, x16k: Tensor, p: LiveParams) -> Tensor   # → model-rate block (+ xfade tail)
    def reset(self) -> None                           # drop all streaming state
```

`RVCv2Backend` is the only implementation in v1. Model code is our own module definitions, which
follow RVC-WebUI/Applio (both MIT) with attribution recorded in `THIRD_PARTY.md`. We own the code so
we can split the decoder (§4.4).

### 4.3 Latency features

1. **TensorRT fp16 engines** for ContentVec, RMVPE/FCPE, `enc_p + flow`, and the decoder body.
   - Export path: ONNX export with **fixed shapes per StreamConfig**, then a TensorRT build via the
     `tensorrt` Python API, with CUDA-graph replay of the per-block launch sequence.
   - Engine cache: `%LOCALAPPDATA%\Resona\engines\<model_hash>\<config_key>.plan`. The `config_key`
     includes the block/context/lookahead sizes, the TRT version, and the GPU UUID.
   - A first build takes about 1–2 minutes per model and config. It reports `trt.progress` while
     PyTorch keeps serving audio.
2. **Tail-only decode.** `enc_p` and `flow` see the full window. The decoder only runs on
   `B + crossfade + receptive-field padding` frames, the same idea as RVC realtime's
   `skip_head`/`return_length`.
3. **Parallel stages.** ContentVec and F0 run on two high-priority CUDA streams and join before
   index blending.
4. **Steady-state hygiene.**
   - All tensors are preallocated per StreamConfig.
   - `gc.freeze()` runs after warm-up, and automatic GC is disabled while streaming. GC runs manually
     only on idle or reconfigure.
   - Nothing allocates per block.
5. **Auto-calibration** (`engine.calibrate`). It runs ~200 synthetic blocks at candidate block
   sizes (40/50/60/80/100 ms) and picks the smallest where `p99 compute ≤ 0.6 × B`. The UI shows
   the result and allows a manual override.

**Latency budget** (B = 60 ms): B 60 + L 40 + crossfade 10 + compute ~15 + device buffers ~10–20
= **≈ 135–145 ms**.

### 4.4 Quality features

1. **Phase-continuous NSF excitation.**
   - The decoder is split into (a) our own stateful fp32 **source module** (sine harmonics + noise,
     CUDA ops in PyTorch) and (b) the decoder body (TRT fp16), which takes the excitation as an
     extra input.
   - The source module carries the phase accumulator keyed to **absolute sample time**. The
     excitation for the overlapping region is therefore identical between consecutive calls, and
     no phase reset happens at block starts.
   - This also keeps the sine generator in fp32, which avoids fp16 phase error.
2. **Time-locked noise.** Every random draw comes from a fixed noise buffer indexed by absolute
   frame or sample position, so overlapping regions match exactly. That covers the flow's `z_p`
   noise (`× 0.66666`), SineGen's additive noise, and unvoiced noise.
3. **Stateful F0 post-processing**, which keeps history across blocks:
   - Median filter (5 frames) over the history/block seam.
   - Octave-jump correction: a frame differing by ≈ ±1200 cents from a running median of the last
     ~150 ms of voiced frames gets folded back.
   - Voiced/unvoiced hysteresis (separate on/off thresholds, minimum segment length of 3 frames).
   - Optional pitch glide (one-pole slew, in ms).
   - Then the pitch shift (semitones) is applied and coarse F0 is quantized for the model.
4. **Consonant/breath protection.**
   - RVC `protect` (default 0.33) applies on unvoiced frames: features there are a `protect`-weighted
     mix of the indexed and un-indexed features.
   - Per-frame index gating: the index rate is scaled down on unvoiced frames, so sibilants keep
     the speaker's own articulation.
5. **GPU exact index.**
   - On load, the `.index` vectors are reconstructed once via faiss-cpu and cached as
     `<model_hash>.f16.npy`.
   - In VRAM, top-k (k = 8) uses `‖a‖² − 2ab + ‖b‖²` as a matmul with 1/d² weights, same as RVC.
   - It is exact (no IVF approximation) and costs about 1 ms. That makes high index rates cheap.
6. **Formant shift.**
   - The 16k input to the embedder is resampled by `2^(shift/12)`, and the resulting features are
     re-interpolated onto the original 100 fps grid.
   - F0 always comes from the unshifted audio, so pitch is unaffected.
   - This must be validated by ear in M1. The fallback is an STFT spectral-envelope warp inside the
     already-buffered window, which adds no extra latency.

### 4.5 Assets

- **Assets:** ContentVec (`lengyue233/content-vec-best`), RMVPE (`rmvpe.pt`), and FCPE
  (`fcpe_c_v001.pt`).
- **First-run fetch:** assets are downloaded with SHA-256 pins into `%LOCALAPPDATA%\Resona\assets\`.
- **Import from Vonovox:** "Import from existing Vonovox install" copies the files from
  `Vonovox-1.6.9/assets/` after verifying their hashes.

### 4.6 VRAM budget (one voice, approx.)

ContentVec ~200 MB, RMVPE ~100 MB, synth ~120 MB, index ≤ 300 MB, plus TRT activations. The total
is under 1.5 GB. Two or three voices can be preloaded on 8 GB alongside a game.

---

## 5. Host: audio I/O and DSP

### 5.1 Devices

- WASAPI shared mode uses `IAudioClient3::InitializeSharedAudioStream` with the minimum period the
  driver allows (~2.67–3 ms at 48 kHz). If IAudioClient3 isn't supported, it falls back to standard
  `IAudioClient` (10 ms). The UI shows which mode is active per device.
- Exclusive mode is optional, per device.
- Capture and render threads register with MMCSS "Pro Audio". Nothing on those threads allocates,
  locks, or does I/O. Parameters arrive through atomics or triple buffers.
- Devices can run at any rate; the host resamples with `rubato`.
  - Capture side: device rate → 48k for the input chain → 16k for the engine.
  - Render side: model rate → 48k for the output chain → device rate.
- `IMMNotificationClient` handles device add/remove/default change, which triggers `DeviceLost`,
  then `Reconnecting`.

### 5.2 Input chain (48 kHz, before the engine)

`input gain → HPF 70 Hz → RNNoise (nnnoiseless, 480-sample frames) → 16k resample → engine`

- Each stage can be toggled. A **"Singing mode"** preset bypasses RNNoise and the VAD gate.
- **Silero VAD** (ONNX via `ort`, CPU, 512-sample frames at 16k) runs on the input, but its gate is
  applied **at the output, aligned to the delayed audio**. The decision is known before the
  corresponding converted audio is rendered, so the gate adds no latency and doesn't clip word
  onsets. It has hold/release controls, with a default release of 300 ms.

### 5.3 SOLA crossfade (render side, model rate)

- The engine returns `B + X` samples, where X is the crossfade length (default 10 ms).
- The host searches for the best alignment offset within ±search (default 5 ms) using normalized
  cross-correlation between the previous tail and the new head.
- It then applies an **equal-power** crossfade and stores the new tail.

### 5.4 Output chain (48 kHz)

- The chain is reorderable, each effect has an enable toggle, and all effects are free.
- **Effects:**
  - **Carried over from Vonovox:** gain, noise gate, compressor, LPF/HPF (12/24 dB/oct with
    resonance), reverb, chorus, lo-fi mic.
  - **New:** de-esser (split-band, 4–10 kHz detector) and a 5-band parametric EQ (shelf/peak bands).
  - **True-peak limiter:** fixed last, on by default, ceiling −1 dBTP.
- Every parameter is **smoothed** (one-pole, ~20 ms) at audio rate, so moving controls mid-speech
  never pops.
- The chain can be exported and imported as a `.resona-chain.json` preset.

### 5.5 Routing and extras

- **Monitor output:** a second device with its own volume. The tap point is selectable: dry,
  converted (pre-chain), or final.
- **Sound-file input:** WAV/MP3/FLAC decoded with `symphonia`, with play/pause/seek. While it
  plays, the file replaces the mic and the mic is zeroed.
- **A/B recorder:** writes synchronized `dry.wav` and `wet.wav` (latency-compensated) into
  `%USERPROFILE%\Music\Resona\`.
- **Global hotkeys** via `tauri-plugin-global-shortcut`: mute toggle, push-to-talk (press/release),
  next/previous voice, and a per-voice slot.
- **Meters:** input/output peak and RMS, gate/VAD state, and the latency breakdown. They are pushed
  to the UI at 30 Hz.

---

## 6. User-definable filters (lower priority, M6)

### 6.1 Composite filters (data, no code)

- `.toml` files live in `filters/` (built-in) and `%APPDATA%\Resona\filters\` (user). Files are
  hot-reloaded on save, and the editor UI is generated from the file.
- A composite filter is a chain of built-in **primitives** plus **macros** (user-facing knobs).
  Primitive parameters may be numbers or expressions over macros.
- **Expressions:** a tiny safe language (`lerp`, `clamp`, `exp`, `+ - * /`, numeric literals, macro
  names). It is evaluated on the control thread when a macro changes, never per sample. Parse
  errors appear inline in the UI, and the filter then stays bypassed.
- **Primitives:** `highpass`, `lowpass`, `bandpass`, `peak`, `lowshelf`, `highshelf`, `downsample`
  (with aliasing), `bitcrush`, `saturate` (soft/hard/tube), `compressor`, `noise`, `crackle`
  (random dropouts and clicks), `delay`, `gain`.

### 6.2 Native effects (code)

```rust
pub trait Effect: Send {
    fn params(&self) -> &'static [ParamSpec];   // name, range, default, unit, curve
    fn prepare(&mut self, sample_rate: f32, max_block: usize);
    fn set_param(&mut self, idx: usize, value: f32);   // smoothed internally by the framework
    fn process(&mut self, buf: &mut [f32]);
    fn reset(&mut self);
}
```

- A native effect is one file in `host/dsp/src/effects/` plus one line in the registry.
- The UI editor and parameter smoothing come for free.
- Every native effect is automatically usable as a composite-filter primitive.

### 6.3 Default filter: Bad Mic (`filters/bad_mic.toml`)

```toml
name = "Bad Mic"
[macros]
strength  = { default = 0.6 }
telephone = { default = 0.4 }
static    = { default = 0.2 }

[[chain]]
type = "highpass"
freq = "lerp(80, 400, telephone)"

[[chain]]
type = "lowpass"
freq = "lerp(12000, 3400, telephone)"

[[chain]]
type = "peak"
freq = 2500
gain_db = "6 * strength"
q = 1.2

[[chain]]
type = "downsample"
rate = "lerp(48000, 11025, strength)"

[[chain]]
type = "bitcrush"
bits = "lerp(16, 8, strength)"

[[chain]]
type = "saturate"
mode = "soft"
drive = "lerp(1, 4, strength)"

[[chain]]
type = "compressor"
threshold_db = -20
ratio = 4

[[chain]]
type = "noise"
level_db = "lerp(-80, -40, static)"

[[chain]]
type = "crackle"
rate = "4 * static"
```

---

## 7. UI and workflow

The visual design is produced with the frontend skill during M4. This section fixes structure and
behavior only. The direction is an instrument/studio-hardware feel that is dense but legible. It
should avoid generic dashboard cards and the Vonovox slider wall.

**Layout:** one window, three zones.
- **Voice library (left rail)**
  - Drag-drop `.pth`/`.index`/`.zip` import with automatic index pairing.
  - Shows metadata: sample rate, version, embedder, f0, vocoder.
  - Favorites, search, and a per-voice hotkey slot.
  - "Import folder" for bulk import (e.g. the Vonovox `models/` dir). Imports are **copied** into
    `%APPDATA%\Resona\voices\<model_hash>\`, so the library owns its files.
- **Live view (center)**
  - Start/stop button.
  - Pitch and formant as primary controls.
  - **Live pitch trace:** input F0 vs output F0, scrolling over ~5 s.
  - **Latency bar:** device-in / block / lookahead / compute / xfade / device-out in ms, plus
    underrun count and GPU ms per block.
- **Signal chain (bottom strip)**
  - Input and output chains drawn as a signal-flow row.
  - Click to expand an editor in place, drag to reorder.
  - The EQ has an interactive response curve.
- **Settings drawer:** devices, monitoring, block size (auto-calibrated plus override), advanced
  pipeline parameters (context, lookahead, crossfade, SOLA search, F0 method, F0 post settings),
  and hotkeys.

**Voice profiles**
- Each voice stores pitch, formant, index rate, protect, F0 method, and an optional per-voice output
  chain. Selecting a voice restores all of it.
- **Seamless switching:** the target voice is preloaded and activated at a block boundary with an
  equal-power crossfade (default 50 ms). No stop/start.

**States that must be designed:**
- Engine starting
- Fetching assets
- Building TensorRT engines (with progress)
- Running
- Worker restarting
- Worker failed (with log tail)
- Device lost/reconnecting
- Underruns occurring
- Empty library
- Model load error (e.g. unsupported vocoder variant)

**Tray:** minimize to tray, optionally start with Windows, and restore the last voice.

**Persistence:** `%APPDATA%\Resona\settings.json`, `profiles\<model_hash>.json`, and `chains\*.json`.
All are versioned with a `schema` field.

---

## 8. Testing and verification

### 8.1 Engine (Python, pytest)

- **Streaming ≈ offline harness** (the core quality metric). The same WAV is rendered (a) block by
  block through the streaming path and (b) as one full-file pass. The reports are:
  - Log-spectral distance overall, and in ±10 ms windows around block boundaries. **Boundary
    excess** = boundary LSD ÷ overall LSD. The target is ≤ 1.1.
  - F0 RMSE in cents, and **octave error rate**: the share of voiced frames with > 600 cents
    deviation.
  - The M1 exit target is boundary excess ≤ 1.1. Per-model regression thresholds are then frozen
    from the M1 results, so later changes can't silently degrade quality.
- **Regression clips** live in `engine/tests/clips/`: speech, sibilant-heavy, breathy, singing, and
  an octave-jump trap (a fast glissando). Golden metrics are stored per model.
- **Unit tests:** source-module phase continuity (no discontinuity across a seam), time-locked
  noise determinism, the F0 post-processor on synthetic contours, GPU index vs faiss results, and
  strict loading of all 10 reference models.

### 8.2 Host (Rust, cargo test)

- DSP: filter frequency responses vs analytic values, the limiter ceiling, SOLA alignment on
  shifted sines, and parameter smoothing (no step > ε).
- Shared-memory rings: multi-threaded stress, wraparound, and seq ordering.
- **Soak test:** a stub engine (echo or naive pitch shift) runs for 30 min with zero underruns as
  the pass condition.
- Composite filters: TOML parsing, expression evaluation, and error reporting.

### 8.3 End-to-end

- **Latency self-test:** the user selects a VAC loopback, the host emits a click, and the sample
  offset is measured through the full chain. The result is compared to the latency bar's estimate.
- **Performance targets on the RTX 3060 Ti:**
  - At B = 60 ms and C = 1 s with TRT: p99 engine compute ≤ 20 ms.
  - Mouth-to-ear ≤ 150 ms.
  - Under 1 underrun per 10 min with an idle GPU.

---

## 9. Repository layout

```
Resona/
  app/                 Tauri 2 shell + Svelte 5/TS UI (Vite)
  host/
    audio-io/          WASAPI, RT threads, device notifications
    dsp/               resample, chains, SOLA, limiter, effects/, filters (composite)
    shm-ipc/           SHM rings, named events, JSON-RPC client
    supervisor/        worker lifecycle + state machine
  engine/              Python 3.12 (uv): resona_engine/{pipeline,backends/rvc_v2,trt,assets,rpc}
    tests/             pytest + clips/
  filters/             built-in composite filters (bad_mic.toml)
  docs/specs/          this spec
  docs/plans/          implementation plans
  THIRD_PARTY.md       attributions/licenses (RVC, Applio, RMVPE, FCPE, ContentVec, Silero, nnnoiseless)
```

The Python env is managed by `uv` (torch cu128, tensorrt-cu12, onnx, faiss-cpu, numpy). In v1 the
host launches the engine with the venv's `python.exe`; packaging comes later.

---

## 10. Milestones

Each milestone is independently usable and has exit criteria.

| # | Milestone | Scope | Exit criteria |
|---|---|---|---|
| M1 | **Engine offline** | RVCv2Backend in PyTorch, streaming pipeline, quality features 1–6, CLI `resona-engine convert in.wav out.wav --model …`, streaming≈offline harness | All 10 models load strictly; boundary excess ≤ 1.1 on regression clips; streamed output by ear is free of block-edge warble |
| M2 | **Host + real-time** | audio-io, shm-ipc, SOLA, supervisor, JSON-RPC, minimal functional UI (device pick, model pick, pitch, start/stop) | Live conversion into VAC works; 30-min stub soak passes; worker crash auto-recovers. **Daily-driver point.** |
| M3 | **Speed** | ONNX→TRT per stage, engine cache, CUDA streams + graphs, tail-only decode, auto-calibration | p99 ≤ 20 ms at B = 60 ms; measured mouth-to-ear ≤ 150 ms |
| M4 | **Full UI** | Frontend-skill design pass, voice library + import, profiles, seamless switching, pitch trace, latency bar, monitoring, hotkeys, tray | All §7 states implemented and reachable; voice switch is glitch-free |
| M5 | **Effects** | Input chain (RNNoise, delayed-VAD gate), full output chain + smoothing + limiter, A/B recorder, sound-file input | DSP tests pass; no pops when sweeping any control mid-speech |
| M6 | **Filters** | Composite filter system, expressions, hot-reload, Bad Mic, `Effect` trait registry | Bad Mic ships; a new composite filter is added by dropping in a TOML file |
| Later | — | AP-BWE bandwidth extension, Beatrice 2 backend, ASIO, AMD/DirectML, installer/auto-update, game-mode GPU throttling | — |

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| ONNX/TRT export of the split decoder or fixed-shape enc_p is fiddly | Keep PyTorch eager fp16 + CUDA graphs as a fully supported runtime. TRT is an optimization, not a dependency. |
| The Realtek USB mic driver lacks IAudioClient3 low-latency periods | Automatic fallback to 10 ms; the latency bar shows the real device period. |
| VAC's own buffering adds latency | Document the recommended VAC settings (48 kHz, mono, minimal buffer); the self-test measures reality. |
| Python GC/allocation jitter in the worker | Preallocation, `gc.freeze()` + disabled automatic GC while streaming; output deadline concealment in the host. |
| Formant shift via resampled embedder input sounds poor | M1 validates by ear; fallback is a spectral-envelope warp (§4.4.6). |
| Applio vocoder variants (RefineGAN, MRF-HiFiGAN, `hifi_refine`) differ in structure | Strict loading with explicit errors; variant support is added per architecture as the library needs it. |
| GPU contention while gaming | High-priority CUDA streams + TRT's lower compute; dedicated game-mode work is a later milestone. |
