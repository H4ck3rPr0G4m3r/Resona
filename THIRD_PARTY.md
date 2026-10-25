# Third-party components

Resona builds on the following projects. Their licenses apply to their code and weights. Lock
files (`Cargo.lock`, `engine/uv.lock`, `app/package-lock.json`) list every transitive dependency.

## Models and algorithms

| Component | Used for | License |
|---|---|---|
| [RVC (Retrieval-based Voice Conversion WebUI)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Model format and synthesizer architecture (re-implemented in `engine/resona_engine/rvc`) | MIT |
| [ContentVec](https://github.com/auspicious3000/contentvec) (`lengyue233/content-vec-best` weights) | Speech features | MIT |
| RMVPE (`rmvpe.pt`, as distributed in the RVC project's model repository) | Pitch (F0) estimation | As released with RVC (MIT) |
| [torchfcpe](https://github.com/CNChTu/FCPE) | Alternative pitch estimator | MIT |
| [RNNoise](https://github.com/xiph/rnnoise) via [nnnoiseless](https://github.com/jneem/nnnoiseless) | Noise suppression and voice detection | BSD-3-Clause |
| Freeverb (Jezar at Dreampoint) | Reverb structure | Public domain |

Voice models (`.pth` / `.index`) belong to their authors and are never bundled with or committed to
this repository.

## Engine (Python)

| Package | License |
|---|---|
| PyTorch | BSD-3-Clause |
| Hugging Face Transformers | Apache-2.0 |
| faiss-cpu | MIT |
| NumPy | BSD-3-Clause |
| soundfile (libsndfile) | BSD-3-Clause (LGPL-2.1 library) |
| soxr (libsoxr) | LGPL-2.1 |

## Host and app (Rust)

| Crate | License |
|---|---|
| Tauri and its plugins | MIT or Apache-2.0 |
| wasapi | MIT |
| windows | MIT or Apache-2.0 |
| rtrb | MIT or Apache-2.0 |
| symphonia | MPL-2.0 |
| hound | Apache-2.0 |
| serde, serde_json, parking_lot, crossbeam-channel, thiserror, log | MIT or Apache-2.0 |
| zip | MIT |

## UI

| Package | License |
|---|---|
| Svelte | MIT |
| Vite | MIT |
| Phosphor Icons (`phosphor-svelte`) | MIT |
| Geist and Geist Mono fonts (`@fontsource-variable`) | SIL Open Font License 1.1 |
