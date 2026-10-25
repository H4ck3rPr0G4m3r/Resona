# Engine quality baselines (M1, 2026-10-24)

Harness: `resona-engine harness tests/clips/*.wav --model … --index … --pitch …`
Config: block 60 / context 1000 / lookahead 40 / crossfade 10 / search 10 / edge 40 ms, RMVPE, eager PyTorch fp16, RTX 3060 Ti.

| Model (sr) | boundary_excess | seam_ncc | seam_mismatch_db | octave_error_rate | p99 ms |
|---|---|---|---|---|---|
| DonaldTrump (40k, pitch −3) | 1.002 | 0.909 | −9.0 | 0.047* | 60 |
| egirl (48k, +10) | 1.002 | 0.908 | −8.0 | 0.010 | 57 |
| supercreek (40k, +10) | 1.015 | 0.889 | −7.4 | 0.003 | 60 |
| JinwooSung (32k, 0) | 0.999 | 0.832 | −5.5 | 0.003 | 38 |

Reference point: emulating naive RVC streaming (phase reset + fresh noise per block) on the
Trump model gives seam_ncc 0.78 / −4.6 dB vs ours 0.85 / −7.8 dB at the same settings
(before edge mirroring).

\* Measured on the outputs; the very low voice (~76 Hz) confuses the pitch tracker used for
measurement. Input-side pitch agreement between streaming and offline is 97% voicing, 0% octave errors.

Notes
- LSD vs offline (~9–10 dB) sits at the noise floor: two offline renders differing only in flow
  noise are 7.3 dB apart, so seam_ncc / seam_mismatch_db are the meaningful seam metrics.
- Lookahead is the main quality lever: seam_ncc 0.855 → 0.912 → 0.950 → 0.965 at 40/60/120/200 ms.

# M3 speed (2026-10-25)

CUDA graphs (ContentVec and RMVPE captured separately and replayed on parallel streams; enc_p →
flow → decoder captured per voice; F0 post, NSF source, noise windows and index blend stay eager).
Graph output is bit-identical to eager (max |Δ| = 0 over 8 × 154 blocks). Trump 40k + index, RTX 3060 Ti.

| | features p50/p99 | synth p50/p99 | total p50/p99 (ms) |
|---|---|---|---|
| eager (M1) | 16.9 / 33.4 | 12.3 / 25.0 | 30.1 / 53.6 |
| graphs | 8.1 / 9.3 | 6.2 / 7.5 | 14.7 / 16.2 |

Compute is set by the 1 s context window, so it is nearly flat in block size (p99 16–18 ms at
40/60/80/100 ms), and auto-calibration (budget p99 ≤ 0.6 × block) picks 40 ms on this GPU.

Live (mic → VAC Line 1, shared mode, jitter 10 ms): estimated mouth-to-ear 147 ms at 60 ms
blocks, 126 ms at 40 ms; 0 underruns over 60 s at 40 ms.

Seam quality at 40 ms is equal or better (Trump −3, speech + singing): seam_ncc 0.921 → 0.936,
boundary_excess 1.016 → 1.019.
