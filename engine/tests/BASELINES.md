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
