"""Generate the synthetic regression clips (speech clips come from Windows SAPI TTS).

Run: uv run python tests/clips/make_clips.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
OUT = Path(__file__).parent


def voiced(f0: np.ndarray, formants=((700, 80), (1200, 90), (2600, 120))) -> np.ndarray:
    """Harmonic source shaped by a few resonances (crude vowel)."""
    phase = np.cumsum(2 * np.pi * f0 / SR)
    x = np.zeros_like(f0)
    for k in range(1, 40):
        fk = k * f0
        gain = sum(1.0 / (1.0 + ((fk - fc) / bw) ** 2) for fc, bw in formants) / k**0.5
        gain *= fk < SR / 2
        x += gain * np.sin(k * phase)
    return x


def envelope(n: int, attack=0.03, release=0.05) -> np.ndarray:
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return env


def main() -> None:
    rng = np.random.default_rng(7)
    # Singing: held notes with vibrato and a slow glissando.
    t = np.arange(int(6.0 * SR)) / SR
    base = np.interp(t, [0, 1.5, 2.5, 4.0, 6.0], [220, 220, 330, 262, 196])
    f0 = base * 2 ** (0.3 * np.sin(2 * np.pi * 5.5 * t) / 12)
    sing = voiced(f0) * envelope(len(t))
    # Octave trap: fast glissando across an octave plus breathy noise bursts between phrases.
    t2 = np.arange(int(4.0 * SR)) / SR
    f2 = 150 * 2 ** (np.clip((t2 - 0.5) / 0.4, 0, 1)) * 2 ** (-np.clip((t2 - 2.2) / 0.3, 0, 1))
    trap = voiced(f2) * envelope(len(t2))
    breath = rng.standard_normal(len(t2)) * 0.02
    breath[(t2 > 1.4) & (t2 < 1.8)] *= 6
    trap = trap + breath
    for name, x in (("singing.wav", sing), ("octave_trap.wav", trap)):
        x = 0.5 * x / np.abs(x).max()
        sf.write(OUT / name, x.astype(np.float32), SR, subtype="PCM_16")


if __name__ == "__main__":
    main()
