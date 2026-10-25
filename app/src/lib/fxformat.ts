import type { ParamSpec } from "./api";

/** Human formatting of an effect parameter value. */
export function formatParam(p: ParamSpec, v: number): string {
  if (typeof p.scale === "object") return p.scale.choice[Math.round(v)] ?? "";
  if (p.unit === "Hz") return v >= 1000 ? `${(v / 1000).toFixed(v >= 10_000 ? 1 : 2)} kHz` : `${Math.round(v)} Hz`;
  if (p.unit === "dB") return `${v > 0 ? "+" : ""}${v.toFixed(Math.abs(v) < 10 ? 1 : 0)} dB`;
  if (p.unit === "ms") return `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`;
  if (p.unit === ":1") return `${v.toFixed(1)}:1`;
  if (p.unit === "x") return `${v.toFixed(1)}x`;
  if (p.max <= 1 && p.min >= 0) return `${Math.round(v * 100)}%`;
  return v.toFixed(p.max - p.min <= 20 ? 1 : 0);
}

type Shape = "lowshelf" | "peak" | "highshelf";

/** RBJ magnitude (dB) of one EQ band at `f` (48 kHz), matching the engine's band shapes. */
export function bandDb(shape: Shape, f0: number, q: number, gain: number, f: number): number {
  const sr = 48000;
  const w0 = (2 * Math.PI * Math.min(f0, sr * 0.49)) / sr;
  const A = 10 ** (gain / 40);
  const alpha = Math.sin(w0) / (2 * Math.max(q, 0.05));
  const c = Math.cos(w0);
  let b0, b1, b2, a0, a1, a2;
  if (shape === "peak") {
    [b0, b1, b2, a0, a1, a2] = [1 + alpha * A, -2 * c, 1 - alpha * A, 1 + alpha / A, -2 * c, 1 - alpha / A];
  } else {
    const s = 2 * Math.sqrt(A) * alpha;
    if (shape === "lowshelf")
      [b0, b1, b2, a0, a1, a2] = [A * (A + 1 - (A - 1) * c + s), 2 * A * (A - 1 - (A + 1) * c), A * (A + 1 - (A - 1) * c - s), A + 1 + (A - 1) * c + s, -2 * (A - 1 + (A + 1) * c), A + 1 + (A - 1) * c - s];
    else
      [b0, b1, b2, a0, a1, a2] = [A * (A + 1 + (A - 1) * c + s), -2 * A * (A - 1 + (A + 1) * c), A * (A + 1 + (A - 1) * c - s), A + 1 - (A - 1) * c + s, 2 * (A - 1 - (A + 1) * c), A + 1 - (A - 1) * c - s];
  }
  const w = (2 * Math.PI * f) / sr;
  const re = (x0: number, x1: number, x2: number) => x0 + x1 * Math.cos(w) + x2 * Math.cos(2 * w);
  const im = (x1: number, x2: number) => -(x1 * Math.sin(w) + x2 * Math.sin(2 * w));
  const num = re(b0, b1, b2) ** 2 + im(b1, b2) ** 2;
  const den = re(a0, a1, a2) ** 2 + im(a1, a2) ** 2;
  return 10 * Math.log10(Math.max(num / den, 1e-20));
}
