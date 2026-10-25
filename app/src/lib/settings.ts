// Shape of %APPDATA%\Resona\settings.json and profiles\<hash>.json (the shell persists them as-is).
import type { DeviceMode, LiveParams, StreamSettings } from "./api";

export const SCHEMA = 1;

export interface Hotkeys {
  mute: string;
  ptt: string;
  next: string;
  prev: string;
}

export interface Settings {
  schema: number;
  input_device: string | null;
  input_mode: DeviceMode;
  output_device: string | null;
  output_mode: DeviceMode;
  monitor_enabled: boolean;
  monitor_device: string | null;
  monitor_mode: DeviceMode;
  monitor_gain_db: number;
  output_gain_db: number;
  jitter_ms: number;
  down_mode: "mute" | "passthrough";
  /** null = use the calibrated block size */
  block_ms: number | null;
  calibration: { block_ms: number; gpu: string; at: number } | null;
  stream: Partial<Omit<StreamSettings, "block_ms">>;
  f0: Pick<LiveParams, "f0_median" | "f0_octave_fix" | "f0_glide_ms" | "f0_threshold">;
  xfade_ms: number;
  last_voice: string | null;
  hotkeys: Hotkeys;
  ptt_enabled: boolean;
  close_to_tray: boolean;
  autostart: boolean;
  start_on_launch: boolean;
  theme: "system" | "dark" | "light";
}

export const DEFAULT_SETTINGS: Settings = {
  schema: SCHEMA,
  input_device: null,
  input_mode: "shared",
  output_device: null,
  output_mode: "shared",
  monitor_enabled: false,
  monitor_device: null,
  monitor_mode: "shared",
  monitor_gain_db: 0,
  output_gain_db: 0,
  jitter_ms: 10,
  down_mode: "mute",
  block_ms: null,
  calibration: null,
  stream: { context_ms: 1000, lookahead_ms: 40, crossfade_ms: 10, sola_search_ms: 10, edge_audio_ms: 40, f0_method: "rmvpe" },
  f0: { f0_median: 5, f0_octave_fix: true, f0_glide_ms: 0, f0_threshold: 0.03 },
  xfade_ms: 50,
  last_voice: null,
  hotkeys: { mute: "Ctrl+Alt+M", ptt: "", next: "Ctrl+Alt+Right", prev: "Ctrl+Alt+Left" },
  ptt_enabled: false,
  close_to_tray: true,
  autostart: false,
  start_on_launch: false,
  theme: "system",
};

/** Per-voice profile: what selecting a voice restores. */
export type Profile = Pick<LiveParams, "pitch" | "formant" | "index_rate" | "protect"> & { schema: number };

export const DEFAULT_PROFILE: Profile = { schema: SCHEMA, pitch: 0, formant: 0, index_rate: 0.5, protect: 0.33 };

/** Merge stored JSON over defaults, one level deep, ignoring keys of the wrong type. */
export function mergeDefaults<T extends object>(defaults: T, stored: unknown): T {
  const out: Record<string, unknown> = structuredClone(defaults) as Record<string, unknown>;
  if (!stored || typeof stored !== "object") return out as T;
  for (const [k, v] of Object.entries(stored as Record<string, unknown>)) {
    if (!(k in out)) continue;
    const d = out[k];
    if (d !== null && typeof d === "object" && !Array.isArray(d) && v && typeof v === "object") {
      out[k] = { ...(d as object), ...(v as object) };
    } else if (d === null || v === null || typeof d === typeof v) {
      out[k] = v;
    }
  }
  return out as T;
}
