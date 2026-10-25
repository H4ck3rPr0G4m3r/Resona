// Typed bridge to the Tauri commands/events in app/src-tauri/src/lib.rs.
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type EngineState =
  | "stopped"
  | "starting"
  | "loading"
  | "running"
  | "restarting"
  | "device_lost"
  | "failed";

export interface DeviceInfo {
  id: string;
  name: string;
  direction: "capture" | "render";
  is_default: boolean;
  mix_rate: number;
  mix_channels: number;
}

export type DeviceMode = "shared" | "low_latency" | "exclusive";

export interface DeviceStreamConfig {
  device_id: string | null;
  mode: DeviceMode;
}

export interface VoiceRef {
  id: string;
  path: string;
  index_path?: string | null;
}

export interface StreamSettings {
  block_ms: number;
  context_ms: number;
  lookahead_ms: number;
  crossfade_ms: number;
  sola_search_ms: number;
  edge_audio_ms: number;
  f0_method: "rmvpe" | "fcpe";
}

export interface LiveParams {
  pitch: number;
  formant: number;
  index_rate: number;
  index_unvoiced: number;
  protect: number;
  f0_median: number;
  f0_octave_fix: boolean;
  f0_glide_ms: number;
  f0_threshold: number;
}

export interface SessionConfig {
  input: DeviceStreamConfig;
  output: DeviceStreamConfig;
  monitor: DeviceStreamConfig | null;
  stream: Partial<StreamSettings>;
  voice: VoiceRef;
  params: Partial<LiveParams>;
  jitter_ms: number;
  down_mode: "mute" | "passthrough";
}

export interface ModelInfo {
  path: string;
  name: string;
  sr: number;
  version: string;
  f0: boolean;
  vocoder: string;
  vocoder_architecture: string | null;
  embedder: string;
  hash: string;
  epoch: number | null;
}

export interface Latency {
  input_ms: number;
  block_ms: number;
  lookahead_ms: number;
  compute_ms: number;
  buffer_ms: number;
  output_ms: number;
  total_ms: number;
}

export interface Stats {
  processed: number;
  dropped: number;
  backlog: number;
  p50_ms: number;
  p99_ms: number;
  features_ms?: number;
  f0post_ms?: number;
  synth_ms?: number;
  vram_mb?: number;
  latency?: Latency;
  underruns?: number;
  fifo_ms?: number;
  sola_ncc?: number;
  devices?: {
    input: { device_name: string; mode: string; latency_ms: number };
    output: { device_name: string; mode: string; latency_ms: number };
    monitor: { device_name: string; mode: string; latency_ms: number } | null;
  };
}

export interface CalibrationResult {
  results: { block_ms: number; p50_ms: number; p99_ms: number; ok: boolean }[];
  recommended: number;
  budget: number;
}

export type RuntimeEvent =
  | { type: "state"; state: EngineState; message: string | null }
  | ({ type: "stats" } & Stats)
  | { type: "meters"; input_peak: number; input_rms: number; output_peak: number; output_rms: number }
  | { type: "pitch"; input: number[]; output: number[] }
  | { type: "progress"; what: string; done: number; total: number }
  | { type: "log"; line: string };

export const api = {
  listDevices: () => invoke<{ inputs: DeviceInfo[]; outputs: DeviceInfo[] }>("list_devices"),
  status: () => invoke<{ state: EngineState; message: string | null; engine_dir: string | null }>("get_status"),
  start: (cfg: SessionConfig) => invoke<void>("start_session", { cfg }),
  stop: () => invoke<void>("stop_session"),
  setParams: (params: Partial<LiveParams>) => invoke<LiveParams>("set_params", { params }),
  inspectModel: (path: string) => invoke<ModelInfo>("inspect_model", { path }),
  calibrate: (voice: VoiceRef, stream?: Partial<StreamSettings>) =>
    invoke<CalibrationResult>("calibrate", { voice, stream: stream ?? null }),
  activateVoice: (voice: VoiceRef) => invoke<void>("activate_voice", { voice }),
  preloadVoice: (voice: VoiceRef) => invoke<ModelInfo>("preload_voice", { voice }),
  setMuted: (muted: boolean) => invoke<void>("set_muted", { muted }),
  setOutputGain: (db: number) => invoke<void>("set_output_gain", { db }),
  setMonitorGain: (db: number) => invoke<void>("set_monitor_gain", { db }),
  engineLog: (lines = 50) => invoke<string[]>("engine_log", { lines }),
  onEvent: (cb: (e: RuntimeEvent) => void): Promise<UnlistenFn> =>
    listen<RuntimeEvent>("resona", (e) => cb(e.payload)),
};
