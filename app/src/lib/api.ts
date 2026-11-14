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
  effects_ms?: number;
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
  gate_open?: boolean;
  /** GPU scheduling priority the engine reached ("high", "above normal", or "normal" if refused). */
  gpu_priority?: string;
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
  gpu: string;
}

export interface LibraryVoice {
  id: string;
  name: string;
  model_file: string;
  index_file: string | null;
  path: string;
  index_path: string | null;
  source: string;
  info: ModelInfo;
  favorite: boolean;
  hotkey: string | null;
  added_at: number;
  /** Other voices using the same index file; it belongs to at most one of them. */
  index_shared_with?: string[];
  /** This voice's name is in the index's file name: it is the voice the index was made for. */
  index_owner?: boolean;
}

export interface ImportReport {
  added: LibraryVoice[];
  skipped: { path: string; reason: string }[];
  failed: { path: string; error: string }[];
}

export interface Binding {
  accel: string;
  action: string;
}

export type ParamScale = "linear" | "log" | { choice: string[] };

export interface ParamSpec {
  key: string;
  name: string;
  min: number;
  max: number;
  default: number;
  unit: string;
  scale: ParamScale;
}

/** What only a filter (an effect loaded from a TOML file) has in its catalog entry. */
export interface FilterMeta {
  file: string;
  /** Ships with the app; editing it edits a copy in the user's folder. */
  builtin: boolean;
  /** Why the file could not be used. The filter passes audio through while this is set. */
  error: string | null;
  description: string;
  /** Stage types, in order. */
  stages: string[];
}

export interface KindSpec {
  kind: string;
  name: string;
  stage: "input" | "output" | "both";
  /** For a filter: its macros. */
  params: ParamSpec[];
  filter?: FilterMeta;
}

/** A chain slot holds a filter when its kind starts with this. */
export const FILTER_PREFIX = "filter:";

export interface SlotSpec {
  id: number;
  kind: string;
  enabled: boolean;
  params: Record<string, number>;
}

export interface ChainSpec {
  slots: SlotSpec[];
}

export interface GateParams {
  enabled: boolean;
  threshold: number;
  hold_ms: number;
  release_ms: number;
  range_db: number;
}

export interface FxSettings {
  input: ChainSpec;
  output: ChainSpec;
  gate: GateParams;
  limiter: { enabled: boolean; ceiling_db: number };
  singing: boolean;
}

export interface PlayerState {
  name: string | null;
  playing: boolean;
  looping: boolean;
  position_s: number;
  duration_s: number;
}

export interface Take {
  dry: string;
  wet: string;
  seconds: number;
}

export interface SelfTestResult {
  measured_ms: number;
  runs_ms: number[];
  spread_ms: number;
  estimate_ms: number | null;
  listen: string;
}

export type RuntimeEvent =
  | { type: "state"; state: EngineState; message: string | null }
  | ({ type: "stats" } & Stats)
  | { type: "meters"; input_peak: number; input_rms: number; output_peak: number; output_rms: number }
  | { type: "pitch"; input: number[]; output: number[] }
  | { type: "progress"; what: string; done: number; total: number }
  | { type: "log"; line: string }
  | { type: "filters"; catalog: KindSpec[] }
  | ({ type: "take" } & Take);

export const api = {
  listDevices: () => invoke<{ inputs: DeviceInfo[]; outputs: DeviceInfo[] }>("list_devices"),
  status: () => invoke<{ state: EngineState; message: string | null; engine_dir: string | null }>("get_status"),
  start: (cfg: SessionConfig) => invoke<void>("start_session", { cfg }),
  stop: () => invoke<void>("stop_session"),
  setParams: (params: Partial<LiveParams>) => invoke<LiveParams>("set_params", { params }),
  inspectModel: (path: string) => invoke<ModelInfo>("inspect_model", { path }),
  calibrate: (voice: VoiceRef, stream?: Partial<StreamSettings>) =>
    invoke<CalibrationResult>("calibrate", { voice, stream: stream ?? null }),
  activateVoice: (voice: VoiceRef, params?: Partial<LiveParams>, xfadeMs?: number) =>
    invoke<void>("activate_voice", { voice, params: params ?? null, xfadeMs: xfadeMs ?? null }),
  preloadVoice: (voice: VoiceRef) => invoke<ModelInfo>("preload_voice", { voice }),
  setMuted: (muted: boolean) => invoke<void>("set_muted", { muted }),
  getMuted: () => invoke<boolean>("get_muted"),
  setOutputGain: (db: number) => invoke<void>("set_output_gain", { db }),
  setMonitorGain: (db: number) => invoke<void>("set_monitor_gain", { db }),
  engineLog: (lines = 50) => invoke<string[]>("engine_log", { lines }),
  assetsStatus: () => invoke<{ root: string; missing: string[] }>("assets_status"),
  settingsLoad: () => invoke<Record<string, unknown>>("settings_load"),
  settingsSave: (settings: unknown) => invoke<void>("settings_save", { settings }),
  profileLoad: (hash: string) => invoke<Record<string, unknown> | null>("profile_load", { hash }),
  profileSave: (hash: string, profile: unknown) => invoke<void>("profile_save", { hash, profile }),
  libraryList: () => invoke<LibraryVoice[]>("library_list"),
  libraryUpdate: (id: string, patch: Partial<Pick<LibraryVoice, "favorite" | "hotkey" | "name">>) =>
    invoke<LibraryVoice>("library_update", { id, patch }),
  libraryRemove: (id: string) => invoke<void>("library_remove", { id }),
  librarySetIndex: (id: string, path: string | null) => invoke<unknown>("library_set_index", { id, path }),
  libraryImport: (paths: string[]) => invoke<ImportReport>("library_import", { paths }),
  applyHotkeys: (bindings: Binding[]) =>
    invoke<{ accel: string; action: string; error: string }[]>("apply_hotkeys", { bindings }),
  setShellPrefs: (closeToTray: boolean, autostart: boolean) =>
    invoke<void>("set_shell_prefs", { closeToTray, autostart }),
  launchedMinimized: () => invoke<boolean>("launched_minimized"),
  fxCatalog: () => invoke<KindSpec[]>("fx_catalog"),
  fxSet: (settings: FxSettings) => invoke<void>("fx_set", { settings }),
  fxParam: (chain: "input" | "output", slot: number, key: string, value: number) =>
    invoke<void>("fx_param", { chain, slot, key, value }),
  chainExport: (path: string, chain: ChainSpec) => invoke<void>("chain_export", { path, chain }),
  chainImport: (path: string) => invoke<ChainSpec>("chain_import", { path }),
  /** Create a filter file from the template and open it. Resolves to the kind it will have. */
  filterNew: (name: string) => invoke<string>("filter_new", { name }),
  /** Open a filter's file in its editor (a built-in one is copied to the user's folder first). */
  filterEdit: (kind: string) => invoke<string>("filter_edit", { kind }),
  filtersFolder: () => invoke<void>("filters_folder"),
  playerLoad: (path: string) => invoke<PlayerState>("player_load", { path }),
  playerUnload: () => invoke<void>("player_unload"),
  playerControl: (c: { playing?: boolean; seekS?: number; looping?: boolean }) =>
    invoke<PlayerState>("player_control", { playing: c.playing ?? null, seekS: c.seekS ?? null, looping: c.looping ?? null }),
  playerState: () => invoke<PlayerState>("player_state"),
  recordStart: (stem: string) => invoke<void>("record_start", { stem }),
  recordStop: () => invoke<Take>("record_stop"),
  reveal: (path: string | null) => invoke<void>("reveal", { path }),
  latencySelftest: (listen: string | null) => invoke<SelfTestResult>("latency_selftest", { listen }),
  onEvent: (cb: (e: RuntimeEvent) => void): Promise<UnlistenFn> =>
    listen<RuntimeEvent>("resona", (e) => cb(e.payload)),
  onImport: (cb: (p: { done: number; total: number; name: string | null }) => void) =>
    listen<{ done: number; total: number; name: string | null }>("import", (e) => cb(e.payload)),
  onMuted: (cb: (muted: boolean) => void) => listen<{ muted: boolean }>("muted", (e) => cb(e.payload.muted)),
  onHotkey: (cb: (action: string) => void) => listen<{ action: string }>("hotkey", (e) => cb(e.payload.action)),
};
