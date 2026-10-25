// Dev-only stand-in for the Tauri shell so the UI can be previewed in a plain browser
// (`npm run dev`, then open http://localhost:5173/?state=running). Never bundled into the app:
// main.ts only imports it when there is no Tauri runtime. Data here is sample data.
import { mockIPC, mockWindows } from "@tauri-apps/api/mocks";
import { emit } from "@tauri-apps/api/event";
import catalog from "./devcatalog.json";

const params = new URLSearchParams(location.search);
const scenario = params.get("state") ?? "stopped";
const empty = params.has("empty");

const info = (name: string, sr: number, epoch: number) => ({
  path: "", name, sr, version: "v2", f0: true, vocoder: "HiFi-GAN", vocoder_architecture: null,
  embedder: "contentvec", hash: name.toLowerCase().replace(/[^a-z0-9]/g, "").slice(0, 16), epoch,
});
const voice = (name: string, sr: number, epoch: number, fav = false, index = true, hotkey: string | null = null) => {
  const i = info(name, sr, epoch);
  return {
    id: i.hash, name, model_file: `${name}.pth`, index_file: index ? `added_${name}.index` : null,
    path: `C:/voices/${name}.pth`, index_path: index ? `C:/voices/${name}.index` : null, source: `D:/models/${name}.pth`,
    info: i, favorite: fav, hotkey, added_at: 0,
  };
};
let library = empty
  ? []
  : [
      voice("DonaldTrump", 40000, 475, true, true, "Ctrl+Alt+1"),
      voice("AnniePants (asmr)", 32000, 860),
      voice("CyreneAidenDawnHSR", 48000, 900, true),
      voice("Irene", 40000, 300),
      voice("JinwooSung", 32000, 490, false, false),
      voice("supercreek", 40000, 250),
      voice("YandereASMR", 40000, 150),
      voice("egirl", 48000, 400),
    ];

const dev = (id: string, name: string, direction: string, is_default = false) => ({
  id, name, direction, is_default, mix_rate: 48000, mix_channels: 2,
});
const devices = {
  inputs: [dev("mic1", "Microphone (Shure MV7)", "capture", true), dev("mic2", "Headset Microphone (Arctis 7)", "capture")],
  outputs: [
    dev("spk", "Speakers (Realtek Audio)", "render", true),
    dev("line1", "Line 1 (Virtual Audio Cable)", "render"),
    dev("hp", "Headphones (Arctis 7)", "render"),
  ],
};

let state = "stopped";
let muted = false;
const setState = (s: string, message: string | null = null) => {
  state = s;
  emit("resona", { type: "state", state: s, message });
};

function startLive() {
  setState("running");
  let t = 0;
  let under = 0;
  setInterval(() => {
    t += 0.1;
    const speaking = Math.sin(t * 0.9) > -0.35;
    const base = 118 + 30 * Math.sin(t * 1.7) + 12 * Math.sin(t * 5.3);
    const inp = Array.from({ length: 10 }, (_, k) => (speaking && (k + Math.floor(t * 10)) % 37 > 3 ? base + k * 0.6 : 0));
    emit("resona", { type: "pitch", input: inp, output: inp.map((f) => (f ? f * 2 ** (5 / 12) : 0)) });
    if (scenario === "underruns" && Math.random() < 0.15) under++;
    emit("resona", {
      type: "stats", processed: Math.round(t * 25), dropped: 0, backlog: 0, p50_ms: 14.6 + Math.random(), p99_ms: 17.2 + Math.random(),
      vram_mb: 707, underruns: under, fifo_ms: 10, sola_ncc: 0.91,
      latency: { input_ms: 10, block_ms: 40, lookahead_ms: 40, compute_ms: 15, buffer_ms: 10, output_ms: 11, total_ms: 126 },
      devices: {
        input: { device_name: "Microphone (Shure MV7)", mode: "shared", latency_ms: 10 },
        output: { device_name: "Line 1 (Virtual Audio Cable)", mode: "shared", latency_ms: 10 },
        monitor: null,
      },
    });
  }, 100);
  setInterval(() => {
    const lvl = 0.05 + 0.25 * Math.abs(Math.sin(performance.now() / 300)) * (Math.random() * 0.6 + 0.4);
    emit("resona", { type: "meters", input_peak: lvl * 1.6, input_rms: lvl * 0.7, output_peak: lvl * 1.9, output_rms: lvl * 0.8 });
  }, 33);
}

export function install() {
  mockWindows("main");
  mockIPC(
    (cmd, args) => {
      const a = (args ?? {}) as Record<string, unknown>;
      switch (cmd) {
        case "list_devices": return devices;
        case "fx_catalog": return catalog;
        case "fx_set": case "fx_param": return null;
        case "get_status": return { state, message: null, engine_dir: "C:/Resona/engine" };
        case "settings_load":
          return {
            output_device: "line1", calibration: { block_ms: 40, gpu: "NVIDIA GeForce RTX 3060 Ti", at: 0 }, last_voice: library[0]?.id ?? null,
            ...(params.has("fx") ? { fx: {
              input: { slots: [
                { id: 1, kind: "gain", enabled: true, params: { gain: 3 } },
                { id: 2, kind: "filter", enabled: true, params: { mode: 1, freq: 70, q: 0.707, slope: 0 } },
                { id: 3, kind: "denoise", enabled: true, params: { amount: 1 } },
              ] },
              output: { slots: [
                { id: 4, kind: "eq", enabled: true, params: { f1: 120, g1: 3, f2: 400, g2: -2.5, q2: 1.2, f3: 1800, g3: 0, q3: 1, f4: 4500, g4: 4, q4: 1.4, f5: 11000, g5: 2 } },
                { id: 5, kind: "compressor", enabled: true, params: {} },
                { id: 6, kind: "reverb", enabled: false, params: {} },
              ] },
              gate: { enabled: true, threshold: 0.6, hold_ms: 200, release_ms: 300, range_db: -45 },
              limiter: { enabled: true, ceiling_db: -1 }, singing: false,
            } } : {}),
          };
        case "settings_save": case "profile_save": return null;
        case "profile_load": return { pitch: 5, formant: 0, index_rate: 0.5, protect: 0.33, chain: null };
        case "library_list": return library;
        case "library_update": {
          const v = library.find((x) => x.id === a.id)!;
          Object.assign(v, a.patch);
          return v;
        }
        case "library_remove": library = library.filter((x) => x.id !== a.id); return null;
        case "library_import":
          return new Promise((r) => setTimeout(() => r({ added: [], skipped: [], failed: [{ path: "D:/m/refine.pth", error: "refine.pth: unsupported vocoder RefineGAN" }] }), 800));
        case "get_muted": return muted;
        case "set_muted": muted = a.muted as boolean; emit("muted", { muted }); return null;
        case "apply_hotkeys": return [];
        case "set_shell_prefs": case "set_output_gain": case "set_monitor_gain": case "set_params": case "activate_voice": return null;
        case "launched_minimized": return false;
        case "engine_log": return ["[engine] feature extractors loaded in 2.1s on cuda", "Traceback (most recent call last):", "  RuntimeError: CUDA error: out of memory"];
        case "start_session":
          setState("starting");
          return new Promise((r) => setTimeout(() => { setState("loading", "loading voice"); setTimeout(() => { startLive(); r(null); }, 700); }, 500));
        case "stop_session": setState("stopped"); location.reload(); return null;
        case "calibrate":
          return new Promise((r) => setTimeout(() => r({ results: [40, 60, 80, 100].map((b) => ({ block_ms: b, p50_ms: 14.8, p99_ms: 17.1, ok: true })), recommended: 40, budget: 0.6, gpu: "NVIDIA GeForce RTX 3060 Ti" }), 1200));
        default: return null;
      }
    },
    { shouldMockEvents: true },
  );
  setTimeout(() => {
    if (scenario === "running" || scenario === "underruns") startLive();
    else if (scenario === "failed") setState("failed", "engine crashed 3 times in a minute; giving up");
    else if (scenario === "device_lost") setState("device_lost", "waiting for device: Line 1 (Virtual Audio Cable) not found");
    else if (scenario === "restarting") setState("restarting", "engine exited unexpectedly (attempt 1 of 3)");
    else if (scenario === "loading") setState("loading", "loading DonaldTrump");
  }, 400);
}
