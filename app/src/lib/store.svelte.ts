// App state (Svelte 5 runes). One instance, imported by every component.
import {
  api,
  type CalibrationResult,
  type DeviceInfo,
  type EngineState,
  type ChainSpec,
  type ImportReport,
  type KindSpec,
  type LibraryVoice,
  type PlayerState,
  type SelfTestResult,
  type Stats,
  type Take,
} from "./api";
import { DEFAULT_PROFILE, DEFAULT_SETTINGS, fitCrossfade, mergeDefaults, type Profile, type Settings } from "./settings";

export const TRACE_FRAMES = 500; // 5 s at 100 fps

export type ChainName = "input" | "output";
export type FxTarget = { chain: ChainName; id: number } | { special: "gate" | "limiter" } | null;

export interface Notice {
  id: number;
  kind: "info" | "error";
  text: string;
  action?: { label: string; run: () => void };
}

const AUDIO_EXT = /\.(wav|mp3|flac)$/i;

/** "2026-10-25 14-03-11", local time, safe in file names. */
function stamp(d = new Date()) {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

function debounce<A extends unknown[]>(fn: (...a: A) => void, ms: number) {
  let t: ReturnType<typeof setTimeout> | undefined;
  return (...a: A) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

class Store {
  ready = $state(false);
  settings = $state<Settings>(structuredClone(DEFAULT_SETTINGS));
  inputs = $state<DeviceInfo[]>([]);
  outputs = $state<DeviceInfo[]>([]);
  library = $state<LibraryVoice[]>([]);
  selectedId = $state<string | null>(null);
  profile = $state<Profile>({ ...DEFAULT_PROFILE });
  engine = $state<{ state: EngineState; message: string | null }>({ state: "stopped", message: null });
  stats = $state<Stats | null>(null);
  meters = $state({ inPeak: 0, inRms: 0, outPeak: 0, outRms: 0 });
  muted = $state(false);
  importing = $state<{ done: number; total: number; name: string | null } | null>(null);
  report = $state<ImportReport | null>(null);
  calibrating = $state(false);
  calibration = $state<CalibrationResult | null>(null);
  assets = $state<{ what: string; done: number; total: number } | null>(null);
  notices = $state<Notice[]>([]);
  settingsOpen = $state(false);
  settingsTab = $state<"devices" | "performance" | "pipeline" | "hotkeys" | "app">("devices");
  hotkeyErrors = $state<{ accel: string; action: string; error: string }[]>([]);
  logTail = $state<string[]>([]);
  catalog = $state<KindSpec[]>([]);
  fxEditing = $state<FxTarget>(null);
  /** Recent underrun count (last ~10 s), for the dropout warning. */
  recentUnderruns = $state(0);
  /** Sound file replacing the mic (null = microphone). */
  player = $state<PlayerState | null>(null);
  /** Seconds into the current A/B take, null when not recording. */
  recording = $state<number | null>(null);
  selftest = $state<{ running: boolean; result: SelfTestResult | null; error: string | null }>({
    running: false,
    result: null,
    error: null,
  });
  private playerPoll: ReturnType<typeof setInterval> | undefined;
  private recTimer: ReturnType<typeof setInterval> | undefined;

  /** Pitch history (Hz, 0 = unvoiced). Not reactive; `traceVersion` ticks on update. */
  traceIn = new Float32Array(TRACE_FRAMES);
  traceOut = new Float32Array(TRACE_FRAMES);
  traceVersion = $state(0);

  private underrunLog: { t: number; n: number }[] = [];
  private noticeId = 0;

  selected = $derived(this.library.find((v) => v.id === this.selectedId) ?? null);
  /** A session exists (anything but stopped/failed). */
  live = $derived(!["stopped", "failed"].includes(this.engine.state));
  running = $derived(this.engine.state === "running");
  busy = $derived(this.engine.state === "starting" || this.engine.state === "loading" || this.calibrating);
  activeVoiceId = $state<string | null>(null);
  ordered = $derived(
    [...this.library].sort((a, b) => Number(b.favorite) - Number(a.favorite) || a.name.localeCompare(b.name)),
  );

  // -- boot -----------------------------------------------------------------------------------
  async init() {
    const stored = await api.settingsLoad().catch(() => ({}));
    this.settings = mergeDefaults(DEFAULT_SETTINGS, stored);
    this.applyTheme();
    await Promise.all([this.refreshDevices(), this.refreshLibrary()]);
    this.muted = await api.getMuted().catch(() => false);
    this.catalog = await api.fxCatalog().catch(() => []);
    const ps = await api.playerState().catch(() => null);
    this.player = ps?.name ? ps : null;
    if (this.player?.playing) this.playerControl({}); // resume position polling
    const st = await api.status().catch(() => null);
    if (st) this.engine = { state: st.state, message: st.message };

    api.onEvent((e) => {
      switch (e.type) {
        case "state":
          this.engine = { state: e.state, message: e.message };
          if (e.state === "failed") this.loadLog();
          if (e.state === "stopped") this.activeVoiceId = null;
          break;
        case "stats":
          this.stats = e;
          this.trackUnderruns(e.underruns ?? 0);
          break;
        case "meters":
          this.meters = { inPeak: e.input_peak, inRms: e.input_rms, outPeak: e.output_peak, outRms: e.output_rms };
          break;
        case "pitch":
          this.pushTrace(e.input, e.output);
          break;
        case "progress":
          this.assets = e.done >= e.total ? null : { what: e.what, done: e.done, total: e.total };
          break;
        case "take":
          // The session stopped mid-take; the runtime finished it.
          this.recordingEnded(e);
          break;
      }
    });
    api.onImport((p) => (this.importing = p.name === null ? null : p));
    api.onMuted((m) => (this.muted = m));
    api.onHotkey((a) => this.onHotkey(a));

    const last = this.settings.last_voice;
    const first = this.library.find((v) => v.id === last) ?? this.ordered[0] ?? null;
    if (first) await this.selectIdle(first.id);
    await this.applyShell();
    await this.pushFxNow();
    this.ready = true;
    if (this.settings.start_on_launch && this.selected) this.start();
  }

  async refreshDevices() {
    try {
      const d = await api.listDevices();
      this.inputs = d.inputs;
      this.outputs = d.outputs;
    } catch (e) {
      this.notify("error", `Could not list audio devices: ${e}`);
    }
  }

  async refreshLibrary() {
    this.library = await api.libraryList().catch(() => []);
  }

  applyTheme() {
    document.documentElement.dataset.theme = this.settings.theme;
  }

  // -- persistence ----------------------------------------------------------------------------
  private persistSettings = debounce(() => {
    api.settingsSave($state.snapshot(this.settings)).catch((e) => this.notify("error", `Settings not saved: ${e}`));
  }, 300);

  saveSettings() {
    this.persistSettings();
  }

  private persistProfile = debounce((id: string, p: Profile) => {
    api.profileSave(id, p).catch(() => {});
  }, 400);

  private pendingPatch: Record<string, unknown> = {};

  private flushParams = debounce(() => {
    const patch = this.pendingPatch;
    this.pendingPatch = {};
    if (!this.live) return;
    api.setParams(patch).catch(() => {
      // The engine answers RPCs one at a time, so this can time out behind a voice load. Keep
      // the values (newer edits win) and try again, or the UI shows a pitch the engine never got.
      this.pendingPatch = { ...patch, ...this.pendingPatch };
      setTimeout(this.retryParams, 500);
    });
  }, 30);

  private retryParams = () => {
    if (!this.live) return;
    if (this.running) this.flushParams();
    else setTimeout(this.retryParams, 500); // engine restarting: wait until it is back
  };

  setParam<K extends keyof Profile>(key: K, value: Profile[K]) {
    this.profile[key] = value;
    if (this.selectedId) this.persistProfile(this.selectedId, $state.snapshot(this.profile));
    this.pendingPatch[key] = value;
    this.flushParams();
  }

  /** Global engine params (F0 post) changed in settings: push to a running session. */
  setF0<K extends keyof Settings["f0"]>(key: K, value: Settings["f0"][K]) {
    this.settings.f0[key] = value;
    this.saveSettings();
    this.pendingPatch[key] = value;
    this.flushParams();
  }

  private async loadProfile(id: string): Promise<Profile> {
    const stored = await api.profileLoad(id).catch(() => null);
    return mergeDefaults(DEFAULT_PROFILE, stored);
  }

  /** Engine params for a session: profile + global F0 post settings. */
  private params(p: Profile) {
    const { schema: _s, chain: _c, ...rest } = p;
    return { ...rest, ...this.settings.f0 };
  }

  // -- effects --------------------------------------------------------------------------------
  /** The output chain in force: the selected voice's own chain, else the global one. */
  outputChain = $derived(this.profile.chain ?? this.settings.fx.output);
  voiceHasChain = $derived(this.profile.chain !== null);

  chain(name: ChainName): ChainSpec {
    return name === "input" ? this.settings.fx.input : this.outputChain;
  }

  kind(kind: string) {
    return this.catalog.find((k) => k.kind === kind);
  }

  private async pushFxNow() {
    const fx = { ...$state.snapshot(this.settings.fx), output: $state.snapshot(this.outputChain) };
    await api.fxSet(fx).catch((e) => this.notify("error", `Effects not applied: ${e}`));
  }

  private pushFx = debounce(() => this.pushFxNow(), 40);

  /** Persist whichever object owns the edited chain. */
  private persistChain(name: ChainName) {
    if (name === "output" && this.profile.chain && this.selectedId) {
      this.persistProfile(this.selectedId, $state.snapshot(this.profile));
    } else {
      this.saveSettings();
    }
  }

  private editChain(name: ChainName, f: (c: ChainSpec) => void) {
    const c = name === "input" ? this.settings.fx.input : (this.profile.chain ?? this.settings.fx.output);
    f(c);
    this.persistChain(name);
    this.pushFx();
  }

  addSlot(name: ChainName, kind: string) {
    const k = this.kind(kind);
    if (!k) return;
    const ids = [...this.settings.fx.input.slots, ...this.outputChain.slots].map((s) => s.id);
    const id = Math.max(0, ...ids) + 1;
    const params = Object.fromEntries(k.params.map((p) => [p.key, p.default]));
    this.editChain(name, (c) => c.slots.push({ id, kind, enabled: true, params }));
    this.fxEditing = { chain: name, id };
  }

  removeSlot(name: ChainName, id: number) {
    this.editChain(name, (c) => (c.slots = c.slots.filter((s) => s.id !== id)));
    if (this.fxEditing && "id" in this.fxEditing && this.fxEditing.id === id) this.fxEditing = null;
  }

  toggleSlot(name: ChainName, id: number) {
    this.editChain(name, (c) => {
      const s = c.slots.find((x) => x.id === id);
      if (s) s.enabled = !s.enabled;
    });
  }

  moveSlot(name: ChainName, from: number, to: number) {
    if (from === to) return;
    this.editChain(name, (c) => {
      const [s] = c.slots.splice(from, 1);
      c.slots.splice(to, 0, s);
    });
  }

  setSlotParam(name: ChainName, id: number, key: string, value: number) {
    const c = name === "input" ? this.settings.fx.input : (this.profile.chain ?? this.settings.fx.output);
    const s = c.slots.find((x) => x.id === id);
    if (!s) return;
    s.params[key] = value;
    this.persistChain(name);
    api.fxParam(name, id, key, value).catch(() => {});
  }

  setGate(patch: Partial<Settings["fx"]["gate"]>) {
    Object.assign(this.settings.fx.gate, patch);
    this.saveSettings();
    this.pushFx();
  }

  setLimiter(patch: Partial<Settings["fx"]["limiter"]>) {
    Object.assign(this.settings.fx.limiter, patch);
    this.saveSettings();
    this.pushFx();
  }

  setSinging(on: boolean) {
    this.settings.fx.singing = on;
    this.saveSettings();
    this.pushFx();
  }

  /** Give the selected voice its own output chain (starting from the global one), or drop it. */
  setVoiceChain(own: boolean) {
    this.profile.chain = own ? structuredClone($state.snapshot(this.settings.fx.output)) : null;
    if (this.selectedId) this.persistProfile(this.selectedId, $state.snapshot(this.profile));
    this.pushFx();
  }

  replaceChain(name: ChainName, spec: ChainSpec) {
    this.editChain(name, (c) => (c.slots = spec.slots));
  }

  // -- voices ---------------------------------------------------------------------------------
  private async selectIdle(id: string) {
    this.selectedId = id;
    this.profile = await this.loadProfile(id);
  }

  /** Select a voice; while a session runs this switches seamlessly (crossfade, no restart). */
  async select(id: string) {
    if (id === this.selectedId && (!this.live || id === this.activeVoiceId)) return;
    const v = this.library.find((x) => x.id === id);
    if (!v) return;
    const profile = await this.loadProfile(id);
    this.selectedId = id;
    this.profile = profile;
    this.settings.last_voice = id;
    this.saveSettings();
    this.pushFx();
    if (this.live) {
      try {
        await api.activateVoice(this.voiceRef(v), this.params(profile), this.settings.xfade_ms);
        this.activeVoiceId = id;
      } catch (e) {
        this.notify("error", `${v.name}: ${e}`);
      }
    }
  }

  step(dir: 1 | -1) {
    const list = this.ordered;
    if (!list.length) return;
    const i = list.findIndex((v) => v.id === this.selectedId);
    const next = list[(i + dir + list.length) % list.length];
    this.select(next.id);
  }

  private voiceRef(v: LibraryVoice) {
    return { id: v.id, path: v.path, index_path: v.index_path };
  }

  async toggleFavorite(v: LibraryVoice) {
    const u = await api.libraryUpdate(v.id, { favorite: !v.favorite });
    this.library = this.library.map((x) => (x.id === v.id ? { ...x, favorite: u.favorite } : x));
  }

  async setVoiceHotkey(v: LibraryVoice, accel: string | null) {
    await api.libraryUpdate(v.id, { hotkey: accel });
    this.library = this.library.map((x) => (x.id === v.id ? { ...x, hotkey: accel } : x));
    await this.applyHotkeys();
  }

  async remove(v: LibraryVoice) {
    if (this.live && v.id === this.activeVoiceId) {
      this.notify("error", "Switch to another voice or stop before removing the active one.");
      return;
    }
    await api.libraryRemove(v.id);
    await this.refreshLibrary();
    if (this.selectedId === v.id) {
      const first = this.ordered[0];
      if (first) await this.selectIdle(first.id);
      else this.selectedId = null;
    }
    await this.applyHotkeys();
  }

  async importPaths(paths: string[]) {
    if (!paths.length || this.importing) return;
    this.importing = { done: 0, total: 0, name: null };
    this.report = null;
    try {
      const rep = await api.libraryImport(paths);
      await this.refreshLibrary();
      this.report = rep;
      if (!this.selectedId && rep.added[0]) await this.selectIdle(rep.added[0].id);
    } catch (e) {
      this.notify("error", `Import failed: ${e}`);
    } finally {
      this.importing = null;
    }
  }

  // -- session --------------------------------------------------------------------------------
  private blockMs(): number | null {
    return this.settings.block_ms ?? this.settings.calibration?.block_ms ?? null;
  }

  async calibrate(): Promise<number | null> {
    const v = this.selected;
    if (!v || this.calibrating) return null;
    this.calibrating = true;
    try {
      const res = await api.calibrate(this.voiceRef(v), this.streamSettings());
      this.calibration = res;
      this.settings.calibration = { block_ms: res.recommended, gpu: res.gpu, at: Date.now() };
      this.saveSettings();
      return res.recommended;
    } catch (e) {
      this.notify("error", `Calibration failed: ${e}`);
      return null;
    } finally {
      this.calibrating = false;
    }
  }

  private streamSettings() {
    const s = { ...this.settings.stream };
    fitCrossfade(s); // settings saved before the drawer clamped it
    return s;
  }

  async start() {
    const v = this.selected;
    if (!v || this.live || this.busy) return;
    let block = this.blockMs();
    if (block === null) block = (await this.calibrate()) ?? 60;
    const s = this.settings;
    try {
      await api.start({
        input: { device_id: s.input_device, mode: s.input_mode },
        output: { device_id: s.output_device, mode: s.output_mode },
        monitor: s.monitor_enabled ? { device_id: s.monitor_device, mode: s.monitor_mode } : null,
        stream: { ...this.streamSettings(), block_ms: block },
        voice: this.voiceRef(v),
        params: this.params(this.profile),
        jitter_ms: s.jitter_ms,
        down_mode: s.down_mode,
      });
      this.activeVoiceId = v.id;
      api.setOutputGain(s.output_gain_db);
      api.setMonitorGain(s.monitor_gain_db);
      this.traceIn.fill(0);
      this.traceOut.fill(0);
      setTimeout(() => this.preloadFavorites(), 3000);
      this.underrunLog = [];
      this.recentUnderruns = 0;
    } catch (e) {
      // The state event carries the message; keep a notice only if no state arrived.
      if (this.engine.state !== "failed") this.notify("error", String(e));
    }
  }

  /** Keep up to two favorites resident in VRAM so hotkey switches between them are instant. */
  private async preloadFavorites() {
    const favs = this.ordered.filter((v) => v.favorite && v.id !== this.activeVoiceId).slice(0, 2);
    for (const v of favs) {
      if (!this.live) return;
      await api.preloadVoice(this.voiceRef(v)).catch(() => {});
    }
  }

  async stop() {
    await api.stop().catch(() => {});
    this.activeVoiceId = null;
  }

  toggleMute() {
    api.setMuted(!this.muted);
  }

  async loadLog() {
    this.logTail = await api.engineLog(40).catch(() => []);
  }

  // -- live data ------------------------------------------------------------------------------
  private pushTrace(inp: number[], out: number[]) {
    const n = Math.min(inp.length, TRACE_FRAMES);
    this.traceIn.copyWithin(0, n);
    this.traceOut.copyWithin(0, n);
    this.traceIn.set(inp.slice(-n), TRACE_FRAMES - n);
    this.traceOut.set(out.slice(-n), TRACE_FRAMES - n);
    this.traceVersion++;
  }

  private trackUnderruns(total: number) {
    const now = performance.now();
    this.underrunLog.push({ t: now, n: total });
    while (this.underrunLog.length > 1 && now - this.underrunLog[0].t > 10_000) this.underrunLog.shift();
    this.recentUnderruns = Math.max(0, total - this.underrunLog[0].n);
  }

  // -- shell ----------------------------------------------------------------------------------
  async applyHotkeys() {
    const h = this.settings.hotkeys;
    const b = [
      { accel: h.mute, action: "mute" },
      { accel: this.settings.ptt_enabled ? h.ptt : "", action: "ptt" },
      { accel: h.next, action: "next" },
      { accel: h.prev, action: "prev" },
      ...this.library.filter((v) => v.hotkey).map((v) => ({ accel: v.hotkey!, action: `voice:${v.id}` })),
    ];
    this.hotkeyErrors = await api.applyHotkeys(b).catch(() => []);
  }

  async applyShell() {
    await this.applyHotkeys();
    // Push to talk means muted until the key is held, from the first second after a launch too.
    if (this.settings.ptt_enabled && !this.muted) await api.setMuted(true).catch(() => {});
    await api.setShellPrefs(this.settings.close_to_tray, this.settings.autostart).catch((e) =>
      this.notify("error", `Startup setting not applied: ${e}`),
    );
  }

  private onHotkey(action: string) {
    if (action === "next") this.step(1);
    else if (action === "prev") this.step(-1);
    else if (action.startsWith("voice:")) this.select(action.slice(6));
  }

  // -- sound-file input -----------------------------------------------------------------------
  async loadSoundFile(path: string) {
    try {
      this.player = await api.playerLoad(path);
      this.playerControl({ playing: true });
    } catch (e) {
      this.notify("error", String(e));
    }
  }

  async playerControl(c: { playing?: boolean; seekS?: number; looping?: boolean }) {
    if (!this.player) return;
    this.player = await api.playerControl(c).catch(() => this.player);
    clearInterval(this.playerPoll);
    if (this.player?.playing) {
      this.playerPoll = setInterval(async () => {
        const st = await api.playerState().catch(() => null);
        if (!st || !st.name) {
          clearInterval(this.playerPoll);
          return;
        }
        this.player = st;
        if (!st.playing) clearInterval(this.playerPoll);
      }, 200);
    }
  }

  async unloadSoundFile() {
    clearInterval(this.playerPoll);
    await api.playerUnload().catch(() => {});
    this.player = null;
  }

  /** Files dropped on the window: audio becomes the input source, the rest goes to the library. */
  drop(paths: string[]) {
    const audio = paths.filter((p) => AUDIO_EXT.test(p));
    const models = paths.filter((p) => /\.(pth|index|zip)$/i.test(p) || !/\.[a-z0-9]{1,5}$/i.test(p));
    if (audio.length) this.loadSoundFile(audio[0]);
    if (models.length) this.importPaths(models);
    if (!audio.length && !models.length) this.notify("error", "Drop voice models (.pth, .index, .zip, folders) or audio (.wav, .mp3, .flac).");
  }

  // -- A/B recorder ---------------------------------------------------------------------------
  async toggleRecording() {
    if (this.recording !== null) {
      clearInterval(this.recTimer);
      try {
        this.recordingEnded(await api.recordStop());
      } catch (e) {
        this.recording = null;
        this.notify("error", `Recording failed: ${e}`);
      }
      return;
    }
    if (!this.running) return;
    try {
      await api.recordStart(stamp());
      const t0 = performance.now();
      this.recording = 0;
      this.recTimer = setInterval(() => (this.recording = (performance.now() - t0) / 1000), 250);
    } catch (e) {
      this.notify("error", `Could not start recording: ${e}`);
    }
  }

  private recordingEnded(take: Take) {
    clearInterval(this.recTimer);
    this.recording = null;
    const secs = take.seconds < 60 ? `${take.seconds.toFixed(1)} s` : `${Math.floor(take.seconds / 60)}:${String(Math.round(take.seconds % 60)).padStart(2, "0")}`;
    this.notify("info", `Saved a ${secs} take (dry + wet) to Music\\Resona`, {
      label: "Show",
      run: () => api.reveal(take.wet).catch(() => {}),
    });
  }

  // -- latency self-test ----------------------------------------------------------------------
  async runSelfTest(listen: string | null) {
    if (!this.running || this.selftest.running) return;
    this.selftest = { running: true, result: null, error: null };
    try {
      this.selftest = { running: false, result: await api.latencySelftest(listen), error: null };
    } catch (e) {
      this.selftest = { running: false, result: null, error: String(e) };
    }
  }

  notify(kind: Notice["kind"], text: string, action?: Notice["action"]) {
    const id = ++this.noticeId;
    this.notices = [...this.notices, { id, kind, text, action }];
    setTimeout(() => this.dismiss(id), kind === "error" || action ? 9000 : 4000);
  }

  dismiss(id: number) {
    this.notices = this.notices.filter((n) => n.id !== id);
  }
}

export const store = new Store();
