<script lang="ts">
  import X from "phosphor-svelte/lib/X";
  import ArrowClockwise from "phosphor-svelte/lib/ArrowClockwise";
  import { fly, fade } from "svelte/transition";
  import { cubicOut } from "svelte/easing";
  import Fader from "./Fader.svelte";
  import HotkeyInput from "./HotkeyInput.svelte";
  import { store } from "../lib/store.svelte";
  import { api, type DeviceMode } from "../lib/api";
  import type { Hotkeys, Settings } from "../lib/settings";

  const TABS = [
    ["devices", "Devices"],
    ["performance", "Performance"],
    ["pipeline", "Pipeline"],
    ["hotkeys", "Hotkeys"],
    ["app", "App"],
  ] as const;
  const MODES: [DeviceMode, string][] = [
    ["shared", "Shared"],
    ["low_latency", "Low latency"],
    ["exclusive", "Exclusive"],
  ];
  const reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  let s = $derived(store.settings);
  let restartNote = $derived(store.live);

  function save() {
    store.saveSettings();
  }
  function setS<K extends keyof Settings>(k: K, v: Settings[K]) {
    store.settings[k] = v;
    save();
  }
  function setStream(k: keyof Settings["stream"], v: number | string) {
    (store.settings.stream as Record<string, unknown>)[k] = v;
    save();
  }
  function setHotkey(k: keyof Hotkeys, accel: string) {
    store.settings.hotkeys[k] = accel;
    save();
    store.applyHotkeys();
  }
  const hkError = (action: string) => store.hotkeyErrors.find((e) => e.action === action)?.error ?? null;

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") store.settingsOpen = false;
  }

  // Engine constraint: crossfade + search/2 must fit in the lookahead.
  let stream = $derived(s.stream);
  let xfadeMax = $derived(Math.max(0, (stream.lookahead_ms ?? 40) - (stream.sola_search_ms ?? 10) / 2));
  let cal = $derived(s.calibration);
  let listen = $state("");
  let st = $derived(store.selftest);
</script>

<svelte:window onkeydown={onKey} />

<div class="scrim" transition:fade={{ duration: reduce ? 0 : 140 }} onclick={() => (store.settingsOpen = false)} aria-hidden="true"></div>
<div
  class="drawer"
  role="dialog"
  aria-modal="true"
  aria-label="Settings"
  transition:fly={{ x: 40, duration: reduce ? 0 : 220, easing: cubicOut }}
>
  <header>
    <h2>Settings</h2>
    <button class="btn ghost icon" aria-label="Close settings" onclick={() => (store.settingsOpen = false)}><X size={16} /></button>
  </header>
  <nav class="tabs" aria-label="Settings sections">
    {#each TABS as [id, name] (id)}
      <button aria-pressed={store.settingsTab === id} onclick={() => (store.settingsTab = id)}>{name}</button>
    {/each}
  </nav>

  <div class="content">
    {#if store.settingsTab === "devices"}
      {#if restartNote}<p class="note">Device changes apply the next time you start.</p>{/if}
      <section>
        <div class="field">
          <label class="label" for="in-dev">Microphone</label>
          <select id="in-dev" class="input" value={s.input_device ?? ""} onchange={(e) => setS("input_device", e.currentTarget.value || null)}>
            <option value="">System default</option>
            {#each store.inputs as d (d.id)}<option value={d.id}>{d.name}</option>{/each}
          </select>
          <div class="seg" role="group" aria-label="Microphone mode">
            {#each MODES as [m, n] (m)}
              <button aria-pressed={s.input_mode === m} onclick={() => setS("input_mode", m)}>{n}</button>
            {/each}
          </div>
        </div>
        <div class="field">
          <label class="label" for="out-dev">Output</label>
          <select id="out-dev" class="input" value={s.output_device ?? ""} onchange={(e) => setS("output_device", e.currentTarget.value || null)}>
            <option value="">System default</option>
            {#each store.outputs as d (d.id)}<option value={d.id}>{d.name}</option>{/each}
          </select>
          <div class="seg" role="group" aria-label="Output mode">
            {#each MODES as [m, n] (m)}
              <button aria-pressed={s.output_mode === m} onclick={() => setS("output_mode", m)}>{n}</button>
            {/each}
          </div>
          <span class="hint">Pick a virtual cable (VB-Cable, VAC) here and select it as the mic in Discord or OBS. Low latency needs a driver that supports small periods; Exclusive locks the device.</span>
        </div>
        <button class="btn ghost refresh" onclick={() => store.refreshDevices()}><ArrowClockwise size={14} /> Refresh devices</button>
      </section>

      <section>
        <div class="row">
          <div>
            <span class="title">Monitor</span>
            <span class="hint">Hear yourself on a second device, such as headphones.</span>
          </div>
          <button class="switch" role="switch" aria-checked={s.monitor_enabled} aria-label="Monitor" onclick={() => setS("monitor_enabled", !s.monitor_enabled)}></button>
        </div>
        {#if s.monitor_enabled}
          <div class="field">
            <label class="label" for="mon-dev">Monitor device</label>
            <select id="mon-dev" class="input" value={s.monitor_device ?? ""} onchange={(e) => setS("monitor_device", e.currentTarget.value || null)}>
              <option value="">System default</option>
              {#each store.outputs as d (d.id)}<option value={d.id}>{d.name}</option>{/each}
            </select>
          </div>
          <div class="field">
            <span class="label">Monitor level <span class="num val">{s.monitor_gain_db} dB</span></span>
            <Fader label="Monitor level" wheel={false} value={s.monitor_gain_db} min={-30} max={6} step={1} defaultValue={0} origin={0}
              onchange={(v) => { setS("monitor_gain_db", v); api.setMonitorGain(v); }} />
          </div>
        {/if}
      </section>

      <section>
        <div class="field">
          <span class="label">While the engine is down</span>
          <div class="seg" role="group" aria-label="While the engine is down">
            <button aria-pressed={s.down_mode === "mute"} onclick={() => setS("down_mode", "mute")}>Silence</button>
            <button aria-pressed={s.down_mode === "passthrough"} onclick={() => setS("down_mode", "passthrough")}>Pass my real voice</button>
          </div>
        </div>
      </section>
    {:else if store.settingsTab === "performance"}
      <section>
        <div class="field">
          <span class="label">Block size</span>
          <div class="seg" role="group" aria-label="Block size">
            <button aria-pressed={s.block_ms === null} onclick={() => setS("block_ms", null)}>Auto</button>
            {#each [40, 60, 80, 100] as b (b)}
              <button class="num" aria-pressed={s.block_ms === b} onclick={() => setS("block_ms", b)}>{b}</button>
            {/each}
          </div>
          <span class="hint">
            {#if cal}Auto uses {cal.block_ms} ms, measured on {cal.gpu}.{:else}Auto measures your GPU on first start.{/if}
            Smaller blocks cut latency; larger ones survive a busy GPU.
          </span>
        </div>
        <div class="row">
          <button class="btn" disabled={store.live || store.calibrating || !store.selected} onclick={() => store.calibrate()}>
            {store.calibrating ? "Measuring" : cal ? "Measure again" : "Measure now"}
          </button>
          {#if store.live}<span class="hint">Stop the voice to measure.</span>{/if}
        </div>
        {#if store.calibration}
          <table class="cal">
            <thead><tr><th>Block</th><th>p50</th><th>p99</th><th>Fits</th></tr></thead>
            <tbody>
              {#each store.calibration.results as r (r.block_ms)}
                <tr class:rec={r.block_ms === store.calibration.recommended}>
                  <td class="num">{r.block_ms} ms</td>
                  <td class="num">{r.p50_ms.toFixed(1)}</td>
                  <td class="num">{r.p99_ms.toFixed(1)}</td>
                  <td>{r.ok ? "Yes" : "No"}</td>
                </tr>
              {/each}
            </tbody>
          </table>
          <span class="hint">A block fits when its slowest 1 % stays under 60 % of the block length.</span>
        {/if}
      </section>
      <section>
        <div class="head">
          <span class="title">Latency self-test</span>
          <span class="hint">Sends five short test tones through the whole chain and times them on the way out. Your mic is muted for about 7 seconds.</span>
        </div>
        <div class="field">
          <label class="label" for="st-listen">Listen on</label>
          <select id="st-listen" class="input" bind:value={listen}>
            <option value="">The output device (loopback)</option>
            {#each store.inputs as d (d.id)}<option value={d.id}>{d.name}</option>{/each}
          </select>
          <span class="hint">To include the virtual cable's own buffering, pick its recording side (for VAC, the "Line" input).</span>
        </div>
        <div class="row">
          <button class="btn" disabled={!store.running || store.selftest.running} onclick={() => store.runSelfTest(listen || null)}>
            {store.selftest.running ? "Testing" : st.result ? "Test again" : "Run test"}
          </button>
          {#if !store.running}<span class="hint">Start the voice changer first.</span>{/if}
        </div>
        {#if store.selftest.running}
          <div class="st-progress" aria-hidden="true"><span></span></div>
        {:else if st.error}
          <p class="st-error">{st.error}</p>
        {:else if st.result}
          {@const r = st.result}
          {@const lo = Math.min(...r.runs_ms, r.estimate_ms ?? Infinity) - 10}
          {@const hi = Math.max(...r.runs_ms, r.estimate_ms ?? 0) + 10}
          {@const pos = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`}
          <div class="st-result">
            <div class="st-figs">
              <div><span class="num st-big">{Math.round(r.measured_ms)}</span><span class="st-unit">ms measured</span></div>
              {#if r.estimate_ms !== null}
                <div class="st-est"><span class="num">{Math.round(r.estimate_ms)} ms</span> estimated · <span class="num">{r.measured_ms - r.estimate_ms >= 0 ? "+" : ""}{Math.round(r.measured_ms - r.estimate_ms)}</span></div>
              {/if}
            </div>
            <div class="st-scale" role="img" aria-label="Five measurements between {Math.round(Math.min(...r.runs_ms))} and {Math.round(Math.max(...r.runs_ms))} ms">
              {#if r.estimate_ms !== null}<span class="st-mark est" style:left={pos(r.estimate_ms)} title="Estimate"></span>{/if}
              {#each r.runs_ms as v, i (i)}<span class="st-mark run" style:left={pos(v)} title="{v.toFixed(1)} ms"></span>{/each}
              <span class="st-mark med" style:left={pos(r.measured_ms)}></span>
            </div>
            <span class="hint">Spread <span class="num">{r.spread_ms.toFixed(1)} ms</span> across 5 tones, heard on {r.listen}.</span>
          </div>
        {/if}
      </section>
      <section>
        <div class="field">
          <span class="label">Output buffer <span class="num val">{s.jitter_ms} ms</span></span>
          <Fader label="Output buffer" wheel={false} value={s.jitter_ms} min={5} max={40} step={1} defaultValue={10} onchange={(v) => setS("jitter_ms", v)} />
          <span class="hint">Starting point only. Resona adds buffer after a dropout and gives it back once playback is stable.</span>
        </div>
        <div class="field">
          <span class="label">Voice switch crossfade <span class="num val">{s.xfade_ms} ms</span></span>
          <Fader label="Voice switch crossfade" wheel={false} value={s.xfade_ms} min={0} max={200} step={10} defaultValue={50} onchange={(v) => setS("xfade_ms", v)} />
        </div>
      </section>
    {:else if store.settingsTab === "pipeline"}
      {#if restartNote}<p class="note">Window and pitch-tracker changes apply on next start. F0 smoothing applies live.</p>{/if}
      <section>
        <div class="field">
          <span class="label">Pitch tracker</span>
          <div class="seg" role="group" aria-label="Pitch tracker">
            <button aria-pressed={stream.f0_method !== "fcpe"} onclick={() => setStream("f0_method", "rmvpe")}>RMVPE</button>
            <button aria-pressed={stream.f0_method === "fcpe"} onclick={() => setStream("f0_method", "fcpe")}>FCPE</button>
          </div>
          <span class="hint">RMVPE is the most accurate. FCPE is lighter on the GPU.</span>
        </div>
        {#each [
          ["context_ms", "Context", 400, 2000, 100, 1000, "How much past audio each block sees. More is steadier and costs GPU."],
          ["lookahead_ms", "Lookahead", 20, 200, 20, 40, "Future audio per block. The biggest quality lever, added straight to latency."],
          ["sola_search_ms", "Seam search", 0, 20, 10, 10, "How far block seams may shift to line up."],
          ["edge_audio_ms", "Edge mirror", 0, 80, 20, 40, "Mirrored padding that steadies the newest audio."],
        ] as const as [k, name, min, max, step, def, hint] (k)}
          <div class="field">
            <span class="label">{name} <span class="num val">{stream[k] ?? def} ms</span></span>
            <Fader label={name} wheel={false} value={Number(stream[k] ?? def)} {min} {max} {step} defaultValue={def} onchange={(v) => setStream(k, v)} />
            <span class="hint">{hint}</span>
          </div>
        {/each}
        <div class="field">
          <span class="label">Seam crossfade <span class="num val">{stream.crossfade_ms ?? 10} ms</span></span>
          <Fader label="Seam crossfade" wheel={false} value={Math.min(stream.crossfade_ms ?? 10, xfadeMax)} min={0} max={Math.max(10, Math.floor(xfadeMax / 10) * 10)} step={10} defaultValue={10}
            onchange={(v) => setStream("crossfade_ms", v)} />
          <span class="hint">Crossfade plus half the seam search must fit inside the lookahead.</span>
        </div>
      </section>
      <section>
        <div class="field">
          <span class="label">Pitch smoothing <span class="num val">{s.f0.f0_median} frames</span></span>
          <Fader label="Pitch smoothing" wheel={false} value={s.f0.f0_median} min={1} max={11} step={2} defaultValue={5} onchange={(v) => store.setF0("f0_median", v)} />
        </div>
        <div class="field">
          <span class="label">Glide <span class="num val">{s.f0.f0_glide_ms} ms</span></span>
          <Fader label="Glide" wheel={false} value={s.f0.f0_glide_ms} min={0} max={120} step={5} defaultValue={0} onchange={(v) => store.setF0("f0_glide_ms", v)} />
        </div>
        <div class="field">
          <span class="label">Voicing threshold <span class="num val">{s.f0.f0_threshold.toFixed(3)}</span></span>
          <Fader label="Voicing threshold" wheel={false} value={s.f0.f0_threshold} min={0.005} max={0.1} step={0.005} defaultValue={0.03} onchange={(v) => store.setF0("f0_threshold", v)} />
        </div>
        <div class="row">
          <div>
            <span class="title">Octave correction</span>
            <span class="hint">Catches the tracker jumping an octave.</span>
          </div>
          <button class="switch" role="switch" aria-label="Octave correction" aria-checked={s.f0.f0_octave_fix} onclick={() => store.setF0("f0_octave_fix", !s.f0.f0_octave_fix)}></button>
        </div>
      </section>
    {:else if store.settingsTab === "hotkeys"}
      <section>
        <p class="hint">Hotkeys work while Resona is in the background. Set a voice's own hotkey from its menu in the library.</p>
        {#each [
          ["mute", "Mute toggle"],
          ["next", "Next voice"],
          ["prev", "Previous voice"],
        ] as const as [k, name] (k)}
          <div class="row">
            <span class="title">{name}</span>
            <div><HotkeyInput value={s.hotkeys[k]} error={hkError(k)} onchange={(a) => setHotkey(k, a)} /></div>
          </div>
        {/each}
        <div class="row">
          <div>
            <span class="title">Push to talk</span>
            <span class="hint">Muted unless the key is held.</span>
          </div>
          <button class="switch" role="switch" aria-label="Push to talk" aria-checked={s.ptt_enabled}
            onclick={() => { setS("ptt_enabled", !s.ptt_enabled); store.applyHotkeys(); if (s.ptt_enabled) api.setMuted(true); }}></button>
        </div>
        {#if s.ptt_enabled}
          <div class="row">
            <span class="title">Talk key</span>
            <div><HotkeyInput value={s.hotkeys.ptt} error={hkError("ptt")} onchange={(a) => setHotkey("ptt", a)} /></div>
          </div>
        {/if}
      </section>
    {:else}
      <section>
        <div class="row">
          <div>
            <span class="title">Keep running in the tray</span>
            <span class="hint">Closing the window hides it; quit from the tray icon.</span>
          </div>
          <button class="switch" role="switch" aria-label="Keep running in the tray" aria-checked={s.close_to_tray}
            onclick={() => { setS("close_to_tray", !s.close_to_tray); store.applyShell(); }}></button>
        </div>
        <div class="row">
          <div>
            <span class="title">Start with Windows</span>
            <span class="hint">Starts hidden in the tray.</span>
          </div>
          <button class="switch" role="switch" aria-label="Start with Windows" aria-checked={s.autostart}
            onclick={() => { setS("autostart", !s.autostart); store.applyShell(); }}></button>
        </div>
        <div class="row">
          <div>
            <span class="title">Go live on launch</span>
            <span class="hint">Starts the last voice as soon as Resona opens.</span>
          </div>
          <button class="switch" role="switch" aria-label="Go live on launch" aria-checked={s.start_on_launch} onclick={() => setS("start_on_launch", !s.start_on_launch)}></button>
        </div>
        <div class="field">
          <span class="label">Theme</span>
          <div class="seg" role="group" aria-label="Theme">
            {#each [["system", "System"], ["dark", "Dark"], ["light", "Light"]] as const as [t, n] (t)}
              <button aria-pressed={s.theme === t} onclick={() => { setS("theme", t); store.applyTheme(); }}>{n}</button>
            {/each}
          </div>
        </div>
      </section>
    {/if}
  </div>
</div>

<style>
  .scrim {
    position: fixed;
    inset: 0;
    z-index: var(--z-drawer);
    background: var(--scrim);
  }
  .drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    z-index: var(--z-drawer);
    width: min(440px, 92vw);
    display: grid;
    grid-template-rows: auto auto 1fr;
    border-left: 1px solid var(--line-strong);
    background: var(--panel);
    box-shadow: var(--shadow);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 10px 6px 18px;
  }
  h2 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
  }
  .tabs {
    display: flex;
    gap: 2px;
    padding: 0 12px;
    border-bottom: 1px solid var(--line);
  }
  .tabs button {
    height: 34px;
    padding: 0 9px;
    border: 0;
    background: none;
    color: var(--text-2);
    font-weight: 500;
    box-shadow: inset 0 -2px 0 transparent;
  }
  .tabs button:hover {
    color: var(--text);
  }
  .tabs button[aria-pressed="true"] {
    color: var(--text);
    box-shadow: inset 0 -2px 0 var(--accent);
  }
  .content {
    overflow-y: auto;
    padding: 6px 18px 28px;
  }
  section {
    display: grid;
    gap: 16px;
    padding: 16px 0;
  }
  section + section {
    border-top: 1px solid var(--line);
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  .row > div:first-child {
    display: grid;
    gap: 2px;
  }
  .title {
    font-weight: 500;
  }
  .val {
    margin-left: 6px;
    color: var(--text-2);
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
  }
  .note {
    margin: 12px 0 0;
    padding: 8px 10px;
    border-radius: var(--r-ctl);
    background: var(--accent-soft);
    color: var(--text);
    font-size: 12px;
  }
  .refresh {
    justify-self: start;
  }
  .cal {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .cal th {
    text-align: left;
    font-weight: 500;
    color: var(--text-3);
    padding: 0 0 6px;
  }
  .cal td {
    padding: 4px 0;
  }
  .cal tr.rec td {
    color: var(--accent);
  }
  .head {
    display: grid;
    gap: 2px;
  }
  .st-progress {
    height: 3px;
    border-radius: 2px;
    background: var(--inset);
    overflow: hidden;
  }
  .st-progress span {
    display: block;
    height: 100%;
    width: 100%;
    background: var(--accent);
    transform-origin: left;
    animation: st-fill 7.4s linear forwards;
  }
  @keyframes st-fill {
    from {
      transform: scaleX(0);
    }
    to {
      transform: scaleX(1);
    }
  }
  .st-error {
    margin: 0;
    padding: 8px 10px;
    border-radius: var(--r-ctl);
    background: color-mix(in srgb, var(--bad) 12%, transparent);
    color: var(--text);
    font-size: 12px;
  }
  .st-result {
    display: grid;
    gap: 10px;
  }
  .st-figs {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .st-big {
    font-size: 30px;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .st-unit {
    margin-left: 6px;
    color: var(--text-2);
  }
  .st-est {
    color: var(--text-3);
    font-size: 12px;
  }
  .st-scale {
    position: relative;
    height: 22px;
    border-radius: var(--r-ctl);
    background: var(--inset);
    border: 1px solid var(--line);
  }
  .st-mark {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
  }
  .st-mark.run {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--accent) 55%, transparent);
  }
  .st-mark.med {
    width: 2px;
    height: 16px;
    background: var(--accent);
  }
  .st-mark.est {
    width: 0;
    height: 16px;
    border-left: 2px dashed var(--text-3);
  }
</style>
