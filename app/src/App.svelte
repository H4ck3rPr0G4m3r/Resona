<script lang="ts">
  // M2 functional shell — replaced by the full UI in M4.
  import { onMount } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import { api, type DeviceInfo, type EngineState, type Stats } from "./lib/api";

  let inputs = $state<DeviceInfo[]>([]);
  let outputs = $state<DeviceInfo[]>([]);
  let input = $state("");
  let output = $state("");
  let model = $state("");
  let index = $state("");
  let pitch = $state(0);
  let state = $state<EngineState>("stopped");
  let message = $state<string | null>(null);
  let stats = $state<Stats | null>(null);
  let inLevel = $state(0);
  let outLevel = $state(0);

  onMount(() => {
    api.listDevices().then((d) => {
      inputs = d.inputs;
      outputs = d.outputs;
      input = d.inputs.find((x) => x.is_default)?.id ?? "";
      output = d.outputs.find((x) => x.name.includes("Line 1"))?.id ?? d.outputs[0]?.id ?? "";
    });
    const un = api.onEvent((e) => {
      if (e.type === "state") {
        state = e.state;
        message = e.message;
      } else if (e.type === "stats") stats = e;
      else if (e.type === "meters") {
        inLevel = e.input_peak;
        outLevel = e.output_peak;
      }
    });
    return () => un.then((f) => f());
  });

  async function pickModel() {
    const p = await open({ filters: [{ name: "RVC model", extensions: ["pth"] }] });
    if (typeof p === "string") model = p;
  }
  async function pickIndex() {
    const p = await open({ filters: [{ name: "Index", extensions: ["index"] }] });
    if (typeof p === "string") index = p;
  }
  async function start() {
    message = null;
    try {
      await api.start({
        input: { device_id: input || null, mode: "shared" },
        output: { device_id: output || null, mode: "shared" },
        monitor: null,
        stream: {},
        voice: { id: model, path: model, index_path: index || null },
        params: { pitch },
        jitter_ms: 10,
        down_mode: "mute",
      });
    } catch (e) {
      message = String(e);
    }
  }
</script>

<main>
  <h1>Resona <small>M2 shell</small></h1>
  <label>Input <select bind:value={input}>{#each inputs as d}<option value={d.id}>{d.name}</option>{/each}</select></label>
  <label>Output <select bind:value={output}>{#each outputs as d}<option value={d.id}>{d.name}</option>{/each}</select></label>
  <label>Model <button onclick={pickModel}>Choose .pth</button> <code>{model || "—"}</code></label>
  <label>Index <button onclick={pickIndex}>Choose .index</button> <code>{index || "—"}</code></label>
  <label>Pitch {pitch}
    <input type="range" min="-24" max="24" step="1" bind:value={pitch} onchange={() => state === "running" && api.setParams({ pitch })} />
  </label>
  <div class="row">
    <button disabled={!model || state === "starting" || state === "loading"} onclick={start}>Start</button>
    <button onclick={() => api.stop()}>Stop</button>
    <span class="state">{state}</span>
  </div>
  {#if message}<pre class="msg">{message}</pre>{/if}
  <div class="meters"><div style:width="{Math.min(inLevel, 1) * 100}%"></div></div>
  <div class="meters"><div style:width="{Math.min(outLevel, 1) * 100}%"></div></div>
  {#if stats}
    <pre>{JSON.stringify({ p50: stats.p50_ms, p99: stats.p99_ms, latency: stats.latency?.total_ms, underruns: stats.underruns, ncc: stats.sola_ncc }, null, 1)}</pre>
  {/if}
</main>

<style>
  :global(body) { margin: 0; background: #0e1012; color: #d9d6cf; font: 14px/1.4 system-ui, sans-serif; }
  main { padding: 24px; display: grid; gap: 12px; max-width: 760px; }
  label { display: flex; gap: 8px; align-items: center; }
  .row { display: flex; gap: 8px; align-items: center; }
  .state { font-family: ui-monospace, monospace; color: #ffb040; }
  .msg { color: #ff8a7a; white-space: pre-wrap; }
  .meters { height: 6px; background: #1c1f22; }
  .meters div { height: 100%; background: #ffb040; }
</style>
