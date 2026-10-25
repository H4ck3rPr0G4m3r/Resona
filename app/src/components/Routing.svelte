<script lang="ts">
  import Microphone from "phosphor-svelte/lib/Microphone";
  import SpeakerHigh from "phosphor-svelte/lib/SpeakerHigh";
  import Headphones from "phosphor-svelte/lib/Headphones";
  import Waveform from "phosphor-svelte/lib/Waveform";
  import CaretRight from "phosphor-svelte/lib/CaretRight";
  import Fader from "./Fader.svelte";
  import { store } from "../lib/store.svelte";
  import { api, type DeviceInfo } from "../lib/api";

  const MODE: Record<string, string> = { shared: "Shared", low_latency: "Low latency", exclusive: "Exclusive" };

  function name(list: DeviceInfo[], id: string | null) {
    const d = id ? list.find((x) => x.id === id) : list.find((x) => x.is_default);
    if (!d) return id ? "Unavailable" : "System default";
    return id ? d.name : `Default (${d.name})`;
  }
  let live = $derived(store.stats?.devices ?? null);
  let s = $derived(store.settings);
  const openDevices = () => {
    store.settingsTab = "devices";
    store.settingsOpen = true;
  };
  const db = (x: number) => (x > 0 ? `+${x}` : `${x}`);
</script>

<footer class="route" aria-label="Signal routing">
  <button class="node" onclick={openDevices} title="Input device">
    <Microphone size={15} />
    <span class="dev">{store.live && live ? live.input.device_name : name(store.inputs, s.input_device)}</span>
    <span class="mode">{MODE[s.input_mode]}</span>
  </button>
  <CaretRight size={12} class="arrow" />
  <div class="node voice">
    <Waveform size={15} />
    <span class="dev">{store.selected?.name ?? "No voice"}</span>
  </div>
  <CaretRight size={12} class="arrow" />
  <button class="node" onclick={openDevices} title="Output device">
    <SpeakerHigh size={15} />
    <span class="dev">{store.live && live ? live.output.device_name : name(store.outputs, s.output_device)}</span>
    <span class="mode">{MODE[s.output_mode]}</span>
  </button>
  {#if s.monitor_enabled}
    <button class="node" onclick={openDevices} title="Monitor device">
      <Headphones size={15} />
      <span class="dev">{name(store.outputs, s.monitor_device)}</span>
    </button>
  {/if}

  <div class="gain">
    <span class="label">Out</span>
    <div class="f">
      <Fader
        label="Output level"
        value={s.output_gain_db}
        min={-24}
        max={12}
        step={0.5}
        defaultValue={0}
        origin={0}
        valueText="{db(s.output_gain_db)} dB"
        onchange={(v) => {
          s.output_gain_db = v;
          api.setOutputGain(v);
          store.saveSettings();
        }}
      />
    </div>
    <span class="num val">{db(s.output_gain_db)} dB</span>
  </div>
</footer>

<style>
  .route {
    grid-area: route;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 14px;
    border-top: 1px solid var(--line);
    background: var(--panel);
    min-width: 0;
  }
  .node {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 10px;
    min-width: 0;
    max-width: 260px;
    flex: 0 1 auto;
    border: 1px solid var(--line);
    border-radius: var(--r-ctl);
    background: var(--inset);
    color: var(--text-2);
  }
  button.node:hover {
    border-color: var(--line-strong);
    color: var(--text);
  }
  .node.voice {
    flex: none;
    max-width: 200px;
    color: var(--text);
    border-color: color-mix(in srgb, var(--accent) 40%, var(--line));
  }
  .dev {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mode {
    font-size: 11px;
    color: var(--text-3);
    white-space: nowrap;
  }
  .route :global(.arrow) {
    flex: none;
    color: var(--text-3);
  }
  .gain {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    flex: none;
  }
  .f {
    width: 150px;
  }
  .val {
    width: 58px;
    font-size: 12px;
    color: var(--text-2);
  }
  @media (max-width: 1100px) {
    .mode {
      display: none;
    }
  }
</style>
