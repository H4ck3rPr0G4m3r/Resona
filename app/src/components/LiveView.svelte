<script lang="ts">
  import Power from "phosphor-svelte/lib/Power";
  import CircleNotch from "phosphor-svelte/lib/CircleNotch";
  import Waveform from "phosphor-svelte/lib/Waveform";
  import Fader from "./Fader.svelte";
  import Meter from "./Meter.svelte";
  import PitchTrace from "./PitchTrace.svelte";
  import LatencyBar from "./LatencyBar.svelte";
  import StatusBanner from "./StatusBanner.svelte";
  import { store } from "../lib/store.svelte";
  import { pickFiles } from "../lib/importing";

  let v = $derived(store.selected);
  let p = $derived(store.profile);
  let hasIndex = $derived(!!v?.index_path);
  let switching = $derived(store.live && v && store.activeVoiceId && v.id !== store.activeVoiceId);

  const semis = (x: number) => (x > 0 ? `+${x}` : `${x}`);
  const pitchTicks = Array.from({ length: 49 }, (_, i) => i - 24).map((at) => ({
    at,
    label: at % 12 === 0 ? semis(at) : undefined,
  }));
  const formantTicks = Array.from({ length: 13 }, (_, i) => i - 6).map((at) => ({
    at,
    label: at % 3 === 0 ? semis(at) : undefined,
  }));
  const unitTicks = (max: number, n: number) =>
    Array.from({ length: n + 1 }, (_, i) => ({ at: (i * max) / n, label: i === 0 || i === n ? `${(i * max) / n}` : undefined }));

  function power() {
    if (store.live) store.stop();
    else store.start();
  }
</script>

{#if !v}
  <div class="none">
    <Waveform size={34} weight="light" />
    <h2>Import a voice to begin</h2>
    <p>Resona runs RVC v2 models. Add a <code>.pth</code> (and its <code>.index</code>) from the library on the left.</p>
    <button class="btn primary" onclick={pickFiles}>Import files</button>
  </div>
{:else}
  <div class="live">
    <header class="voice">
      <div class="title">
        <h1>{v.name}</h1>
        <p class="meta">
          <span class="num">{v.info.sr / 1000} kHz</span>
          <span>RVC {v.info.version}</span>
          <span>{hasIndex ? "Index paired" : "No index"}</span>
          {#if v.info.epoch}<span class="num">{v.info.epoch} epochs</span>{/if}
          {#if switching}<span class="switching">Switching</span>{/if}
        </p>
      </div>
      <button
        class="power"
        class:on={store.live}
        class:busy={store.busy}
        disabled={store.busy}
        aria-pressed={store.live}
        onclick={power}
      >
        {#if store.busy}
          <span class="spin"><CircleNotch size={18} weight="bold" /></span>
          <span>{store.calibrating ? "Measuring" : "Starting"}</span>
        {:else}
          <Power size={18} weight="bold" />
          <span>{store.live ? "Stop" : "Start"}</span>
        {/if}
      </button>
    </header>

    <StatusBanner />

    <div class="body">
      <section class="controls" aria-label="Voice controls">
        <div class="ctl primary">
          <div class="ctl-head">
            <span class="label">Pitch</span>
            <span class="readout num">{semis(p.pitch)}<small>st</small></span>
          </div>
          <Fader
            size="lg"
            label="Pitch in semitones"
            value={p.pitch}
            min={-24}
            max={24}
            step={1}
            defaultValue={0}
            origin={0}
            ticks={pitchTicks}
            valueText="{semis(p.pitch)} semitones"
            onchange={(x) => store.setParam("pitch", x)}
          />
          <div class="quick">
            {#each [-12, -5, 0, 5, 12] as q (q)}
              <button class="btn" aria-pressed={p.pitch === q} onclick={() => store.setParam("pitch", q)}>
                {q === 0 ? "0" : semis(q)}
              </button>
            {/each}
          </div>
        </div>

        <div class="ctl">
          <div class="ctl-head">
            <span class="label">Formant</span>
            <span class="readout sm num">{semis(p.formant)}<small>st</small></span>
          </div>
          <Fader
            label="Formant shift in semitones"
            value={p.formant}
            min={-6}
            max={6}
            step={0.5}
            defaultValue={0}
            origin={0}
            ticks={formantTicks}
            onchange={(x) => store.setParam("formant", x)}
          />
        </div>

        <div class="pair">
          <div class="ctl">
            <div class="ctl-head">
              <span class="label">Index</span>
              <span class="readout sm num">{hasIndex ? p.index_rate.toFixed(2) : "off"}</span>
            </div>
            <Fader
              label="Index rate"
              value={p.index_rate}
              min={0}
              max={1}
              step={0.01}
              defaultValue={0.5}
              ticks={unitTicks(1, 4)}
              disabled={!hasIndex}
              onchange={(x) => store.setParam("index_rate", x)}
            />
          </div>
          <div class="ctl">
            <div class="ctl-head">
              <span class="label">Protect</span>
              <span class="readout sm num">{p.protect >= 0.5 ? "off" : p.protect.toFixed(2)}</span>
            </div>
            <Fader
              label="Consonant protection"
              value={p.protect}
              min={0}
              max={0.5}
              step={0.01}
              defaultValue={0.33}
              ticks={unitTicks(0.5, 2)}
              onchange={(x) => store.setParam("protect", x)}
            />
          </div>
        </div>
        <p class="hint">
          {hasIndex ? "Index pulls timbre toward the training voice." : "This voice has no index; timbre comes from the model alone."}
          Protect keeps breaths and consonants natural.
        </p>
      </section>

      <section class="scope" aria-label="Pitch and levels">
        <PitchTrace />
        <div class="meters">
          <Meter label="IN" peak={store.meters.inPeak} rms={store.meters.inRms} />
          <Meter label="OUT" peak={store.meters.outPeak} rms={store.meters.outRms} />
        </div>
      </section>
    </div>

    <LatencyBar />
  </div>
{/if}

<style>
  .live {
    display: flex;
    flex-direction: column;
    gap: 18px;
    height: 100%;
    min-height: 0;
    padding: 20px 22px 18px;
  }
  .voice {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  .title {
    min-width: 0;
  }
  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    margin: 5px 0 0;
    color: var(--text-2);
    font-size: 12px;
  }
  .switching {
    color: var(--accent);
  }
  .power {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-width: 128px;
    height: 44px;
    padding: 0 20px;
    border-radius: var(--r-ctl);
    border: 1px solid var(--line-strong);
    background: var(--raised);
    font-size: 14px;
    font-weight: 650;
    flex: none;
    transition: background 0.18s var(--ease), color 0.18s, border-color 0.18s, transform 0.08s;
  }
  .power:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .power:active:not(:disabled) {
    transform: translateY(1px);
  }
  .power.on {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--accent-ink);
  }
  .power:disabled {
    cursor: progress;
  }
  .spin {
    display: inline-flex;
    animation: spin 0.9s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .body {
    flex: 1;
    display: grid;
    grid-template-columns: minmax(300px, 380px) 1fr;
    gap: 22px;
    min-height: 0;
  }
  .controls {
    display: grid;
    align-content: start;
    gap: 18px;
  }
  .ctl {
    display: grid;
    gap: 4px;
  }
  .ctl-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0 8px;
  }
  .readout {
    font-size: 30px;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .readout.sm {
    font-size: 15px;
  }
  .readout small {
    margin-left: 3px;
    font-size: 11px;
    color: var(--text-3);
    font-family: var(--font);
  }
  .quick {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    padding: 4px 8px 0;
  }
  .quick .btn {
    justify-content: center;
    height: 26px;
    font-family: var(--mono);
    font-size: 12px;
  }
  .quick .btn[aria-pressed="true"] {
    border-color: var(--accent);
    color: var(--accent);
  }
  .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .controls .hint {
    margin: -4px 8px 0;
    max-width: 52ch;
  }
  .scope {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    min-height: 220px;
  }
  .meters {
    display: flex;
    gap: 8px;
    padding: 6px 2px 2px;
  }
  .none {
    display: grid;
    place-content: center;
    justify-items: center;
    gap: 10px;
    height: 100%;
    padding: 24px;
    text-align: center;
    color: var(--text-2);
  }
  .none h2 {
    margin: 4px 0 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
  }
  .none p {
    margin: 0 0 8px;
    max-width: 44ch;
  }
  code {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
  }
  @media (max-width: 1080px) {
    .body {
      grid-template-columns: 1fr;
      grid-template-rows: auto minmax(200px, 1fr);
    }
  }
</style>
