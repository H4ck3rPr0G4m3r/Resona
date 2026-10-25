<script lang="ts">
  // In-place editor for the selected chain slot (generated from the catalog), the voice gate or
  // the limiter.
  import X from "phosphor-svelte/lib/X";
  import Trash from "phosphor-svelte/lib/Trash";
  import { slide } from "svelte/transition";
  import ParamControl from "./ParamControl.svelte";
  import EqCurve from "./EqCurve.svelte";
  import { store } from "../lib/store.svelte";
  import type { ParamSpec } from "../lib/api";

  const GATE: ParamSpec[] = [
    { key: "threshold", name: "Voice certainty", min: 0.1, max: 0.95, default: 0.6, unit: "", scale: "linear" },
    { key: "hold_ms", name: "Hold", min: 0, max: 1000, default: 200, unit: "ms", scale: "linear" },
    { key: "release_ms", name: "Release", min: 20, max: 1500, default: 300, unit: "ms", scale: "log" },
    { key: "range_db", name: "Closed level", min: -80, max: 0, default: -45, unit: "dB", scale: "linear" },
  ];
  const LIMITER: ParamSpec[] = [{ key: "ceiling_db", name: "Ceiling", min: -12, max: 0, default: -1, unit: "dB", scale: "linear" }];

  let t = $derived(store.fxEditing);
  let slot = $derived(t && "id" in t ? (store.chain(t.chain).slots.find((s) => s.id === t.id) ?? null) : null);
  let kind = $derived(slot ? store.kind(slot.kind) : null);
  let reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function close() {
    store.fxEditing = null;
  }
</script>

{#if t}
  <section class="editor" aria-label="Effect editor" transition:slide={{ duration: reduce ? 0 : 160 }}>
    {#if slot && kind && "id" in t}
      {@const chain = t.chain}
      <header>
        <button class="switch" role="switch" aria-checked={slot.enabled} aria-label="Enabled" onclick={() => store.toggleSlot(chain, slot.id)}></button>
        <h3>{kind.name}</h3>
        <span class="where">{chain === "input" ? "Input chain" : store.voiceHasChain ? "This voice's output chain" : "Output chain"}</span>
        <div class="acts">
          <button class="btn ghost danger" onclick={() => store.removeSlot(chain, slot.id)}><Trash size={14} /> Remove</button>
          <button class="btn ghost icon" aria-label="Close editor" onclick={close}><X size={15} /></button>
        </div>
      </header>
      <div class="body" class:eq={slot.kind === "eq"}>
        {#if slot.kind === "eq"}
          <div class="curve"><EqCurve params={slot.params} onchange={(k, v) => store.setSlotParam(chain, slot.id, k, v)} /></div>
        {/if}
        <div class="grid">
          {#each kind.params as p (p.key)}
            <ParamControl spec={p} value={slot.params[p.key] ?? p.default} onchange={(v) => store.setSlotParam(chain, slot.id, p.key, v)} />
          {/each}
        </div>
      </div>
    {:else if t && "special" in t && t.special === "gate"}
      <header>
        <button class="switch" role="switch" aria-checked={store.settings.fx.gate.enabled} aria-label="Enabled"
          onclick={() => store.setGate({ enabled: !store.settings.fx.gate.enabled })}></button>
        <h3>Voice gate</h3>
        <span class="where">Silences the output between words, using voice detection on your mic. Adds no latency.</span>
        <div class="acts"><button class="btn ghost icon" aria-label="Close editor" onclick={close}><X size={15} /></button></div>
      </header>
      <div class="body">
        <div class="grid">
          {#each GATE as p (p.key)}
            <ParamControl spec={p} value={store.settings.fx.gate[p.key as keyof typeof store.settings.fx.gate] as number}
              onchange={(v) => store.setGate({ [p.key]: v })} />
          {/each}
        </div>
      </div>
    {:else if t && "special" in t}
      <header>
        <button class="switch" role="switch" aria-checked={store.settings.fx.limiter.enabled} aria-label="Enabled"
          onclick={() => store.setLimiter({ enabled: !store.settings.fx.limiter.enabled })}></button>
        <h3>True-peak limiter</h3>
        <span class="where">Always last. Keeps the output, including peaks between samples, under the ceiling.</span>
        <div class="acts"><button class="btn ghost icon" aria-label="Close editor" onclick={close}><X size={15} /></button></div>
      </header>
      <div class="body">
        <div class="grid narrow">
          {#each LIMITER as p (p.key)}
            <ParamControl spec={p} value={store.settings.fx.limiter.ceiling_db} onchange={(v) => store.setLimiter({ ceiling_db: v })} />
          {/each}
        </div>
      </div>
    {/if}
  </section>
{/if}

<style>
  .editor {
    border-top: 1px solid var(--line);
    background: var(--panel);
    padding: 10px 14px 14px;
    max-height: 48vh;
    overflow-y: auto;
  }
  header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }
  .where {
    color: var(--text-3);
    font-size: 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .acts {
    margin-left: auto;
    display: flex;
    gap: 4px;
    flex: none;
  }
  .body {
    display: grid;
    gap: 14px;
  }
  .body.eq {
    grid-template-columns: minmax(260px, 1.1fr) 2fr;
    align-items: stretch;
  }
  .curve {
    min-height: 170px;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 12px 18px;
    align-content: start;
  }
  .grid.narrow {
    max-width: 320px;
  }
</style>
