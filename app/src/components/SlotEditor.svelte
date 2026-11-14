<script lang="ts">
  // In-place editor for the selected chain slot (generated from the catalog), the voice gate or
  // the limiter.
  import X from "phosphor-svelte/lib/X";
  import Trash from "phosphor-svelte/lib/Trash";
  import PencilSimple from "phosphor-svelte/lib/PencilSimple";
  import FolderOpen from "phosphor-svelte/lib/FolderOpen";
  import CaretRight from "phosphor-svelte/lib/CaretRight";
  import { slide } from "svelte/transition";
  import ParamControl from "./ParamControl.svelte";
  import EqCurve from "./EqCurve.svelte";
  import { store } from "../lib/store.svelte";
  import type { ParamSpec } from "../lib/api";
  import { fileName, filterNameFromKind, isFilter } from "../lib/fxformat";

  const GATE: ParamSpec[] = [
    { key: "threshold", name: "Voice certainty", min: 0.1, max: 0.95, default: 0.6, unit: "", scale: "linear" },
    { key: "hold_ms", name: "Hold", min: 0, max: 1000, default: 200, unit: "ms", scale: "linear" },
    { key: "release_ms", name: "Release", min: 20, max: 1500, default: 300, unit: "ms", scale: "log" },
    { key: "range_db", name: "Closed level", min: -80, max: 0, default: -45, unit: "dB", scale: "linear" },
  ];
  const LIMITER: ParamSpec[] = [{ key: "ceiling_db", name: "Ceiling", min: -12, max: 0, default: -1, unit: "dB", scale: "linear" }];

  let t = $derived(store.fxEditing);
  let slot = $derived(t && "id" in t ? (store.chain(t.chain).slots.find((s) => s.id === t.id) ?? null) : null);
  let kind = $derived(slot ? (store.kind(slot.kind) ?? null) : null);
  /** A filter slot whose file is no longer in the filters folder. */
  let missing = $derived(!!slot && !kind && isFilter(slot.kind));
  let filter = $derived(kind?.filter ?? null);
  let reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;

  function close() {
    store.fxEditing = null;
  }
</script>

{#if t}
  <section class="editor" aria-label="Effect editor" transition:slide={{ duration: reduce ? 0 : 160 }}>
    {#if slot && (kind || missing) && "id" in t}
      {@const chain = t.chain}
      <header>
        <button class="switch" role="switch" aria-checked={slot.enabled} aria-label="Enabled" onclick={() => store.toggleSlot(chain, slot.id)}></button>
        <h3>{kind?.name ?? filterNameFromKind(slot.kind)}</h3>
        <span class="where">{chain === "input" ? "Input chain" : store.voiceHasChain ? "This voice's output chain" : "Output chain"}</span>
        <div class="acts">
          {#if filter}
            <button
              class="btn ghost"
              title={filter.builtin ? "Copies this built-in filter to your filters folder and opens the copy" : "Opens the file. Saving it updates the filter."}
              onclick={() => store.editFilter(slot.kind)}
            >
              <PencilSimple size={14} /> {filter.builtin ? "Customize" : "Edit file"}
            </button>
          {/if}
          <button class="btn ghost danger" onclick={() => store.removeSlot(chain, slot.id)}><Trash size={14} /> Remove</button>
          <button class="btn ghost icon" aria-label="Close editor" onclick={close}><X size={15} /></button>
        </div>
      </header>
      {#if missing}
        <div class="notice" role="status">
          <p>This filter's file is no longer in the filters folder, so it passes audio through unchanged. Put the file back, or remove the filter from the chain.</p>
          <button class="btn" onclick={() => store.openFiltersFolder()}><FolderOpen size={14} /> Open filters folder</button>
        </div>
      {:else if kind && filter?.error}
        <div class="notice" role="alert">
          <p><strong>{fileName(filter.file)}</strong> could not be used, so the filter passes audio through unchanged. Fix the file and save it.</p>
          <pre class="why">{filter.error}</pre>
        </div>
      {:else if kind}
        <div class="body" class:eq={slot.kind === "eq"}>
          {#if slot.kind === "eq"}
            <div class="curve"><EqCurve params={slot.params} onchange={(k, v) => store.setSlotParam(chain, slot.id, k, v)} /></div>
          {/if}
          {#if filter?.description}<p class="about">{filter.description}</p>{/if}
          <div class="grid">
            {#each kind.params as p (p.key)}
              <ParamControl spec={p} value={slot.params[p.key] ?? p.default} onchange={(v) => store.setSlotParam(chain, slot.id, p.key, v)} />
            {/each}
          </div>
          {#if filter}
            <div class="stages" aria-label="Stages of this filter">
              {#each filter.stages as st, i (i)}
                {#if i > 0}<CaretRight size={10} />{/if}
                <span class="num">{st}</span>
              {/each}
              <span class="file num" title={filter.file}>{fileName(filter.file)}</span>
            </div>
          {/if}
        </div>
      {/if}
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
  .about {
    margin: 0;
    max-width: 78ch;
    color: var(--text-2);
  }
  /* What the filter is made of, read from its file: stage types in signal order. */
  .stages {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 6px;
    padding-top: 10px;
    border-top: 1px solid var(--line);
    font-size: 11.5px;
    color: var(--text-3);
  }
  .stages .file {
    margin-left: auto;
    padding-left: 12px;
  }
  .notice {
    display: grid;
    justify-items: start;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid color-mix(in srgb, var(--warn) 45%, var(--line));
    border-radius: var(--r-panel);
    background: var(--warn-soft);
  }
  .notice p {
    margin: 0;
    max-width: 78ch;
  }
  .why {
    margin: 0;
    max-width: 100%;
    padding: 8px 10px;
    border-radius: var(--r-ctl);
    background: var(--inset);
    font: 12px/1.5 var(--mono);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    user-select: text;
  }
</style>
