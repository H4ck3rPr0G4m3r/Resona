<script lang="ts">
  // Signal flow, left to right: mic → input chain → voice → output chain → limiter → output.
  // Click a chip to edit it in place, drag to reorder, the lamp toggles bypass.
  import SourceNode from "./SourceNode.svelte";
  import SpeakerHigh from "phosphor-svelte/lib/SpeakerHigh";
  import Waveform from "phosphor-svelte/lib/Waveform";
  import Plus from "phosphor-svelte/lib/Plus";
  import DotsThree from "phosphor-svelte/lib/DotsThree";
  import CaretRight from "phosphor-svelte/lib/CaretRight";
  import MusicNotes from "phosphor-svelte/lib/MusicNotes";
  import { open, save } from "@tauri-apps/plugin-dialog";
  import Fader from "./Fader.svelte";
  import SlotEditor from "./SlotEditor.svelte";
  import { store, type ChainName } from "../lib/store.svelte";
  import { api, type DeviceInfo, type SlotSpec } from "../lib/api";
  import { formatParam } from "../lib/fxformat";

  let addMenu = $state<ChainName | null>(null);
  let outMenu = $state(false);
  let dragFrom = $state<{ chain: ChainName; index: number } | null>(null);
  let dragOver = $state<{ chain: ChainName; index: number } | null>(null);

  const s = $derived(store.settings);
  const live = $derived(store.stats?.devices ?? null);
  const db = (x: number) => (x > 0 ? `+${x}` : `${x}`);

  function devName(list: DeviceInfo[], id: string | null) {
    const d = id ? list.find((x) => x.id === id) : list.find((x) => x.is_default);
    return d?.name ?? (id ? "Unavailable" : "System default");
  }
  function summary(slot: SlotSpec): string {
    const k = store.kind(slot.kind);
    if (!k) return "";
    const v = (key: string) => slot.params[key] ?? k.params.find((p) => p.key === key)?.default ?? 0;
    const f = (key: string) => formatParam(k.params.find((p) => p.key === key)!, v(key));
    switch (slot.kind) {
      case "filter":
        return `${v("mode") >= 0.5 ? "HP" : "LP"} ${f("freq")}`;
      case "eq": {
        const n = [1, 2, 3, 4, 5].filter((i) => Math.abs(v(`g${i}`)) >= 0.05).length;
        return n ? `${n} band${n > 1 ? "s" : ""}` : "flat";
      }
      case "gate":
      case "compressor":
      case "deesser":
        return f("threshold");
      case "reverb":
      case "chorus":
      case "lofi":
        return `${f("mix")} wet`;
      default:
        return f(k.params[0].key);
    }
  }
  function isEditing(chain: ChainName, id: number) {
    const t = store.fxEditing;
    return !!t && "id" in t && t.chain === chain && t.id === id;
  }
  function edit(chain: ChainName, id: number) {
    store.fxEditing = isEditing(chain, id) ? null : { chain, id };
  }
  function editSpecial(special: "gate" | "limiter") {
    const t = store.fxEditing;
    store.fxEditing = t && "special" in t && t.special === special ? null : { special };
  }
  function addable(chain: ChainName) {
    return store.catalog.filter((k) => k.stage === "both" || k.stage === chain);
  }
  function drop(chain: ChainName, index: number) {
    if (dragFrom && dragFrom.chain === chain) store.moveSlot(chain, dragFrom.index, index);
    dragFrom = dragOver = null;
  }
  async function exportChain() {
    outMenu = false;
    const path = await save({ defaultPath: "output.resona-chain.json", filters: [{ name: "Resona chain", extensions: ["json"] }] });
    if (path) api.chainExport(path, $state.snapshot(store.outputChain)).then(() => store.notify("info", "Chain exported"), (e) => store.notify("error", String(e)));
  }
  async function importChain() {
    outMenu = false;
    const path = await open({ filters: [{ name: "Resona chain", extensions: ["json"] }] });
    if (typeof path === "string") api.chainImport(path).then((c) => store.replaceChain("output", c), (e) => store.notify("error", String(e)));
  }
  let flow: HTMLDivElement | undefined = $state();
  let fadeL = $state(false);
  let fadeR = $state(false);
  function edges() {
    if (!flow) return;
    fadeL = flow.scrollLeft > 1;
    fadeR = flow.scrollLeft + flow.clientWidth < flow.scrollWidth - 1;
  }
  // The scrollbar is hidden, so a vertical wheel scrolls the chain sideways.
  function wheel(e: WheelEvent) {
    if (!flow || flow.scrollWidth <= flow.clientWidth || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    flow.scrollLeft += e.deltaY;
    e.preventDefault();
  }
  $effect(() => {
    if (!flow) return;
    const ro = new ResizeObserver(edges);
    ro.observe(flow);
    for (const c of flow.children) ro.observe(c);
    edges();
    return () => ro.disconnect();
  });
  $effect(() => {
    // Chain contents change the scroll width without resizing the container.
    void store.settings.fx.input.slots.length;
    void store.outputChain.slots.length;
    queueMicrotask(edges);
  });

  const openDevices = () => {
    store.settingsTab = "devices";
    store.settingsOpen = true;
  };

  $effect(() => {
    if (!addMenu && !outMenu) return;
    const close = (e: PointerEvent) => {
      if (!(e.target as HTMLElement).closest(".pop, .pop-btn")) {
        addMenu = null;
        outMenu = false;
      }
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  });
</script>

{#snippet chips(chain: ChainName)}
  {#each store.chain(chain).slots as slot, i (slot.id)}
    {@const k = store.kind(slot.kind)}
    <div
      class="chip"
      class:off={!slot.enabled}
      class:editing={isEditing(chain, slot.id)}
      class:over={dragOver?.chain === chain && dragOver.index === i && dragFrom?.index !== i}
      draggable="true"
      role="listitem"
      ondragstart={() => (dragFrom = { chain, index: i })}
      ondragover={(e) => {
        if (dragFrom?.chain === chain) {
          e.preventDefault();
          dragOver = { chain, index: i };
        }
      }}
      ondrop={() => drop(chain, i)}
      ondragend={() => (dragFrom = dragOver = null)}
    >
      <button
        class="lamp"
        class:on={slot.enabled}
        aria-label="{slot.enabled ? 'Bypass' : 'Enable'} {k?.name}"
        onclick={() => store.toggleSlot(chain, slot.id)}
      ></button>
      <button class="chip-body" onclick={() => edit(chain, slot.id)} aria-expanded={isEditing(chain, slot.id)}>
        <span class="nm">{k?.name ?? slot.kind}</span>
        <span class="sum num">{summary(slot)}</span>
      </button>
    </div>
  {/each}
  <div class="pop-wrap">
    <button class="add pop-btn" aria-label="Add effect" title="Add effect" aria-expanded={addMenu === chain} onclick={() => (addMenu = addMenu === chain ? null : chain)}>
      <Plus size={13} weight="bold" />
    </button>
    {#if addMenu === chain}
      <div class="pop" role="menu">
        {#each addable(chain) as k (k.kind)}
          <button role="menuitem" onclick={() => { store.addSlot(chain, k.kind); addMenu = null; }}>{k.name}</button>
        {/each}
      </div>
    {/if}
  </div>
{/snippet}

<div class="strip-wrap">
  <SlotEditor />
  <footer class="strip" aria-label="Signal chain">
    <div class="flow" class:fade-l={fadeL} class:fade-r={fadeR} role="list" bind:this={flow} onscroll={edges} onwheel={wheel}>
      <SourceNode />
      <CaretRight size={11} class="arrow" />
      {@render chips("input")}
      <button
        class="chip special"
        class:off={!s.fx.gate.enabled || s.fx.singing}
        class:editing={store.fxEditing && "special" in store.fxEditing && store.fxEditing.special === "gate"}
        onclick={() => editSpecial("gate")}
      >
        <span class="lamp" class:on={s.fx.gate.enabled && !s.fx.singing} class:shut={store.live && store.stats?.gate_open === false}></span>
        <span class="nm">Voice gate</span>
      </button>
      <button class="sing" aria-pressed={s.fx.singing} title="Singing mode: bypass noise suppression and the voice gate" onclick={() => store.setSinging(!s.fx.singing)}>
        <MusicNotes size={14} />
      </button>
      <CaretRight size={11} class="arrow" />
      <div class="node voice"><Waveform size={15} /><span class="dev">{store.selected?.name ?? "No voice"}</span></div>
      <CaretRight size={11} class="arrow" />
      {@render chips("output")}
      <div class="pop-wrap">
        <button class="add pop-btn" class:own={store.voiceHasChain} aria-label="Output chain options" aria-expanded={outMenu} onclick={() => (outMenu = !outMenu)}>
          <DotsThree size={14} weight="bold" />
        </button>
        {#if outMenu}
          <div class="pop" role="menu">
            <button role="menuitemcheckbox" aria-checked={store.voiceHasChain} disabled={!store.selected} onclick={() => { store.setVoiceChain(!store.voiceHasChain); outMenu = false; }}>
              {store.voiceHasChain ? "Use the shared output chain" : "Give this voice its own chain"}
            </button>
            <button role="menuitem" onclick={exportChain}>Export preset</button>
            <button role="menuitem" onclick={importChain}>Import preset</button>
          </div>
        {/if}
      </div>
      <button
        class="chip special"
        class:off={!s.fx.limiter.enabled}
        class:editing={store.fxEditing && "special" in store.fxEditing && store.fxEditing.special === "limiter"}
        onclick={() => editSpecial("limiter")}
      >
        <span class="lamp" class:on={s.fx.limiter.enabled}></span>
        <span class="nm">Limiter</span>
        <span class="sum num">{s.fx.limiter.ceiling_db} dB</span>
      </button>
      <CaretRight size={11} class="arrow" />
      <button class="node" onclick={openDevices} title="Output device">
        <SpeakerHigh size={15} />
        <span class="dev">{store.live && live ? live.output.device_name : devName(store.outputs, s.output_device)}</span>
      </button>
    </div>
    <div class="gain">
      <div class="f">
        <Fader label="Output level" value={s.output_gain_db} min={-24} max={12} step={0.5} defaultValue={0} origin={0}
          valueText="{db(s.output_gain_db)} dB"
          onchange={(v) => { s.output_gain_db = v; api.setOutputGain(v); store.saveSettings(); }} />
      </div>
      <span class="num val">{db(s.output_gain_db)} dB</span>
    </div>
  </footer>
</div>

<style>
  .strip-wrap {
    grid-area: route;
    min-width: 0;
  }
  .strip {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 56px;
    padding: 0 12px;
    border-top: 1px solid var(--line);
    background: var(--panel);
  }
  .flow {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
    overflow-x: auto;
    overflow-y: visible;
    padding: 4px 2px;
    scrollbar-width: none;
  }
  /* Edge fades show there is more chain to scroll to. */
  .flow.fade-r {
    mask-image: linear-gradient(to right, #000 calc(100% - 28px), transparent);
  }
  .flow.fade-l {
    mask-image: linear-gradient(to right, transparent, #000 28px);
  }
  .flow.fade-l.fade-r {
    mask-image: linear-gradient(to right, transparent, #000 28px, #000 calc(100% - 28px), transparent);
  }
  .flow :global(.arrow) {
    flex: none;
    color: var(--text-3);
  }
  .node {
    flex: 0 1 auto;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 10px;
    min-width: 0;
    max-width: 210px;
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
    max-width: 180px;
    color: var(--text);
    border-color: color-mix(in srgb, var(--accent) 45%, var(--line));
  }
  .dev {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .chip {
    flex: none;
    display: inline-flex;
    align-items: center;
    height: 32px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-ctl);
    background: var(--raised);
    transition: border-color 0.15s, opacity 0.15s;
  }
  .chip.special {
    gap: 7px;
    padding: 0 10px 0 9px;
  }
  .chip.editing {
    border-color: var(--accent);
  }
  .chip.off .nm,
  .chip.off .sum {
    color: var(--text-3);
  }
  .chip.over {
    box-shadow: -3px 0 0 var(--accent);
  }
  .chip[draggable="true"] {
    cursor: grab;
  }
  /* The lamp itself is drawn by span.lamp or by button.lamp::before (a bigger hit area). */
  span.lamp,
  button.lamp::before {
    flex: none;
    width: 8px;
    height: 8px;
    box-sizing: border-box;
    border: 1px solid var(--line-strong);
    border-radius: 2px;
    background: var(--inset);
  }
  span.lamp.on,
  button.lamp.on::before {
    background: var(--accent);
    border-color: var(--accent);
  }
  span.lamp.on.shut {
    background: var(--inset);
  }
  button.lamp {
    flex: none;
    display: grid;
    place-items: center;
    width: 24px;
    height: 100%;
    padding: 0 0 0 4px;
    border: 0;
    background: none;
  }
  button.lamp::before {
    content: "";
  }
  button.lamp:hover:not(.on)::before {
    border-color: var(--text-3);
  }
  .chip-body {
    display: inline-flex;
    align-items: baseline;
    gap: 7px;
    height: 100%;
    padding: 0 10px 0 2px;
    border: 0;
    background: none;
  }
  .nm {
    font-weight: 500;
    white-space: nowrap;
  }
  .sum {
    font-size: 11px;
    color: var(--text-3);
    white-space: nowrap;
  }
  .add,
  .sing {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: 1px dashed var(--line-strong);
    border-radius: var(--r-ctl);
    background: none;
    color: var(--text-3);
  }
  .add:hover,
  .sing:hover {
    color: var(--text);
    border-color: var(--text-3);
  }
  .add.own {
    border-style: solid;
    border-color: var(--accent);
    color: var(--accent);
  }
  .sing {
    border-style: solid;
    border-color: var(--line);
  }
  .sing[aria-pressed="true"] {
    color: var(--accent);
    border-color: var(--accent);
  }
  .pop-wrap {
    position: relative;
    flex: none;
  }
  .pop {
    position: fixed;
    z-index: var(--z-drawer);
    transform: translateY(calc(-100% - 40px));
    display: grid;
    min-width: 200px;
    padding: 5px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-panel);
    background: var(--raised);
    box-shadow: var(--shadow);
  }
  .pop button {
    height: 30px;
    padding: 0 10px;
    border: 0;
    border-radius: var(--r-ctl);
    background: none;
    text-align: left;
    white-space: nowrap;
  }
  .pop button:hover:not(:disabled) {
    background: var(--inset);
  }
  .pop button:disabled {
    color: var(--text-3);
  }
  .gain {
    display: flex;
    align-items: center;
    gap: 4px;
    flex: none;
  }
  .f {
    width: 120px;
  }
  .val {
    width: 54px;
    font-size: 12px;
    color: var(--text-2);
  }
</style>
