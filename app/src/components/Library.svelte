<script lang="ts">
  import MagnifyingGlass from "phosphor-svelte/lib/MagnifyingGlass";
  import Star from "phosphor-svelte/lib/Star";
  import FilePlus from "phosphor-svelte/lib/FilePlus";
  import FolderOpen from "phosphor-svelte/lib/FolderOpen";
  import DotsThree from "phosphor-svelte/lib/DotsThree";
  import Waveform from "phosphor-svelte/lib/Waveform";
  import X from "phosphor-svelte/lib/X";
  import { slide } from "svelte/transition";
  import { open } from "@tauri-apps/plugin-dialog";
  import HotkeyInput from "./HotkeyInput.svelte";
  import { store } from "../lib/store.svelte";
  import { pickFiles, pickFolder } from "../lib/importing";
  import type { LibraryVoice } from "../lib/api";

  let query = $state("");
  let favOnly = $state(false);
  let menuFor = $state<string | null>(null);
  let confirmRemove = $state<string | null>(null);
  let reportOpen = $state(false);

  let list = $derived(
    store.ordered.filter(
      (v) => (!favOnly || v.favorite) && (!query || v.name.toLowerCase().includes(query.trim().toLowerCase())),
    ),
  );

  const khz = (v: LibraryVoice) => `${v.info.sr / 1000}k`;
  const short = (p: string) => p.split(/[\\/]/).pop() ?? p;

  async function chooseIndex(v: LibraryVoice) {
    const path = await open({ filters: [{ name: "RVC index", extensions: ["index"] }] });
    if (typeof path === "string") store.setVoiceIndex(v, path);
  }

  function onKey(e: KeyboardEvent, v: LibraryVoice) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      store.select(v.id);
    }
  }

  $effect(() => {
    if (!menuFor) return;
    const close = (e: PointerEvent) => {
      if (!(e.target as HTMLElement).closest(".menu, .more")) {
        menuFor = null;
        confirmRemove = null;
      }
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  });
</script>

<aside class="rail" aria-label="Voice library">
  <div class="head">
    <h2>Voices <span class="num count">{store.library.length || ""}</span></h2>
    <div class="imp">
      <button class="btn ghost icon" title="Import files" aria-label="Import files" onclick={pickFiles}>
        <FilePlus size={17} />
      </button>
      <button class="btn ghost icon" title="Import a folder" aria-label="Import a folder" onclick={pickFolder}>
        <FolderOpen size={17} />
      </button>
    </div>
  </div>

  {#if store.library.length}
    <div class="find">
      <label class="search">
        <span class="sr-only">Search voices</span>
        <MagnifyingGlass size={14} />
        <input class="input" type="search" placeholder="Search voices" bind:value={query} />
      </label>
      <button
        class="btn icon"
        class:fav-on={favOnly}
        aria-pressed={favOnly}
        title="Favorites only"
        aria-label="Favorites only"
        onclick={() => (favOnly = !favOnly)}
      >
        <Star size={15} weight={favOnly ? "fill" : "regular"} />
      </button>
    </div>
  {/if}

  {#if store.importing}
    <div class="importing" transition:slide={{ duration: 150 }}>
      <div class="row-line">
        <span>Importing{store.importing.total ? ` ${store.importing.done + 1} of ${store.importing.total}` : ""}</span>
        {#if store.importing.name}<span class="nm">{store.importing.name}</span>{/if}
      </div>
      <div class="prog">
        <span style:width="{store.importing.total ? ((store.importing.done + 0.5) / store.importing.total) * 100 : 8}%"></span>
      </div>
    </div>
  {/if}

  <div class="list" role="listbox" aria-label="Voices">
    {#each list as v (v.id)}
      {@const isLive = store.live && v.id === store.activeVoiceId}
      <div
        class="voice"
        class:sel={v.id === store.selectedId}
        class:is-live={isLive}
        role="option"
        aria-selected={v.id === store.selectedId}
        tabindex="0"
        onclick={() => store.select(v.id)}
        onkeydown={(e) => onKey(e, v)}
      >
        <button
          class="star"
          class:on={v.favorite}
          aria-label={v.favorite ? "Remove from favorites" : "Add to favorites"}
          onclick={(e) => {
            e.stopPropagation();
            store.toggleFavorite(v);
          }}
        >
          <Star size={14} weight={v.favorite ? "fill" : "regular"} />
        </button>
        <div class="txt">
          <span class="name">{v.name}</span>
          <span class="sub num">
            {khz(v)}{#if v.index_path}<span class:shared={store.indexSuspect(v)} title={store.indexSuspect(v) ? `Index shared with ${v.index_shared_with?.join(", ")}` : undefined}>{"  idx"}</span>{/if}{#if v.hotkey}<kbd>{v.hotkey}</kbd>{/if}
          </span>
        </div>
        {#if isLive}<span class="live-tag">Live</span>{/if}
        <button
          class="more"
          aria-label="Voice options"
          aria-expanded={menuFor === v.id}
          onclick={(e) => {
            e.stopPropagation();
            menuFor = menuFor === v.id ? null : v.id;
            confirmRemove = null;
          }}
        >
          <DotsThree size={18} weight="bold" />
        </button>
        {#if menuFor === v.id}
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div class="menu" onclick={(e) => e.stopPropagation()}>
            <div class="field">
              <span class="label">Hotkey</span>
              <HotkeyInput value={v.hotkey ?? ""} onchange={(a) => store.setVoiceHotkey(v, a || null)} />
            </div>
            <p class="file" title={v.source}>{v.model_file}</p>
            <div class="field">
              <span class="label">Index</span>
              <p class="file">{v.index_file ?? "None"}</p>
              {#if store.indexSuspect(v)}
                <p class="shared">Also paired with {v.index_shared_with?.join(", ")}. An index belongs to one voice, so this is probably not {v.name}'s.</p>
              {:else if v.index_shared_with?.length}
                <p class="hint">Made for this voice, but also paired with {v.index_shared_with.join(", ")}.</p>
              {/if}
              <div class="ix-acts">
                <button class="btn" onclick={() => chooseIndex(v)}>Choose index…</button>
                {#if v.index_file}<button class="btn ghost" onclick={() => store.setVoiceIndex(v, null)}>Remove</button>{/if}
              </div>
            </div>
            {#if confirmRemove === v.id}
              <div class="confirm">
                <span>Remove {v.name}? Its copied files are deleted.</span>
                <div>
                  <button class="btn" onclick={() => (confirmRemove = null)}>Keep</button>
                  <button
                    class="btn danger"
                    onclick={() => {
                      store.remove(v);
                      menuFor = null;
                    }}>Remove</button
                  >
                </div>
              </div>
            {:else}
              <button class="btn ghost danger" onclick={() => (confirmRemove = v.id)}>Remove from library</button>
            {/if}
          </div>
        {/if}
      </div>
    {:else}
      {#if store.library.length}
        <p class="nomatch">No voices match{favOnly ? " among favorites" : ""}.</p>
      {:else if !store.importing}
        <div class="empty">
          <Waveform size={28} weight="light" />
          <strong>No voices yet</strong>
          <p>Drop <code>.pth</code>, <code>.index</code> or <code>.zip</code> files anywhere in the window, or import a whole folder such as Vonovox's <code>models</code>.</p>
          <div class="empty-acts">
            <button class="btn primary" onclick={pickFiles}>Import files</button>
            <button class="btn" onclick={pickFolder}>Import folder</button>
          </div>
        </div>
      {/if}
    {/each}
  </div>

  {#if store.report}
    {@const r = store.report}
    <div class="report" transition:slide={{ duration: 150 }}>
      <div class="report-head">
        <button class="sum" onclick={() => (reportOpen = !reportOpen)} aria-expanded={reportOpen}>
          Added {r.added.length}{r.skipped.length ? `, skipped ${r.skipped.length}` : ""}{r.failed.length
            ? `, ${r.failed.length} failed`
            : ""}
        </button>
        <button class="btn ghost icon" aria-label="Dismiss" onclick={() => (store.report = null)}><X size={14} /></button>
      </div>
      {#if reportOpen || r.failed.length}
        <ul>
          {#each r.failed as f (f.path)}
            <li class="bad"><span class="nm">{short(f.path)}</span><span>{f.error}</span></li>
          {/each}
          {#if reportOpen}
            {#each r.skipped as s (s.path)}
              <li><span class="nm">{short(s.path)}</span><span>{s.reason}</span></li>
            {/each}
          {/if}
        </ul>
      {/if}
    </div>
  {/if}
</aside>

<style>
  /* A column, not a fixed row template: the search, progress and report rows come and go, and
     the list must always be the part that takes the spare height and scrolls. */
  .rail {
    grid-area: rail;
    display: flex;
    flex-direction: column;
    min-height: 0;
    border-right: 1px solid var(--line);
    background: var(--panel);
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 10px 8px 14px;
  }
  h2 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }
  .count {
    margin-left: 4px;
    color: var(--text-3);
    font-weight: 400;
  }
  .imp {
    display: flex;
  }
  .find {
    display: flex;
    gap: 6px;
    padding: 0 10px 8px;
  }
  .search {
    position: relative;
    flex: 1;
    display: flex;
    align-items: center;
    color: var(--text-3);
  }
  .search :global(svg) {
    position: absolute;
    left: 9px;
    pointer-events: none;
  }
  .search .input {
    width: 100%;
    padding-left: 29px;
  }
  .fav-on {
    color: var(--accent);
    border-color: var(--accent);
  }
  .list {
    flex: 1;
    overflow-y: auto;
    padding: 2px 6px 10px;
    min-height: 0;
  }
  .voice {
    position: relative;
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    align-items: center;
    gap: 6px;
    padding: 7px 6px 7px 4px;
    border-radius: var(--r-ctl);
    cursor: pointer;
    outline-offset: -2px;
  }
  .voice:hover {
    background: var(--raised);
  }
  .voice.sel {
    background: var(--raised);
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .voice.is-live::before {
    content: "";
    position: absolute;
    left: -6px;
    top: 7px;
    bottom: 7px;
    width: 3px;
    border-radius: 0 2px 2px 0;
    background: var(--accent);
  }
  .star,
  .more {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: 0;
    border-radius: var(--r-ctl);
    background: transparent;
    color: var(--text-3);
  }
  .star.on {
    color: var(--accent);
  }
  .star:hover,
  .more:hover {
    color: var(--text);
    background: var(--inset);
  }
  .more {
    opacity: 0;
  }
  .voice:hover .more,
  .voice:focus-within .more,
  .more[aria-expanded="true"] {
    opacity: 1;
  }
  .txt {
    display: grid;
    min-width: 0;
  }
  .name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sub {
    font-size: 11px;
    color: var(--text-3);
    white-space: pre;
  }
  kbd {
    margin-left: 8px;
    font: 10.5px var(--mono);
    color: var(--text-2);
  }
  .live-tag {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
  }
  .menu {
    position: absolute;
    z-index: var(--z-drawer);
    top: calc(100% - 2px);
    right: 4px;
    width: 236px;
    display: grid;
    gap: 10px;
    padding: 12px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-panel);
    background: var(--raised);
    box-shadow: var(--shadow);
    cursor: default;
  }
  .file {
    margin: 0;
    font: 11px var(--mono);
    color: var(--text-3);
    overflow-wrap: anywhere;
  }
  .shared {
    color: var(--warn);
  }
  p.shared {
    margin: 0;
    font-size: 12px;
  }
  .ix-acts {
    display: flex;
    gap: 6px;
  }
  .confirm {
    display: grid;
    gap: 8px;
    font-size: 12px;
    color: var(--text-2);
  }
  .confirm div {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
  }
  .nomatch {
    margin: 16px 10px;
    color: var(--text-3);
  }
  .empty {
    display: grid;
    justify-items: start;
    gap: 8px;
    margin: 14px 8px;
    padding: 16px 14px;
    border: 1px dashed var(--line-strong);
    border-radius: var(--r-panel);
    color: var(--text-3);
  }
  .empty strong {
    color: var(--text);
    font-weight: 600;
  }
  .empty p {
    margin: 0;
    color: var(--text-2);
    font-size: 12.5px;
  }
  .empty-acts {
    display: flex;
    gap: 6px;
    margin-top: 4px;
  }
  code {
    font: 11.5px var(--mono);
    color: var(--text);
  }
  .importing {
    display: grid;
    gap: 6px;
    margin: 0 10px 8px;
    padding: 8px 10px;
    border-radius: var(--r-ctl);
    background: var(--inset);
    font-size: 12px;
  }
  .row-line {
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }
  .nm {
    color: var(--text-3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .prog {
    height: 3px;
    border-radius: 2px;
    background: var(--line);
    overflow: hidden;
  }
  .prog span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.3s var(--ease);
  }
  .report {
    flex: none;
    margin: 0 8px 8px;
    border: 1px solid var(--line);
    border-radius: var(--r-panel);
    background: var(--raised);
    max-height: 40vh;
    overflow: auto;
  }
  .report-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 4px 4px 10px;
  }
  .sum {
    border: 0;
    background: none;
    padding: 0;
    font-weight: 500;
    text-align: left;
  }
  .report ul {
    margin: 0;
    padding: 0 10px 10px;
    list-style: none;
    display: grid;
    gap: 6px;
    font-size: 12px;
  }
  .report li {
    display: grid;
    gap: 1px;
    color: var(--text-2);
    overflow-wrap: anywhere;
  }
  .report li.bad span:last-child {
    color: var(--bad);
  }
</style>
