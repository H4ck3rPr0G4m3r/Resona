<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import DownloadSimple from "phosphor-svelte/lib/DownloadSimple";
  import X from "phosphor-svelte/lib/X";
  import TopBar from "./components/TopBar.svelte";
  import Library from "./components/Library.svelte";
  import LiveView from "./components/LiveView.svelte";
  import ChainStrip from "./components/ChainStrip.svelte";
  import SettingsDrawer from "./components/SettingsDrawer.svelte";
  import { store } from "./lib/store.svelte";

  let dragging = $state(false);

  onMount(() => {
    store.init();
    const un = getCurrentWebview().onDragDropEvent((e) => {
      const t = e.payload.type;
      if (t === "enter" || t === "over") dragging = true;
      else if (t === "leave") dragging = false;
      else if (t === "drop") {
        dragging = false;
        const paths = e.payload.paths.filter((p) => /\.(pth|index|zip)$/i.test(p) || !/\.[a-z0-9]{1,5}$/i.test(p));
        if (paths.length) store.importPaths(paths);
        else store.notify("error", "Drop .pth, .index or .zip files, or a folder.");
      }
    });
    return () => un.then((f) => f());
  });
</script>

<div class="app" class:loading={!store.ready}>
  <TopBar />
  <Library />
  <main class="center">
    {#if store.ready}<LiveView />{:else}<div class="skeleton" aria-hidden="true"><span></span><span></span><span></span></div>{/if}
  </main>
  <ChainStrip />
</div>

{#if store.settingsOpen}<SettingsDrawer />{/if}

{#if dragging}
  <div class="drop" transition:fade={{ duration: 120 }}>
    <div class="drop-box">
      <DownloadSimple size={28} />
      <strong>Drop to add voices</strong>
      <span>Models, indexes, zips or whole folders. Files are copied into the library.</span>
    </div>
  </div>
{/if}

<div class="toasts" aria-live="polite">
  {#each store.notices as n (n.id)}
    <div class="toast {n.kind}" transition:fade={{ duration: 140 }}>
      <span>{n.text}</span>
      <button class="btn ghost icon" aria-label="Dismiss" onclick={() => store.dismiss(n.id)}><X size={13} /></button>
    </div>
  {/each}
</div>

<style>
  .app {
    display: grid;
    grid-template-columns: 272px minmax(0, 1fr);
    grid-template-rows: 48px minmax(0, 1fr) auto;
    grid-template-areas:
      "bar bar"
      "rail center"
      "rail route";
    height: 100dvh;
  }
  .center {
    grid-area: center;
    min-width: 0;
    min-height: 0;
    overflow: auto;
  }
  .skeleton {
    display: grid;
    gap: 14px;
    padding: 22px;
  }
  .skeleton span {
    height: 26px;
    width: 40%;
    border-radius: var(--r-ctl);
    background: var(--raised);
  }
  .skeleton span:nth-child(2) {
    width: 70%;
    height: 120px;
  }
  .skeleton span:nth-child(3) {
    width: 100%;
    height: 220px;
  }
  .drop {
    position: fixed;
    inset: 0;
    z-index: var(--z-overlay);
    display: grid;
    place-items: center;
    background: var(--scrim);
  }
  .drop-box {
    display: grid;
    justify-items: center;
    gap: 6px;
    padding: 36px 44px;
    border: 2px dashed var(--accent);
    border-radius: var(--r-panel);
    background: var(--panel);
    color: var(--text-2);
    text-align: center;
    max-width: 380px;
  }
  .drop-box :global(svg) {
    color: var(--accent);
  }
  .drop-box strong {
    color: var(--text);
    font-size: 15px;
  }
  .toasts {
    position: fixed;
    right: 14px;
    bottom: 64px;
    z-index: var(--z-toast);
    display: grid;
    gap: 8px;
    max-width: 380px;
  }
  .toast {
    display: flex;
    align-items: start;
    gap: 8px;
    padding: 9px 6px 9px 12px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-panel);
    background: var(--raised);
    box-shadow: var(--shadow);
    overflow-wrap: anywhere;
  }
  .toast.error {
    border-color: color-mix(in srgb, var(--bad) 50%, var(--line));
  }
  .toast span {
    padding-top: 5px;
  }
  @media (max-width: 1000px) {
    .app {
      grid-template-columns: 236px minmax(0, 1fr);
    }
  }
</style>
