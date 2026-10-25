<script lang="ts">
  import Warning from "phosphor-svelte/lib/Warning";
  import WarningOctagon from "phosphor-svelte/lib/WarningOctagon";
  import Info from "phosphor-svelte/lib/Info";
  import { slide } from "svelte/transition";
  import { store } from "../lib/store.svelte";

  let showLog = $state(false);

  /** What to do about dropouts, from what the engine reports. The GPU work per block is set by
   *  the context window, not the block size, so a bigger block alone rarely fixes a busy GPU. */
  function dropoutAdvice(): string {
    const st = store.stats;
    const block = st?.latency?.block_ms ?? 0;
    if (st?.gpu_priority === "normal")
      return "A game gets as much of the GPU as the voice: Resona could not raise its GPU priority. Start Resona once as administrator (right-click, Run as administrator).";
    if (st && block && st.p99_ms > 0.8 * block)
      return `A block of ${block} ms takes the GPU up to ${Math.round(st.p99_ms)} ms. Lower Context in Settings, Pipeline (it sets the work per block), or cap the game's frame rate.`;
    return "The GPU is busy elsewhere now and then. Resona adds buffer by itself; capping a game's frame rate also helps.";
  }

  type Banner = {
    kind: "bad" | "warn" | "info";
    title: string;
    body?: string | null;
    progress?: number;
    action?: { label: string; run: () => void };
    log?: boolean;
  };

  let banner = $derived.by((): Banner | null => {
    const e = store.engine;
    if (e.state === "failed")
      return {
        kind: "bad",
        title: "The engine stopped",
        body: e.message,
        action: { label: "Try again", run: () => store.start() },
        log: true,
      };
    if (e.state === "device_lost")
      return { kind: "warn", title: "Audio device unavailable. Reconnecting every 2 s.", body: e.message };
    if (e.state === "restarting")
      return { kind: "warn", title: "Engine crashed and is restarting", body: e.message };
    if (store.assets)
      return {
        kind: "info",
        title: `Downloading ${store.assets.what}`,
        progress: store.assets.total ? store.assets.done / store.assets.total : undefined,
      };
    if (store.calibrating)
      return { kind: "info", title: "Measuring this GPU at 40, 60, 80 and 100 ms blocks to pick the lowest safe latency." };
    if (e.state === "starting" || e.state === "loading")
      return { kind: "info", title: e.message ? `${e.message[0].toUpperCase()}${e.message.slice(1)}` : "Starting the engine" };
    if (store.running && store.recentUnderruns > 0)
      return {
        kind: "warn",
        title: `${store.recentUnderruns} dropout${store.recentUnderruns > 1 ? "s" : ""} in the last 10 s`,
        body: dropoutAdvice(),
        action: {
          label: "Performance",
          run: () => {
            store.settingsTab = "performance";
            store.settingsOpen = true;
          },
        },
      };
    return null;
  });

  async function copyLog() {
    await store.loadLog();
    await navigator.clipboard.writeText(store.logTail.join("\n"));
    store.notify("info", "Engine log copied");
  }
</script>

{#if banner}
  <div class="banner {banner.kind}" role={banner.kind === "bad" ? "alert" : "status"} transition:slide={{ duration: 160 }}>
    <span class="ico">
      {#if banner.kind === "bad"}<WarningOctagon size={17} weight="fill" />
      {:else if banner.kind === "warn"}<Warning size={17} weight="fill" />
      {:else}<Info size={17} weight="fill" />{/if}
    </span>
    <div class="text">
      <strong>{banner.title}</strong>
      {#if banner.body}<p>{banner.body}</p>{/if}
      {#if banner.progress !== undefined}
        <div class="prog"><span style:width="{banner.progress * 100}%"></span></div>
      {/if}
      {#if banner.log && showLog}
        <pre class="log">{store.logTail.length ? store.logTail.join("\n") : "No engine output captured."}</pre>
      {/if}
    </div>
    <div class="acts">
      {#if banner.log}
        <button class="btn ghost" onclick={() => (showLog = !showLog)}>{showLog ? "Hide log" : "Show log"}</button>
        <button class="btn ghost" onclick={copyLog}>Copy log</button>
      {/if}
      {#if banner.action}
        <button class="btn" onclick={banner.action.run}>{banner.action.label}</button>
      {/if}
    </div>
  </div>
{/if}

<style>
  .banner {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 10px;
    align-items: start;
    padding: 10px 12px;
    border-radius: var(--r-panel);
    border: 1px solid var(--line);
    background: var(--panel);
  }
  .banner.bad {
    border-color: color-mix(in srgb, var(--bad) 45%, transparent);
    background: var(--bad-soft);
  }
  .banner.warn {
    border-color: color-mix(in srgb, var(--warn) 45%, transparent);
    background: var(--warn-soft);
  }
  .ico {
    display: inline-flex;
    padding-top: 1px;
    color: var(--text-2);
  }
  .bad .ico {
    color: var(--bad);
  }
  .warn .ico {
    color: var(--warn);
  }
  .text {
    min-width: 0;
  }
  strong {
    font-weight: 600;
  }
  p {
    margin: 2px 0 0;
    color: var(--text-2);
    overflow-wrap: anywhere;
  }
  .acts {
    display: flex;
    gap: 4px;
  }
  .acts .btn {
    height: 26px;
  }
  .prog {
    margin-top: 8px;
    height: 4px;
    border-radius: 2px;
    background: var(--inset);
    overflow: hidden;
  }
  .prog span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 0.3s var(--ease);
  }
  .log {
    margin: 8px 0 0;
    max-height: 160px;
    overflow: auto;
    padding: 8px 10px;
    border-radius: var(--r-ctl);
    background: var(--inset);
    font: 11px/1.5 var(--mono);
    color: var(--text-2);
    white-space: pre-wrap;
    user-select: text;
  }
</style>
