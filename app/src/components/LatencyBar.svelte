<script lang="ts">
  import { store } from "../lib/store.svelte";

  let lat = $derived(store.stats?.latency ?? null);
  let segs = $derived(
    lat
      ? [
          { key: "in", name: "Input", ms: lat.input_ms },
          { key: "block", name: "Block", ms: lat.block_ms },
          { key: "look", name: "Lookahead", ms: lat.lookahead_ms },
          { key: "gpu", name: "GPU", ms: lat.compute_ms },
          { key: "buf", name: "Buffer", ms: lat.buffer_ms },
          { key: "out", name: "Output", ms: lat.output_ms },
        ]
      : [],
  );
  let total = $derived(lat ? lat.total_ms : 0);
  let s = $derived(store.stats);
  let block = $derived(store.settings.block_ms ?? store.settings.calibration?.block_ms ?? null);
</script>

<section class="lat" aria-label="Latency">
  <div class="head">
    <span class="label">Mouth to ear</span>
    <span class="total num">{lat ? Math.round(total) : "-"}<small>ms</small></span>
  </div>

  <div class="bar" class:empty={!lat}>
    {#each segs as g (g.key)}
      <div class="seg {g.key}" style:flex-grow={Math.max(g.ms, 0.5)} title="{g.name}: {g.ms.toFixed(1)} ms">
        {#if g.ms / total > 0.075}
          <span class="seg-name">{g.name}</span>
          <span class="seg-ms num">{Math.round(g.ms)}</span>
        {/if}
      </div>
    {/each}
  </div>

  <dl class="facts">
    <div>
      <dt>GPU per block</dt>
      <dd class="num">{s ? `${s.p50_ms.toFixed(1)} / ${s.p99_ms.toFixed(1)}` : "-"}<small>p50 / p99 ms</small></dd>
    </div>
    <div>
      <dt>Block</dt>
      <dd class="num">
        {block ?? "auto"}{#if block}<small>ms{store.settings.block_ms === null ? ", auto" : ""}</small>{/if}
      </dd>
    </div>
    <div class:warn={store.recentUnderruns > 0}>
      <dt>Dropouts</dt>
      <dd class="num">{s?.underruns ?? 0}{#if store.recentUnderruns > 0}<small>{store.recentUnderruns} in 10 s</small>{/if}</dd>
    </div>
    <div>
      <dt>VRAM</dt>
      <dd class="num">{s?.vram_mb ?? "-"}<small>MB</small></dd>
    </div>
  </dl>
</section>

<style>
  .lat {
    display: grid;
    grid-template-columns: auto 1fr;
    grid-template-rows: auto auto;
    column-gap: 18px;
    row-gap: 10px;
    align-items: center;
  }
  .head {
    display: grid;
    gap: 2px;
    grid-row: span 2;
    align-self: start;
    min-width: 96px;
  }
  .total {
    font-size: 30px;
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1;
  }
  small {
    margin-left: 3px;
    font-size: 11px;
    color: var(--text-3);
    font-family: var(--font);
    letter-spacing: 0;
  }
  .bar {
    display: flex;
    gap: 2px;
    height: 30px;
  }
  .bar.empty {
    border: 1px dashed var(--line-strong);
    border-radius: var(--r-ctl);
  }
  .seg {
    flex-basis: 0;
    min-width: 3px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    padding: 0 7px;
    overflow: hidden;
    background: var(--raised);
    border: 1px solid var(--line);
    font-size: 11px;
    color: var(--text-2);
    transition: flex-grow 0.4s var(--ease);
  }
  .seg:first-child {
    border-radius: var(--r-ctl) 0 0 var(--r-ctl);
  }
  .seg:last-child {
    border-radius: 0 var(--r-ctl) var(--r-ctl) 0;
  }
  .seg.gpu {
    background: var(--accent-soft);
    border-color: color-mix(in srgb, var(--accent) 45%, transparent);
    color: var(--text);
  }
  .seg-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .seg-ms {
    color: var(--text);
  }
  .facts {
    display: flex;
    gap: 26px;
    margin: 0;
  }
  .facts div {
    display: grid;
    gap: 1px;
  }
  dt {
    font-size: 11px;
    color: var(--text-3);
  }
  dd {
    margin: 0;
    font-size: 13px;
  }
  .facts .warn dd {
    color: var(--warn);
  }
</style>
