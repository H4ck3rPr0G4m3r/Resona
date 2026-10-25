<script lang="ts">
  // One generated control for an effect parameter (from the engine's catalog).
  import Fader from "./Fader.svelte";
  import type { ParamSpec } from "../lib/api";
  import { formatParam as format } from "../lib/fxformat";

  let { spec, value, onchange }: { spec: ParamSpec; value: number; onchange: (v: number) => void } = $props();

  const isLog = $derived(spec.scale === "log");
  const choices = $derived(typeof spec.scale === "object" ? spec.scale.choice : null);

  // Log parameters move the fader in log space, so each octave/decade gets equal travel.
  const toPos = (v: number) => (isLog ? Math.log(v) : v);
  const fromPos = (x: number) => (isLog ? Math.exp(x) : x);
  const lo = $derived(toPos(spec.min));
  const hi = $derived(toPos(spec.max));
  const step = $derived((hi - lo) / (isLog ? 400 : spec.max - spec.min > 50 ? 200 : 100));
  const origin = $derived(spec.min < 0 && spec.max > 0 ? 0 : undefined);
</script>

<div class="pc">
  <div class="head">
    <span class="label">{spec.name}</span>
    {#if !choices}<span class="num v">{format(spec, value)}</span>{/if}
  </div>
  {#if choices}
    <div class="seg" role="group" aria-label={spec.name}>
      {#each choices as c, i (c)}
        <button aria-pressed={Math.round(value) === i} onclick={() => onchange(i)}>{c}</button>
      {/each}
    </div>
  {:else}
    <Fader
      label={spec.name}
      value={toPos(value)}
      min={lo}
      max={hi}
      {step}
      defaultValue={toPos(spec.default)}
      {origin}
      wheel={false}
      valueText={format(spec, value)}
      onchange={(x) => onchange(Number(fromPos(x).toPrecision(5)))}
    />
  {/if}
</div>

<style>
  .pc {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0 8px;
    gap: 8px;
  }
  .v {
    font-size: 12px;
    color: var(--text-2);
    white-space: nowrap;
  }
  .seg {
    margin: 4px 8px 0;
    align-self: start;
    justify-self: start;
  }
</style>
