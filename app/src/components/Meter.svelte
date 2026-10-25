<script lang="ts">
  // Vertical peak/RMS meter, -60..0 dBFS, with 1.2 s peak hold. Values arrive at 30 Hz; the bar is
  // eased per animation frame (fast attack, ~20 dB/s release) so it reads like a hardware meter.
  interface Props {
    peak: number; // linear
    rms: number; // linear
    label: string;
  }
  let { peak, rms, label }: Props = $props();

  const FLOOR = -60;
  const db = (x: number) => (x > 1e-6 ? 20 * Math.log10(x) : FLOOR);
  const frac = (d: number) => Math.min(1, Math.max(0, (d - FLOOR) / -FLOOR));

  let shownPeak = $state(0);
  let shownRms = $state(0);
  let hold = $state(0);
  let holdAt = 0;

  $effect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      const fall = (20 * dt) / -FLOOR;
      const tp = frac(db(peak));
      const tr = frac(db(rms));
      shownPeak = tp > shownPeak ? tp : Math.max(tp, shownPeak - fall);
      shownRms = tr > shownRms ? tr : Math.max(tr, shownRms - fall);
      if (tp >= hold || now - holdAt > 1200) {
        hold = tp;
        holdAt = now;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  });

  let clip = $derived(peak >= 0.989);
</script>

<div class="meter" role="meter" aria-label={label} aria-valuemin={FLOOR} aria-valuemax={0} aria-valuenow={Math.round(db(peak))}>
  <div class="col">
    <div class="rms" style:clip-path="inset({(1 - shownRms) * 100}% 0 0 0)"></div>
    <div class="peak" style:clip-path="inset({(1 - shownPeak) * 100}% 0 0 0)"></div>
    <div class="hold" class:clip style:bottom="{hold * 100}%"></div>
  </div>
  <span class="name">{label}</span>
</div>

<style>
  .meter {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    height: 100%;
    min-height: 0;
  }
  .col {
    position: relative;
    flex: 1;
    width: 10px;
    border-radius: 2px;
    overflow: hidden;
    background:
      repeating-linear-gradient(to top, transparent 0 3px, var(--panel) 3px 4px),
      var(--inset);
    box-shadow: inset 0 0 0 1px var(--line);
  }
  .rms,
  .peak {
    position: absolute;
    inset: 0;
    /* green to -12 dB (80 %), amber to -3 dB (95 %), red above */
    background: linear-gradient(to top, var(--ok) 0 80%, var(--warn) 80% 95%, var(--bad) 95%);
    -webkit-mask: repeating-linear-gradient(to top, #000 0 3px, transparent 3px 4px);
    mask: repeating-linear-gradient(to top, #000 0 3px, transparent 3px 4px);
  }
  .peak {
    opacity: 0.38;
  }
  .hold {
    position: absolute;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--text-2);
  }
  .hold.clip {
    background: var(--bad);
  }
  .name {
    font: 600 10px var(--mono);
    color: var(--text-3);
    letter-spacing: 0.06em;
  }
</style>
