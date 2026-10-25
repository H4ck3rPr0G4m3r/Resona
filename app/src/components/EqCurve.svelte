<script lang="ts">
  // Interactive response of the 5-band EQ: drag a band handle (left/right = frequency,
  // up/down = gain), scroll on it for Q, double-click to flatten it.
  import { bandDb } from "../lib/fxformat";

  let { params, onchange }: { params: Record<string, number>; onchange: (key: string, v: number) => void } = $props();

  const BANDS = [
    { f: "f1", g: "g1", q: null, shape: "lowshelf", fmin: 20, fmax: 500 },
    { f: "f2", g: "g2", q: "q2", shape: "peak", fmin: 100, fmax: 2000 },
    { f: "f3", g: "g3", q: "q3", shape: "peak", fmin: 300, fmax: 6000 },
    { f: "f4", g: "g4", q: "q4", shape: "peak", fmin: 1000, fmax: 16000 },
    { f: "f5", g: "g5", q: null, shape: "highshelf", fmin: 2000, fmax: 20000 },
  ] as const;
  const DB = 18;
  const F_LO = 20, F_HI = 20000;

  let w = $state(0), h = $state(0);
  let drag = $state<number | null>(null);
  let svg: SVGSVGElement | undefined = $state();

  const x = (f: number) => (Math.log(f / F_LO) / Math.log(F_HI / F_LO)) * w;
  const fAt = (px: number) => F_LO * (F_HI / F_LO) ** Math.min(1, Math.max(0, px / w));
  const y = (db: number) => h / 2 - (db / DB) * (h / 2 - 8);
  const dbAt = (py: number) => Math.max(-DB, Math.min(DB, ((h / 2 - py) / (h / 2 - 8)) * DB));

  let path = $derived.by(() => {
    if (!w || !h) return "";
    const pts: string[] = [];
    for (let i = 0; i <= 160; i++) {
      const f = F_LO * (F_HI / F_LO) ** (i / 160);
      let db = 0;
      for (const b of BANDS) db += bandDb(b.shape, params[b.f], b.q ? params[b.q] : 0.707, params[b.g], f);
      pts.push(`${i ? "L" : "M"}${x(f).toFixed(1)},${y(Math.max(-DB - 2, Math.min(DB + 2, db))).toFixed(1)}`);
    }
    return pts.join("");
  });

  function move(e: PointerEvent) {
    if (drag === null || !svg) return;
    const r = svg.getBoundingClientRect();
    const b = BANDS[drag];
    const f = Math.min(b.fmax, Math.max(b.fmin, fAt(e.clientX - r.left)));
    onchange(b.f, Number(f.toPrecision(4)));
    onchange(b.g, Math.round(dbAt(e.clientY - r.top) * 10) / 10);
  }
  function wheel(e: WheelEvent, i: number) {
    const b = BANDS[i];
    if (!b.q) return;
    e.preventDefault();
    const q = params[b.q] * (e.deltaY < 0 ? 1.12 : 1 / 1.12);
    onchange(b.q, Number(Math.min(8, Math.max(0.3, q)).toPrecision(3)));
  }
  const grid = [100, 1000, 10000];
</script>

<div class="eq" bind:clientWidth={w} bind:clientHeight={h}>
  <svg bind:this={svg} width={w} height={h} role="img" aria-label="Equalizer response" onpointermove={move} onpointerup={() => (drag = null)}>
    {#each grid as f (f)}
      <line x1={x(f)} x2={x(f)} y1="0" y2={h} class="grid" />
      <text x={x(f) + 4} y={h - 6} class="tick">{f >= 1000 ? `${f / 1000}k` : f}</text>
    {/each}
    {#each [-12, -6, 0, 6, 12] as d (d)}
      <line x1="0" x2={w} y1={y(d)} y2={y(d)} class:zero={d === 0} class="grid" />
    {/each}
    <path d={path} class="curve" />
    {#each BANDS as b, i (b.f)}
      <g
        class="handle"
        class:active={drag === i}
        class:flat={Math.abs(params[b.g]) < 0.05}
        role="slider"
        tabindex="0"
        aria-label="Band {i + 1}"
        aria-valuenow={params[b.g]}
        transform="translate({x(params[b.f])},{y(params[b.g])})"
        onpointerdown={(e) => {
          (e.currentTarget as Element).setPointerCapture(e.pointerId);
          drag = i;
        }}
        onwheel={(e) => wheel(e, i)}
        ondblclick={() => onchange(b.g, 0)}
      >
        <circle r="9" class="hit" />
        <circle r="5" />
        <text y="-11" class="num-label">{i + 1}</text>
      </g>
    {/each}
  </svg>
</div>

<style>
  .eq {
    position: relative;
    height: 100%;
    min-height: 140px;
    border: 1px solid var(--line);
    border-radius: var(--r-panel);
    background: var(--inset);
    overflow: hidden;
    touch-action: none;
  }
  svg {
    position: absolute;
    inset: 0;
  }
  .grid {
    stroke: var(--line);
    stroke-width: 1;
  }
  .grid.zero {
    stroke: var(--line-strong);
  }
  .tick {
    font: 10px var(--mono);
    fill: var(--text-3);
  }
  .curve {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2;
  }
  .handle {
    cursor: grab;
  }
  .handle circle:not(.hit) {
    fill: var(--raised);
    stroke: var(--accent);
    stroke-width: 2;
  }
  .handle.flat circle:not(.hit) {
    stroke: var(--text-3);
  }
  .handle.active circle:not(.hit) {
    fill: var(--accent);
  }
  .hit {
    fill: transparent;
  }
  .num-label {
    font: 600 10px var(--mono);
    fill: var(--text-2);
    text-anchor: middle;
  }
</style>
