<script lang="ts">
  import { store, TRACE_FRAMES } from "../lib/store.svelte";

  const F_MIN = 55; // A1
  const F_MAX = 1320; // E6
  const NOTES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

  let canvas: HTMLCanvasElement | undefined = $state();
  let box: HTMLDivElement | undefined = $state();

  function note(f: number) {
    if (f <= 0) return null;
    const m = 69 + 12 * Math.log2(f / 440);
    const r = Math.round(m);
    const cents = Math.round((m - r) * 100);
    return { name: `${NOTES[((r % 12) + 12) % 12]}${Math.floor(r / 12) - 1}`, cents, hz: Math.round(f) };
  }
  function lastVoiced(a: Float32Array) {
    for (let i = a.length - 1; i >= a.length - 30; i--) if (a[i] > 0) return a[i];
    return 0;
  }

  let version = $derived(store.traceVersion);
  let inNote = $derived.by(() => {
    void version;
    return note(lastVoiced(store.traceIn));
  });
  let outNote = $derived.by(() => {
    void version;
    return note(lastVoiced(store.traceOut));
  });
  let idle = $derived(!store.live);

  $effect(() => {
    if (!canvas || !box) return;
    const ctx = canvas.getContext("2d")!;
    let w = 0, h = 0, dpr = 1;
    const ro = new ResizeObserver(() => {
      dpr = window.devicePixelRatio || 1;
      w = box!.clientWidth;
      h = box!.clientHeight;
      canvas!.width = Math.round(w * dpr);
      canvas!.height = Math.round(h * dpr);
    });
    ro.observe(box);

    let seen = -1, chunkAt = performance.now(), chunk = 10, raf = 0;
    const y = (f: number) => {
      const t = Math.log(f / F_MIN) / Math.log(F_MAX / F_MIN);
      return h - 18 - t * (h - 30);
    };

    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (!w || !h) return;
      if (store.traceVersion !== seen) {
        seen = store.traceVersion;
        chunk = Math.max(1, Math.min(20, Math.round((now - chunkAt) / 10)));
        chunkAt = now;
      }
      const css = getComputedStyle(box!);
      const c = (v: string) => css.getPropertyValue(v).trim();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Octave grid (C1..C6) with labels.
      ctx.font = `10px ${c("--mono")}`;
      ctx.textBaseline = "middle";
      for (let oct = 1; oct <= 6; oct++) {
        const f = 32.703 * 2 ** (oct - 1);
        if (f < F_MIN || f > F_MAX) continue;
        const yy = Math.round(y(f)) + 0.5;
        ctx.strokeStyle = c("--line");
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(30, yy);
        ctx.lineTo(w, yy);
        ctx.stroke();
        ctx.fillStyle = c("--text-3");
        ctx.fillText(`C${oct}`, 6, yy);
      }
      if (idle) return;

      // Scroll continuously, one chunk behind the newest data.
      const drift = Math.min(chunk, ((now - chunkAt) / 1000) * 100);
      const x = (i: number) => 30 + ((i + chunk - drift) / TRACE_FRAMES) * (w - 30);
      const line = (a: Float32Array, color: string, width: number) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineJoin = "round";
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < TRACE_FRAMES; i++) {
          const f = a[i];
          if (f <= 0 || f < F_MIN * 0.9 || f > F_MAX * 1.1) {
            pen = false;
            continue;
          }
          const px = x(i), py = y(f);
          if (pen) ctx.lineTo(px, py);
          else ctx.moveTo(px, py);
          pen = true;
        }
        ctx.stroke();
      };
      ctx.save();
      ctx.beginPath();
      ctx.rect(30, 0, w - 30, h);
      ctx.clip();
      line(store.traceIn, c("--text-3"), 1.5);
      line(store.traceOut, c("--accent"), 2);
      ctx.restore();
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  });
</script>

<div class="trace" bind:this={box}>
  <canvas bind:this={canvas} aria-label="Input and output pitch over the last five seconds"></canvas>
  <div class="legend">
    <div class="in">
      <span class="sw" aria-hidden="true"></span>
      <span class="label">In</span>
      <span class="num">{inNote ? `${inNote.name}  ${inNote.hz} Hz` : "-"}</span>
    </div>
    <div class="out">
      <span class="sw" aria-hidden="true"></span>
      <span class="label">Out</span>
      <span class="num">{outNote ? `${outNote.name}  ${outNote.hz} Hz` : "-"}</span>
    </div>
  </div>
  {#if idle}
    <p class="empty">The pitch trace runs while the voice is live.</p>
  {/if}
</div>

<style>
  .trace {
    position: relative;
    min-height: 0;
    height: 100%;
    border: 1px solid var(--line);
    border-radius: var(--r-panel);
    background: var(--inset);
    overflow: hidden;
  }
  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }
  .legend {
    position: absolute;
    top: 8px;
    right: 10px;
    display: flex;
    gap: 14px;
    padding: 4px 8px;
    border-radius: var(--r-ctl);
    background: color-mix(in srgb, var(--inset) 85%, transparent);
  }
  .legend > div {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend .num {
    color: var(--text-2);
    white-space: pre;
    min-width: 92px;
  }
  .sw {
    width: 12px;
    height: 2px;
    background: var(--text-3);
  }
  .out .sw {
    background: var(--accent);
    height: 3px;
  }
  .empty {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    margin: 0;
    color: var(--text-3);
  }
</style>
