<script lang="ts">
  // Horizontal hardware-style fader. Drag, wheel, arrow keys (Shift = fine, PageUp/Down = coarse),
  // double-click resets. Bipolar faders fill from the origin and have a detent there.
  interface Props {
    value: number;
    min: number;
    max: number;
    step: number;
    defaultValue?: number;
    origin?: number;
    ticks?: { at: number; label?: string }[];
    label: string;
    valueText?: string;
    size?: "lg" | "md";
    disabled?: boolean;
    /** Wheel adjusts the value (off inside scrolling panels). */
    wheel?: boolean;
    onchange: (v: number) => void;
  }
  let {
    value, min, max, step, defaultValue = min, origin, ticks = [], label, valueText,
    size = "md", disabled = false, wheel: useWheel = true, onchange,
  }: Props = $props();

  let track: HTMLDivElement | undefined = $state();
  let dragging = $state(false);

  const pct = (v: number) => ((v - min) / (max - min)) * 100;
  const decimals = $derived(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  function snap(v: number) {
    const s = Math.round((v - min) / step) * step + min;
    let c = Math.min(max, Math.max(min, Number(s.toFixed(decimals))));
    // Detent: values within 1.5 % of the range around the origin stick to it.
    if (origin !== undefined && Math.abs(c - origin) < (max - min) * 0.015) c = origin;
    return c;
  }
  function set(v: number) {
    const c = snap(v);
    if (c !== value) onchange(c);
  }
  function fromPointer(e: PointerEvent) {
    if (!track) return;
    const r = track.getBoundingClientRect();
    set(min + ((e.clientX - r.left) / r.width) * (max - min));
  }
  function down(e: PointerEvent) {
    if (disabled || e.button !== 0) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragging = true;
    fromPointer(e);
  }
  function key(e: KeyboardEvent) {
    if (disabled) return;
    const big = (max - min) / 8;
    const d: Record<string, number> = {
      ArrowRight: step, ArrowUp: step, ArrowLeft: -step, ArrowDown: -step, PageUp: big, PageDown: -big,
    };
    if (e.key in d) set(value + d[e.key] * (e.shiftKey && !e.key.startsWith("Page") ? 0.25 : 1));
    else if (e.key === "Home") set(min);
    else if (e.key === "End") set(max);
    else return;
    e.preventDefault();
  }
  function wheel(e: WheelEvent) {
    if (disabled || !useWheel) return;
    e.preventDefault();
    set(value + (e.deltaY < 0 ? step : -step));
  }

  let o = $derived(pct(origin ?? min));
  let p = $derived(pct(value));
</script>

<div
  class="fader {size}"
  class:dragging
  class:disabled
  role="slider"
  tabindex={disabled ? -1 : 0}
  aria-label={label}
  aria-valuemin={min}
  aria-valuemax={max}
  aria-valuenow={value}
  aria-valuetext={valueText}
  aria-disabled={disabled}
  onpointerdown={down}
  onpointermove={(e) => dragging && fromPointer(e)}
  onpointerup={() => (dragging = false)}
  onpointercancel={() => (dragging = false)}
  onkeydown={key}
  onwheel={wheel}
  ondblclick={() => !disabled && set(defaultValue)}
>
  <div class="track" bind:this={track}>
    {#each ticks as t (t.at)}
      <span class="tick" class:major={t.label !== undefined} style:left="{pct(t.at)}%"></span>
    {/each}
    <span class="fill" style:left="{Math.min(o, p)}%" style:width="{Math.abs(p - o)}%"></span>
    <span class="cap" style:left="{p}%"></span>
  </div>
  {#if ticks.some((t) => t.label)}
    <div class="scale" aria-hidden="true">
      {#each ticks.filter((t) => t.label !== undefined) as t (t.at)}
        <span style:left="{pct(t.at)}%">{t.label}</span>
      {/each}
    </div>
  {/if}
</div>

<style>
  .fader {
    --h: 22px;
    position: relative;
    padding: 4px 8px 0;
    touch-action: none;
    cursor: ew-resize;
    border-radius: var(--r-ctl);
  }
  .fader.lg {
    --h: 34px;
  }
  .fader.disabled {
    opacity: 0.45;
    cursor: default;
  }
  .track {
    position: relative;
    height: var(--h);
  }
  .track::before {
    /* the slot */
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    height: 4px;
    transform: translateY(-50%);
    border-radius: 2px;
    background: var(--inset);
    box-shadow: inset 0 0 0 1px var(--line-strong);
  }
  .tick {
    position: absolute;
    top: 3px;
    bottom: 3px;
    width: 1px;
    background: var(--line-strong);
    transform: translateX(-0.5px);
  }
  .tick.major {
    top: 0;
    bottom: 0;
    background: var(--text-3);
  }
  .fill {
    position: absolute;
    top: 50%;
    height: 4px;
    transform: translateY(-50%);
    background: var(--accent);
    border-radius: 2px;
  }
  .cap {
    position: absolute;
    top: 50%;
    width: 12px;
    height: calc(var(--h) - 2px);
    transform: translate(-50%, -50%);
    border-radius: 3px;
    background: linear-gradient(var(--raised), var(--panel));
    border: 1px solid var(--line-strong);
    box-shadow: 0 2px 6px rgb(0 0 0 / 0.35);
    transition: border-color 0.15s;
  }
  .cap::after {
    content: "";
    position: absolute;
    left: 50%;
    top: 3px;
    bottom: 3px;
    width: 2px;
    transform: translateX(-50%);
    background: var(--accent);
    border-radius: 1px;
  }
  .fader:hover:not(.disabled) .cap,
  .fader:focus-visible .cap,
  .dragging .cap {
    border-color: var(--text-3);
  }
  .fader:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .scale {
    position: relative;
    height: 16px;
    margin-top: 2px;
  }
  .scale span:first-child {
    transform: translateX(-4px) !important;
  }
  .scale span:last-child {
    transform: translateX(calc(-100% + 4px)) !important;
  }
  .scale span {
    position: absolute;
    transform: translateX(-50%);
    font: 10.5px var(--mono);
    color: var(--text-3);
  }
</style>
