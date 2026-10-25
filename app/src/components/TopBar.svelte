<script lang="ts">
  import Gear from "phosphor-svelte/lib/Gear";
  import Microphone from "phosphor-svelte/lib/Microphone";
  import MicrophoneSlash from "phosphor-svelte/lib/MicrophoneSlash";
  import Logo from "./Logo.svelte";
  import { store } from "../lib/store.svelte";

  const LABEL: Record<string, string> = {
    stopped: "Stopped",
    starting: "Starting engine",
    loading: "Loading voice",
    running: "Live",
    restarting: "Engine restarting",
    device_lost: "Device lost",
    failed: "Engine failed",
  };

  let state = $derived(store.calibrating ? "calibrating" : store.engine.state);
  let label = $derived(store.calibrating ? "Measuring GPU" : LABEL[store.engine.state]);
  let tone = $derived(
    state === "running" ? (store.recentUnderruns > 0 ? "warn" : "ok")
    : state === "failed" ? "bad"
    : state === "restarting" || state === "device_lost" ? "warn"
    : state === "stopped" ? "idle" : "busy",
  );
  let latency = $derived(store.running ? store.stats?.latency?.total_ms : undefined);
  let mutedKey = $derived(store.settings.hotkeys.mute);
</script>

<header class="bar" data-tauri-drag-region>
  <div class="brand" data-tauri-drag-region>
    <Logo />
    <span class="word">Resona</span>
  </div>

  <div class="status" data-tone={tone} role="status" aria-live="polite">
    <span class="lamp" aria-hidden="true"></span>
    <span>{label}</span>
    {#if latency !== undefined}
      <span class="sep" aria-hidden="true"></span>
      <span class="num">{Math.round(latency)} ms</span>
    {/if}
  </div>

  <div class="right">
    <button
      class="rec"
      class:on={store.recording !== null}
      aria-pressed={store.recording !== null}
      disabled={!store.running && store.recording === null}
      title={store.recording !== null
        ? "Stop and save the take"
        : store.running
          ? "Record an A/B take: your raw input and the converted output, sample-aligned, into Music\\Resona"
          : "Start the voice changer to record"}
      onclick={() => store.toggleRecording()}
    >
      <span class="dot" aria-hidden="true"></span>
      {#if store.recording !== null}
        <span class="num">{Math.floor(store.recording / 60)}:{String(Math.floor(store.recording % 60)).padStart(2, "0")}</span>
      {:else}
        <span>Rec</span>
      {/if}
    </button>
    <button
      class="mute"
      class:on={store.muted}
      aria-pressed={store.muted}
      title={mutedKey ? `Mute (${mutedKey})` : "Mute"}
      onclick={() => store.toggleMute()}
    >
      {#if store.muted}<MicrophoneSlash size={16} weight="bold" />{:else}<Microphone size={16} />{/if}
      <span>{store.muted ? "Muted" : "Mute"}</span>
    </button>
    <button
      class="btn ghost icon"
      aria-label="Settings"
      title="Settings"
      onclick={() => (store.settingsOpen = !store.settingsOpen)}
    >
      <Gear size={17} />
    </button>
  </div>
</header>

<style>
  .bar {
    grid-area: bar;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    padding: 0 10px 0 14px;
    border-bottom: 1px solid var(--line);
    background: var(--panel);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 9px;
  }
  .word {
    font-weight: 650;
    letter-spacing: -0.01em;
    font-size: 14px;
  }
  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 28px;
    padding: 0 12px;
    border: 1px solid var(--line);
    border-radius: var(--r-ctl);
    background: var(--inset);
    color: var(--text-2);
    font-weight: 500;
  }
  .lamp {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-3);
  }
  .status[data-tone="ok"] .lamp {
    background: var(--ok);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent);
  }
  .status[data-tone="ok"] {
    color: var(--text);
  }
  .status[data-tone="warn"] .lamp {
    background: var(--warn);
  }
  .status[data-tone="bad"] .lamp {
    background: var(--bad);
  }
  .status[data-tone="busy"] .lamp {
    background: var(--accent);
    animation: blink 1s steps(2, jump-none) infinite;
  }
  @keyframes blink {
    50% {
      opacity: 0.25;
    }
  }
  .sep {
    width: 1px;
    height: 12px;
    background: var(--line-strong);
  }
  .right {
    display: flex;
    justify-content: flex-end;
    gap: 6px;
  }
  .rec {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 30px;
    min-width: 66px;
    padding: 0 11px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-ctl);
    background: var(--raised);
    font-weight: 600;
    transition: border-color 0.15s, color 0.15s;
  }
  .rec:hover:not(:disabled) {
    border-color: var(--text-3);
  }
  .rec:disabled {
    color: var(--text-3);
    cursor: default;
  }
  .rec .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    border: 1.5px solid currentColor;
    flex: none;
  }
  .rec:not(:disabled) .dot {
    border-color: var(--bad);
  }
  .rec.on {
    border-color: color-mix(in srgb, var(--bad) 60%, var(--line));
  }
  .rec.on .dot {
    background: var(--bad);
    border-color: var(--bad);
    animation: rec 1.2s ease-in-out infinite;
  }
  @keyframes rec {
    50% {
      opacity: 0.35;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .rec.on .dot {
      animation: none;
    }
  }
  .mute {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 12px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-ctl);
    background: var(--raised);
    font-weight: 600;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
  }
  .mute:active {
    transform: translateY(1px);
  }
  .mute.on {
    background: color-mix(in srgb, var(--bad) 80%, #000);
    border-color: transparent;
    color: #fff;
  }
</style>
