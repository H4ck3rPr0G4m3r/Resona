<script lang="ts">
  // Head of the signal chain: the microphone, or a sound file playing in its place.
  import Microphone from "phosphor-svelte/lib/Microphone";
  import FileAudio from "phosphor-svelte/lib/FileAudio";
  import Play from "phosphor-svelte/lib/Play";
  import Pause from "phosphor-svelte/lib/Pause";
  import Repeat from "phosphor-svelte/lib/Repeat";
  import X from "phosphor-svelte/lib/X";
  import { open } from "@tauri-apps/plugin-dialog";
  import { store } from "../lib/store.svelte";
  import type { DeviceInfo } from "../lib/api";

  const p = $derived(store.player);
  const live = $derived(store.stats?.devices ?? null);
  const mic = $derived.by(() => {
    if (store.live && live) return live.input.device_name;
    const id = store.settings.input_device;
    const list: DeviceInfo[] = store.inputs;
    const d = id ? list.find((x) => x.id === id) : list.find((x) => x.is_default);
    return d?.name ?? (id ? "Unavailable" : "System default");
  });
  const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  async function pick() {
    const path = await open({ filters: [{ name: "Audio", extensions: ["wav", "mp3", "flac"] }] });
    if (typeof path === "string") store.loadSoundFile(path);
  }
  function seek(e: PointerEvent) {
    if (!p || !p.duration_s) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    store.playerControl({ seekS: ((e.clientX - r.left) / r.width) * p.duration_s });
  }
  function seekKey(e: KeyboardEvent) {
    if (!p) return;
    const d = e.key === "ArrowRight" ? 5 : e.key === "ArrowLeft" ? -5 : 0;
    if (d) {
      e.preventDefault();
      store.playerControl({ seekS: Math.max(0, Math.min(p.duration_s, p.position_s + d)) });
    }
  }
  const openDevices = () => {
    store.settingsTab = "devices";
    store.settingsOpen = true;
  };
</script>

{#if p}
  <div class="file" class:playing={p.playing} role="group" aria-label="Sound file input">
    <button class="ic" aria-label={p.playing ? "Pause" : "Play"} onclick={() => store.playerControl({ playing: !p.playing })}>
      {#if p.playing}<Pause size={13} weight="fill" />{:else}<Play size={13} weight="fill" />{/if}
    </button>
    <div class="body">
      <span class="nm" title={p.name}>{p.name}</span>
      <div
        class="track"
        role="slider"
        tabindex="0"
        aria-label="Position"
        aria-valuemin={0}
        aria-valuemax={Math.round(p.duration_s)}
        aria-valuenow={Math.round(p.position_s)}
        aria-valuetext="{clock(p.position_s)} of {clock(p.duration_s)}"
        onpointerdown={seek}
        onkeydown={seekKey}
      >
        <span style:width="{p.duration_s ? (p.position_s / p.duration_s) * 100 : 0}%"></span>
      </div>
    </div>
    <span class="t num">{clock(p.position_s)}</span>
    <button class="ic" class:on={p.looping} aria-pressed={p.looping} aria-label="Loop" title="Loop" onclick={() => store.playerControl({ looping: !p.looping })}>
      <Repeat size={13} />
    </button>
    <button class="ic" aria-label="Back to the microphone" title="Back to the microphone" onclick={() => store.unloadSoundFile()}>
      <X size={13} />
    </button>
  </div>
{:else}
  <button class="node" onclick={openDevices} title="Microphone">
    <Microphone size={15} />
    <span class="dev">{mic}</span>
  </button>
  <button class="pick" aria-label="Play a sound file instead of the microphone" title="Play a sound file instead of the microphone (WAV, MP3, FLAC)" onclick={pick}>
    <FileAudio size={14} />
  </button>
{/if}

<style>
  .node {
    flex: none; /* the strip scrolls; a shrinking node would just lose its name */
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 10px;
    min-width: 0;
    max-width: 210px;
    border: 1px solid var(--line);
    border-radius: var(--r-ctl);
    background: var(--inset);
    color: var(--text-2);
  }
  .node:hover {
    border-color: var(--line-strong);
    color: var(--text);
  }
  .dev,
  .nm {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .pick {
    flex: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: 1px solid var(--line);
    border-radius: var(--r-ctl);
    background: none;
    color: var(--text-3);
  }
  .pick:hover {
    color: var(--text);
    border-color: var(--text-3);
  }
  .file {
    flex: none;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 32px;
    width: 250px;
    padding: 0 3px;
    border: 1px solid color-mix(in srgb, var(--accent) 45%, var(--line));
    border-radius: var(--r-ctl);
    background: var(--inset);
  }
  .ic {
    flex: none;
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    padding: 0;
    border: 0;
    border-radius: 3px;
    background: none;
    color: var(--text-2);
  }
  .ic:hover {
    color: var(--text);
    background: var(--raised);
  }
  .ic.on {
    color: var(--accent);
  }
  .file.playing .ic:first-child {
    color: var(--accent);
  }
  .body {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 3px;
  }
  .nm {
    font-size: 12px;
    line-height: 1;
    font-weight: 500;
  }
  .track {
    position: relative;
    height: 4px;
    border-radius: 2px;
    background: var(--line);
    cursor: pointer;
  }
  .track::before {
    /* a taller hit area than the 4 px line */
    content: "";
    position: absolute;
    inset: -5px 0;
  }
  .track span {
    position: absolute;
    inset: 0 auto 0 0;
    border-radius: 2px;
    background: var(--accent);
  }
  .t {
    flex: none;
    width: 34px;
    text-align: right;
    font-size: 11px;
    color: var(--text-3);
  }
</style>
