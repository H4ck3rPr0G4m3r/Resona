<script lang="ts">
  // Click, then press a combination. Esc cancels, Backspace clears. Bare keys are only accepted for
  // keys that never type text (F1-F24, Pause, Insert, ...), so a hotkey can't eat normal typing.
  interface Props {
    value: string;
    error?: string | null;
    onchange: (accel: string) => void;
  }
  let { value, error = null, onchange }: Props = $props();
  let listening = $state(false);

  const BARE = /^(F\d{1,2}|Pause|ScrollLock|Insert|PrintScreen|Numpad\w+|Media\w+|AudioVolume\w+)$/;
  const keyName = (code: string) =>
    code.startsWith("Key") ? code.slice(3) : code.startsWith("Digit") ? code.slice(5) : code;

  function capture(e: KeyboardEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Escape") return (listening = false);
    if (e.key === "Backspace" || e.key === "Delete") {
      listening = false;
      return onchange("");
    }
    if (["Control", "Alt", "Shift", "Meta"].includes(e.key)) return;
    const mods = [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Super"].filter(Boolean);
    const key = keyName(e.code);
    if (!mods.length && !BARE.test(key)) return;
    listening = false;
    onchange([...mods, key].join("+"));
  }

  $effect(() => {
    if (!listening) return;
    window.addEventListener("keydown", capture, true);
    return () => window.removeEventListener("keydown", capture, true);
  });
</script>

<button
  class="hk"
  class:listening
  class:err={!!error}
  title={error ?? "Click, then press keys. Backspace clears."}
  onclick={() => (listening = !listening)}
  onblur={() => (listening = false)}
>
  {#if listening}
    <span class="wait">Press keys</span>
  {:else if value}
    {#each value.split("+") as part, i (i)}<kbd>{part}</kbd>{/each}
  {:else}
    <span class="none">Not set</span>
  {/if}
</button>
{#if error}<span class="msg">{error}</span>{/if}

<style>
  .hk {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    min-width: 132px;
    height: 30px;
    padding: 0 8px;
    border: 1px solid var(--line-strong);
    border-radius: var(--r-ctl);
    background: var(--inset);
    text-align: left;
  }
  .hk.listening {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .hk.err {
    border-color: var(--bad);
  }
  kbd {
    padding: 1px 5px;
    border: 1px solid var(--line-strong);
    border-bottom-width: 2px;
    border-radius: 3px;
    background: var(--raised);
    font: 11px var(--mono);
  }
  .wait {
    color: var(--accent);
  }
  .none {
    color: var(--text-3);
  }
  .msg {
    display: block;
    margin-top: 4px;
    font-size: 11.5px;
    color: var(--bad);
  }
</style>
