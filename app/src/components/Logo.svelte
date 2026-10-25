<script lang="ts">
  // Resona mark: two offset rings, one becoming the other.
  let { size = 22 }: { size?: number } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
  <circle cx="14.5" cy="12" r="7" fill="var(--accent)" />
  <circle cx="9.5" cy="12" r="6.25" fill="none" stroke="var(--text)" stroke-width="1.5" />
</svg>
