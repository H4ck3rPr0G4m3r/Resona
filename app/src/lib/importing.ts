import { open } from "@tauri-apps/plugin-dialog";
import { store } from "./store.svelte";

export async function pickFiles() {
  const r = await open({
    multiple: true,
    filters: [{ name: "RVC voice", extensions: ["pth", "index", "zip"] }],
  });
  if (r) store.importPaths(Array.isArray(r) ? r : [r]);
}

export async function pickFolder() {
  const r = await open({ directory: true });
  if (typeof r === "string") store.importPaths([r]);
}
