import "./app.css";
import { mount } from "svelte";
import App from "./App.svelte";

async function boot() {
  // Browser preview of the UI without the Tauri shell (dev server only).
  if (import.meta.env.DEV && !("__TAURI_INTERNALS__" in window && (window as any).__TAURI_INTERNALS__?.invoke)) {
    (await import("./lib/devmock")).install();
  }
  mount(App, { target: document.getElementById("app")! });
}

boot();
