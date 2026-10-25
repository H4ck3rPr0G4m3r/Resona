<#
.SYNOPSIS
  Run Resona in development mode: Vite hot reload for the UI, debug build of the Rust shell.
  Needs scripts\setup.ps1 to have created engine\.venv first.

.PARAMETER Mock
  Only serve the UI in a browser with a mocked backend (no engine, no audio):
  http://localhost:5173/?state=running&fx  (states: running, stopped, failed, loading,
  device_lost, restarting, underruns; add &empty for an empty library, &file for file input).
#>
param([switch]$Mock)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location (Join-Path $root "app")
try {
    if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund }
    if ($Mock) { npm run dev } else { npx tauri dev }
} finally { Pop-Location }
