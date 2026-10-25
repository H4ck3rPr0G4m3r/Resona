<#
.SYNOPSIS
  One-time setup for Resona: engine environment, model assets, app build, Start Menu shortcut.

.DESCRIPTION
  Safe to re-run; every step skips work that is already done.
    1. checks for uv, Node.js and Rust
    2. creates engine\.venv with the pinned Python + CUDA PyTorch (uv sync)
    3. downloads the feature-extractor weights (ContentVec, RMVPE; about 550 MB)
    4. installs the UI packages and builds target\release\resona.exe
    5. adds a "Resona" shortcut to the Start Menu (skip with -NoShortcut)

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#>
param(
    [switch]$NoShortcut,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root "engine"
$app = Join-Path $root "app"
$exe = Join-Path $root "target\release\resona.exe"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`n$text" -ForegroundColor Red; exit 1 }
function Run($file, [string[]]$arguments, $dir) {
    Push-Location $dir
    try {
        & $file @arguments
        if ($LASTEXITCODE -ne 0) { Fail "'$file $($arguments -join ' ')' failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
}

Step "Checking tools"
$missing = @()
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { $missing += "uv      -> powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { $missing += "Node.js -> https://nodejs.org (LTS)" }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { $missing += "Rust    -> https://rustup.rs (needs the MSVC build tools)" }
if ($missing.Count -gt 0) { Fail ("Install these first, then re-run:`n  " + ($missing -join "`n  ")) }
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = (& nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    Write-Host "GPU: $gpu"
} else {
    Write-Host "No nvidia-smi found. Resona needs an NVIDIA GPU with a recent driver (CUDA 12.8)." -ForegroundColor Yellow
}

Step "Engine environment (Python 3.12 + PyTorch CUDA; the first run downloads ~3 GB)"
Run "uv" @("sync", "--frozen") $engine

Step "Model assets (ContentVec + RMVPE)"
Run "uv" @("run", "--frozen", "resona-engine", "assets", "fetch") $engine

if (-not $SkipBuild) {
    Step "UI packages"
    if (Test-Path (Join-Path $app "node_modules")) {
        Run "npm" @("install", "--no-audit", "--no-fund") $app
    } else {
        Run "npm" @("ci", "--no-audit", "--no-fund") $app
    }

    Step "Building Resona (release; the first build takes a few minutes)"
    Run "npx" @("tauri", "build") $app
    if (-not (Test-Path $exe)) { Fail "Build finished but $exe is missing." }
}

if (-not $NoShortcut -and (Test-Path $exe)) {
    Step "Start Menu shortcut"
    $lnk = Join-Path ([Environment]::GetFolderPath("Programs")) "Resona.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = $exe
    $s.WorkingDirectory = $root
    $s.IconLocation = "$exe,0"
    $s.Description = "Resona real-time voice changer"
    $s.Save()
    Write-Host "Created $lnk"
}

Write-Host "`nDone. Start Resona from the Start Menu, by double-clicking Resona.cmd, or with:" -ForegroundColor Green
Write-Host "  $exe"
