@echo off
rem Launch Resona (built by scripts\setup.ps1).
if exist "%~dp0target\release\resona.exe" (
  start "" "%~dp0target\release\resona.exe"
) else (
  echo Resona is not built yet. Running setup first...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
  if exist "%~dp0target\release\resona.exe" start "" "%~dp0target\release\resona.exe"
  pause
)
