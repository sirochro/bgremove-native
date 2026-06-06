@echo off
cd /d "%~dp0"
uv run python -m app.main
if errorlevel 1 (
  echo.
  echo [bgremove-native] exited with error. Press any key to close.
  pause >nul
)
