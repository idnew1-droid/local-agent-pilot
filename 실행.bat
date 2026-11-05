@echo off
cd /d %~dp0
rem LocalDesk launcher - starts Ollama if needed, then opens the UI.
set LOCAL_AGENT_NUM_GPU=32
ollama list >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
  timeout /t 6 >nul
)
python app.py
pause
