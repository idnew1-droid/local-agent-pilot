@echo off
cd /d %~dp0
rem LocalDesk launcher.
rem Ollama runs only while this tool runs: if the server is not up,
rem we start it, mark that we own it, and stop it again on exit.
rem If it was already running (started by something else), we leave it alone.
set LOCAL_AGENT_NUM_GPU=32
set LOCALDESK_OWN_OLLAMA=
ollama list >nul 2>nul
if errorlevel 1 (
  set LOCALDESK_OWN_OLLAMA=1
  start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
  timeout /t 6 >nul
)
python app.py
if "%LOCALDESK_OWN_OLLAMA%"=="1" taskkill /F /IM ollama.exe >nul 2>nul
pause
