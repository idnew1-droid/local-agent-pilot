@echo off
cd /d %~dp0
rem LocalDesk launcher.
rem Ollama runs only while this tool runs: if the server is not up,
rem we start it, mark that we own it, and stop it again on exit.
rem If it was already running (started by something else), we leave it alone.
rem The up-check must be an HTTP probe: the ollama CLI starts the
rem server by itself, which would hide ownership.
set LOCAL_AGENT_NUM_GPU=32
if not defined OLLAMA_MODELS if exist E:\OllamaModels set OLLAMA_MODELS=E:\OllamaModels
set LOCALDESK_OWN_OLLAMA=
python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2)" >nul 2>nul
if errorlevel 1 (
  set LOCALDESK_OWN_OLLAMA=1
  start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
  timeout /t 3 /nobreak >nul
)
start "" http://127.0.0.1:8765/
python app.py
if defined LOCALDESK_OWN_OLLAMA (
  taskkill /F /IM ollama.exe >nul 2>nul
  taskkill /F /IM "ollama app.exe" >nul 2>nul
)
pause
