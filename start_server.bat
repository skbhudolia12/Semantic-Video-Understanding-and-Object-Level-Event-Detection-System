@echo off
echo Stopping existing server...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*" 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do taskkill /F /PID %%a 2>nul

echo Starting server...
cd /d "%~dp0"
python -m uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
