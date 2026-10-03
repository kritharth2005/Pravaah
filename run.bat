@echo off
setlocal EnableExtensions
rem Pravaah local launcher for Windows.
rem   run.bat          set up whatever is missing, then start the backend and frontend
rem   run.bat check    only report what is missing; changes nothing

cd /d "%~dp0"
set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend\Legal-Summarizer"
set "PYTHON=%BACKEND%\.venv\Scripts\python.exe"
set "OLLAMA_URL=http://localhost:11434"
set "CHECK_ONLY="
if /i "%~1"=="check" set "CHECK_ONLY=1"
set "MISSING=0"
set "RESULT=0"

echo.
echo  Pravaah local setup
echo  -------------------

rem ---------------------------------------------------------------- tools
call :require ollama "Ollama: https://ollama.com/download" || goto :failed
call :require uv "uv: https://docs.astral.sh/uv/getting-started/installation/" || goto :failed
call :require npm "Node.js 20 or later: https://nodejs.org" || goto :failed
call :require curl "curl, included with Windows 10 and later" || goto :failed
where tesseract >nul 2>&1 && call :ok "Tesseract OCR found" || call :warn "Tesseract not found: image uploads and scanned PDFs will be rejected; text PDFs and pasted text still work"

rem ---------------------------------------------------------------- ollama
curl -s -o nul "%OLLAMA_URL%/api/version" && goto :ollama_up
if defined CHECK_ONLY (call :missing "Ollama is not running" & goto :backend)
echo  Starting Ollama...
start "Ollama" /min ollama serve
call :wait_for "%OLLAMA_URL%/api/version" 30 || (call :fail "Ollama did not start. Run 'ollama serve' to see why." & goto :failed)
:ollama_up
call :ok "Ollama is running"
call :model nomic-embed-text || goto :failed
call :model hermes3:8b || goto :failed

rem ---------------------------------------------------------------- backend
:backend
if exist "%PYTHON%" goto :backend_update
if defined CHECK_ONLY (call :missing "Backend environment is not installed" & goto :frontend)
echo  Creating the backend environment...
pushd "%BACKEND%"
uv sync --frozen
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (call :fail "uv sync failed" & goto :failed)
goto :backend_ready

:backend_update
if defined CHECK_ONLY goto :backend_verify
rem Install into the existing environment rather than `uv sync`: after an in-place Python upgrade,
rem uv recreates .venv, and that fails halfway through if an editor holds files in it open.
pushd "%BACKEND%"
uv export --frozen --no-hashes --no-emit-project --quiet -o "%TEMP%\pravaah-requirements.txt" && uv pip sync --quiet --python "%PYTHON%" "%TEMP%\pravaah-requirements.txt"
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (call :fail "Installing backend dependencies failed" & goto :failed)

:backend_verify
"%PYTHON%" -c "import fastapi, langchain_ollama, chromadb, edge_tts" >nul 2>&1 || (call :missing "Backend dependencies are incomplete" & goto :frontend)
:backend_ready
call :ok "Backend dependencies are installed"

rem The index is checked by content: starting the API without one creates an empty database.
pushd "%BACKEND%"
"%PYTHON%" -c "import sys; from vector import get_vector_store; sys.exit(0 if get_vector_store()._collection.count() else 1)" >nul 2>&1
set "RC=%errorlevel%"
popd
if "%RC%"=="0" (call :ok "Vector index is built" & goto :frontend)
if defined CHECK_ONLY (call :missing "Vector index is not built" & goto :frontend)
echo  Building the vector index from 23 statutes. This takes about 40 minutes on CPU, once...
pushd "%BACKEND%"
"%PYTHON%" ingest.py
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (call :fail "Building the index failed; rerun to resume where it stopped" & goto :failed)
call :ok "Vector index is built"

rem ---------------------------------------------------------------- frontend
:frontend
if exist "%FRONTEND%\package.json" goto :frontend_deps
if defined CHECK_ONLY (call :missing "Frontend source is missing: run git submodule update --init" & goto :report)
echo  Fetching the frontend submodule...
git submodule update --init || (call :fail "Could not fetch the frontend submodule" & goto :failed)
:frontend_deps
if exist "%FRONTEND%\node_modules\.bin\vite.cmd" (call :ok "Frontend dependencies are installed" & goto :report)
if defined CHECK_ONLY (call :missing "Frontend dependencies are not installed" & goto :report)
echo  Installing frontend dependencies...
pushd "%FRONTEND%"
call npm install
set "RC=%errorlevel%"
popd
if not "%RC%"=="0" (call :fail "npm install failed" & goto :failed)
call :ok "Frontend dependencies are installed"

rem ---------------------------------------------------------------- start
:report
if not defined CHECK_ONLY goto :start
echo.
if "%MISSING%"=="0" (echo  Everything is set up. Start with: run.bat) else (echo  %MISSING% item^(s^) need setup. run.bat sets them up automatically.& set "RESULT=1")
goto :end

:start
echo.
call :port_in_use 8000 && call :ok "Backend is already running on port 8000" || start "Pravaah API" /d "%BACKEND%" cmd /k .venv\Scripts\python.exe -m uvicorn app:app --port 8000
call :port_in_use 3000 && call :ok "Frontend is already running on port 3000" || start "Pravaah UI" /d "%FRONTEND%" cmd /k npm run dev
call :wait_for "http://127.0.0.1:8000/health" 60 || call :warn "The backend has not answered yet; check the Pravaah API window"
echo.
echo  Pravaah is running. Close the "Pravaah API" and "Pravaah UI" windows to stop it.
echo    App:      http://localhost:3000
echo    API docs: http://127.0.0.1:8000/docs
goto :end

rem ---------------------------------------------------------------- helpers
:ok
echo  [ok] %~1
exit /b 0

:warn
echo  [!!] %~1
exit /b 0

:missing
echo  [--] %~1
set /a MISSING+=1
exit /b 0

:fail
echo  [XX] %~1
exit /b 1

:require
where %~1 >nul 2>&1 && exit /b 0
echo  [XX] %~1 is not installed. Install %~2
exit /b 1

:model
ollama list 2>nul | findstr /b /i /c:"%~1" >nul && (call :ok "Model %~1 is available" & exit /b 0)
if defined CHECK_ONLY (call :missing "Model %~1 is not pulled" & exit /b 0)
echo  Pulling %~1...
ollama pull %~1 || (call :fail "Could not pull %~1" & exit /b 1)
call :ok "Model %~1 is available"
exit /b 0

:port_in_use
netstat -ano | findstr /r /c:":%~1 .*LISTENING" >nul
exit /b %errorlevel%

:wait_for
set /a "WAIT_LEFT=%~2"
:wait_loop
curl -s -f -o nul "%~1" && exit /b 0
set /a WAIT_LEFT-=1
if %WAIT_LEFT% leq 0 exit /b 1
rem ping as a 1-second sleep: timeout fails when input is redirected.
ping -n 2 127.0.0.1 >nul
goto :wait_loop

:failed
set "RESULT=1"
echo.
echo  Setup stopped. Fix the item marked [XX] and run run.bat again.
rem Keep the window open when started by double-click.
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause

:end
endlocal & exit /b %RESULT%
