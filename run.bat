@echo off
setlocal
set "PYTHONUTF8=1"
set "TATR_PYTHON=%~dp0.venv-tatr\Scripts\python.exe"
if not exist "%TATR_PYTHON%" (
    for /d %%D in ("%~dp0..\PubTables-1M*") do (
        if exist "%%~fD\.venv-download\Scripts\python.exe" set "TATR_PYTHON=%%~fD\.venv-download\Scripts\python.exe"
    )
)
if not exist "%TATR_PYTHON%" (
    echo Python environment not found. Follow README.md to run scripts/setup.ps1 first.
    exit /b 1
)
pushd "%~dp0"
"%TATR_PYTHON%" "%~dp0run_table_ocr.py" --interactive %*
set "TATR_EXIT=%ERRORLEVEL%"
popd
exit /b %TATR_EXIT%
