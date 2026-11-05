param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $RepoRoot
try {
    & $Python -m venv .venv-tatr
    if ($LASTEXITCODE -ne 0) { throw "Could not create TATR environment" }
    & .\.venv-tatr\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "Could not install CPU torch" }
    & .\.venv-tatr\Scripts\python.exe -m pip install -r requirements-tatr-lock.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install TATR dependencies" }
    & $Python -m venv .venv-ocr
    if ($LASTEXITCODE -ne 0) { throw "Could not create OCR environment" }
    & .\.venv-ocr\Scripts\python.exe -m pip install -r requirements-ocr-lock.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install OCR dependencies" }
    Write-Host "Ready. Run prepare_model.py if you do not already have the local TATR checkpoint."
} finally {
    Pop-Location
}
