$ErrorActionPreference = "Stop"
$python = "C:\Python314\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python -m PyInstaller --noconfirm --clean .\s2cool_app.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE." }
Write-Host "Build complete: $PWD\dist\S2Cool\S2Cool.exe"
