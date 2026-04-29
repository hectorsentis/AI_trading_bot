param(
  [switch]$SkipPip
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Resolve-Python {
  $candidates = @(
    (Join-Path $Root ".tools\py314\python.exe"),
    (Join-Path $Root ".tools\python.exe"),
    (Join-Path $Root ".venv\Scripts\python.exe"),
    "python",
    "py"
  )
  foreach ($candidate in $candidates) {
    try {
      $cmd = Get-Command $candidate -ErrorAction Stop
      return $cmd.Source
    } catch {}
  }
  throw "No Python executable found."
}

$Python = Resolve-Python
Write-Host "[install] Python: $Python"
New-Item -ItemType Directory -Force -Path data, data\raw, data\db, logs, logs\services, logs\dashboard, models, reports | Out-Null

if (!(Test-Path .env) -and (Test-Path .env.example)) {
  Copy-Item .env.example .env
  Write-Host "[install] Created .env from .env.example. Fill API keys before Demo Mode execution."
}

if (-not $SkipPip) {
  Write-Host "[install] Installing requirements..."
  & $Python -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "pip install failed. Re-run with a Python that has pip, or use -SkipPip if dependencies already exist."
  }
}

$env:PYTHONPATH = (Join-Path $Root "src")
$env:PYTHONIOENCODING = "utf-8"
$env:LOKY_MAX_CPU_COUNT = "4"
Write-Host "[install] Initializing SQLite schema..."
& $Python src\install_setup.py --json
if ($LASTEXITCODE -ne 0) { throw "install_setup failed" }
Write-Host "[install] Done. Start everything with: .tools\run.cmd"
