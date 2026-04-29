param(
  [string]$Symbols = "",
  [string]$Timeframe = "",
  [switch]$NoDashboard,
  [switch]$NoMaintenance
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
    try { return (Get-Command $candidate -ErrorAction Stop).Source } catch {}
  }
  throw "No Python executable found. Run .tools\install.cmd first."
}

$Python = Resolve-Python
$env:PYTHONPATH = (Join-Path $Root "src")
$env:PYTHONIOENCODING = "utf-8"
$env:LOKY_MAX_CPU_COUNT = "4"

$argsList = @("src\autonomous_runner.py")
if ($Symbols.Trim().Length -gt 0) { $argsList += "--symbols"; $argsList += ($Symbols -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
if ($Timeframe.Trim().Length -gt 0) { $argsList += "--timeframe"; $argsList += $Timeframe }
if ($NoDashboard) { $argsList += "--no-dashboard" }
if ($NoMaintenance) { $argsList += "--no-maintenance" }

Write-Host "[run] Starting autonomous platform. Press Ctrl+C to stop all child processes."
Write-Host "[run] Dashboard is launched automatically unless -NoDashboard is used."
& $Python @argsList
