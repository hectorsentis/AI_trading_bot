# Legacy entrypoint kept for compatibility. Use .tools\run.ps1 for the full autonomous platform.
param(
  [string]$Symbols = "",
  [string]$Timeframe = "",
  [switch]$NoDashboard,
  [switch]$NoMaintenance
)
& (Join-Path $PSScriptRoot "run.ps1") -Symbols $Symbols -Timeframe $Timeframe -NoDashboard:$NoDashboard -NoMaintenance:$NoMaintenance
