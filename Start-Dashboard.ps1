$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Set up the Python environment first. See docs/dashboard_runbook.md.'
}
if (-not (Test-Path -LiteralPath 'dashboard/dist/index.html')) {
    throw 'Build the dashboard first: npm.cmd ci --prefix dashboard; npm.cmd run build --prefix dashboard'
}
$previousLab = Get-ChildItem -LiteralPath 'runtime' -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'lab-*' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'server/model.json')) } |
    Sort-Object @{ Expression = { ((Get-Content -LiteralPath (Join-Path $_.FullName 'server/model.json') -Raw | ConvertFrom-Json).trees).Count }; Descending = $true }, @{ Expression = { $_.LastWriteTime }; Descending = $true } | Select-Object -First 1
if ($previousLab) {
    & $pythonPath -m service.lab --resume-dir $previousLab.FullName --preview-only --serve
} else {
    & $pythonPath -m service.lab --preview-only --serve
}
