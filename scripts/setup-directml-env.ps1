[CmdletBinding()]
param(
    [string]$PythonVersion = '3.12',
    [string]$VenvName = '.venv-directml',
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
$venvPath = Join-Path $repoRoot $VenvName
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'
$requirementsPath = Join-Path $repoRoot 'requirements-directml.txt'

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
    throw "requirements file not found: $requirementsPath"
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    & py "-$PythonVersion" -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create venv at $venvPath"
    }
}

& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to upgrade pip in the DirectML venv.'
}

& $pythonExe -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install DirectML experiment requirements.'
}

Write-Output "DirectML venv ready: $venvPath"
Write-Output "Python: $pythonExe"
