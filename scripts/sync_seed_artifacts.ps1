[CmdletBinding()]
param(
    [string]$SourceRoot = 'E:\Production\AI\Copilot'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Path $PSScriptRoot -Parent
$dataDir = Join-Path $repoRoot 'data\seed'
$evalDir = Join-Path $repoRoot 'evals'

$copies = @(
    [ordered]@{
        source = Join-Path $SourceRoot 'local-agent-training\dataset.jsonl'
        destination = Join-Path $dataDir 'local-agent-training-dataset.jsonl'
    }
    [ordered]@{
        source = Join-Path $SourceRoot 'local-agent-training\dataset-report.json'
        destination = Join-Path $dataDir 'local-agent-training-report.json'
    }
    [ordered]@{
        source = Join-Path $SourceRoot 'local-agent-evals.json'
        destination = Join-Path $evalDir 'local-agent-evals.json'
    }
)

New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
New-Item -ItemType Directory -Path $evalDir -Force | Out-Null

foreach ($copy in $copies) {
    if (-not (Test-Path -LiteralPath $copy.source -PathType Leaf)) {
        throw "Source artifact not found: $($copy.source)"
    }

    Copy-Item -LiteralPath $copy.source -Destination $copy.destination -Force
    Write-Output "Copied $($copy.source) -> $($copy.destination)"
}
