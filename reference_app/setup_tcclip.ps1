[CmdletBinding()]
param(
    [string]$CheckpointSource = $env:TC_CLIP_CHECKPOINT_SOURCE,
    [string]$BackboneSource = $env:TC_CLIP_BACKBONE_SOURCE,
    [string]$CheckpointSha256 = "1c6c01bc65894ca9b32f96e3fce813e98be7f287d723dba1860a2ea2aad9fcb1",
    [string]$BackboneSha256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    [switch]$SkipDependencySync,
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runtimeBase = Join-Path $projectRoot "workspace\third_party\tc_clip"
$runtimeRoot = Join-Path $runtimeBase "tc-clip"
$modelRoot = Join-Path $projectRoot "workspace\models\TC_CLIP"
$checkpointPath = Join-Path $modelRoot "zero_shot_k400_llm_tc_clip.pth"
$backbonePath = Join-Path $modelRoot "ViT-B-16.pt"
$backboneUrl = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
$officialCheckpointPage = "https://drive.google.com/drive/folders/18CWhh4MZ9EvGlJeCHpW48mEfjMTLMCSk"

function Assert-LastCommand {
    param([Parameter(Mandatory = $true)][string]$Operation)

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Get-AssetHash {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Install-ModelAsset {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [AllowEmptyString()][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256,
        [AllowEmptyString()][string]$DefaultUrl,
        [Parameter(Mandatory = $true)][string]$MissingHelp
    )

    $expectedHash = $ExpectedSha256.Trim().ToLowerInvariant()
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $currentHash = Get-AssetHash -Path $Destination
        if ($currentHash -eq $expectedHash) {
            Write-Host "[TC-CLIP] $Label verified: $Destination"
            return
        }
        if ($VerifyOnly) {
            throw "$Label checksum mismatch at $Destination. Expected $expectedHash, got $currentHash."
        }
        Write-Warning "$Label exists but its checksum is invalid; a backup will be retained before replacement."
    }
    elseif ($VerifyOnly) {
        throw "$Label is missing at $Destination. $MissingHelp"
    }

    $effectiveSource = $Source.Trim()
    if (-not $effectiveSource -and $DefaultUrl) {
        $effectiveSource = $DefaultUrl
    }
    if (-not $effectiveSource) {
        throw "$Label is missing. $MissingHelp"
    }

    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    $temporaryPath = "$Destination.download"
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }

    try {
        [System.Uri]$sourceUri = $null
        $isRemote = [System.Uri]::TryCreate($effectiveSource, [System.UriKind]::Absolute, [ref]$sourceUri) -and
            $sourceUri.Scheme -in @("http", "https")

        if ($isRemote) {
            Write-Host "[TC-CLIP] Downloading $Label..."
            Invoke-WebRequest -Uri $sourceUri.AbsoluteUri -OutFile $temporaryPath -UseBasicParsing
        }
        elseif (Test-Path -LiteralPath $effectiveSource -PathType Leaf) {
            Write-Host "[TC-CLIP] Copying $Label from $effectiveSource"
            Copy-Item -LiteralPath $effectiveSource -Destination $temporaryPath
        }
        else {
            throw "$Label source is neither a readable file nor an HTTP(S) URL: $effectiveSource"
        }

        $downloadedHash = Get-AssetHash -Path $temporaryPath
        if ($downloadedHash -ne $expectedHash) {
            throw "$Label checksum mismatch. Expected $expectedHash, got $downloadedHash."
        }

        if (Test-Path -LiteralPath $Destination) {
            $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
            Move-Item -LiteralPath $Destination -Destination "$Destination.invalid-$timestamp.bak"
        }
        Move-Item -LiteralPath $temporaryPath -Destination $Destination
        Write-Host "[TC-CLIP] $Label installed and verified."
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

$requiredRuntimeDirectories = @("clip", "trainers", "utils")
$runtimeReady = (Test-Path -LiteralPath $runtimeRoot -PathType Container) -and
    -not ($requiredRuntimeDirectories | Where-Object { -not (Test-Path -LiteralPath (Join-Path $runtimeRoot $_) -PathType Container) })

if (-not $runtimeReady) {
    throw "TC-CLIP runtime is incomplete at $runtimeRoot. Runtime bersifat vendored (ikut repo di workspace/third_party/tc_clip/tc-clip) - pulihkan dengan 'git checkout -- workspace/third_party/tc_clip', lalu jalankan ulang script ini."
}

$missingRuntimeDirectories = $requiredRuntimeDirectories | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $runtimeRoot $_) -PathType Container)
}
if ($missingRuntimeDirectories) {
    throw "TC-CLIP runtime is incomplete at $runtimeRoot. Missing: $($missingRuntimeDirectories -join ', ')."
}
Write-Host "[TC-CLIP] Runtime verified: $runtimeRoot"

Install-ModelAsset `
    -Label "TC-CLIP checkpoint" `
    -Source $CheckpointSource `
    -Destination $checkpointPath `
    -ExpectedSha256 $CheckpointSha256 `
    -DefaultUrl "" `
    -MissingHelp "Download zero_shot_k400_llm_tc_clip.pth from $officialCheckpointPage, then pass -CheckpointSource <path-or-direct-url>."

Install-ModelAsset `
    -Label "CLIP ViT-B/16 backbone" `
    -Source $BackboneSource `
    -Destination $backbonePath `
    -ExpectedSha256 $BackboneSha256 `
    -DefaultUrl $backboneUrl `
    -MissingHelp "Pass -BackboneSource <path-or-direct-url>."

if (-not $SkipDependencySync -and -not $VerifyOnly) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is not installed or not available in PATH. Install uv, then rerun this script."
    }
    Write-Host "[TC-CLIP] Synchronizing Python dependencies..."
    & uv sync
    Assert-LastCommand -Operation "uv sync"
}

$env:TC_CLIP_SETUP_RUNTIME_ROOT = $runtimeRoot
$smokeCode = "import importlib, os, pathlib, sys; root = pathlib.Path(os.environ['TC_CLIP_SETUP_RUNTIME_ROOT']).resolve(); sys.path.insert(0, str(root)); importlib.import_module('clip'); importlib.import_module('trainers.build_trainer'); importlib.import_module('utils.tools'); print('TC-CLIP runtime import OK: ' + str(root))"
$pythonCandidates = @(
    (Join-Path $projectRoot ".venv\Scripts\python.exe"),
    (Join-Path $projectRoot ".venv/bin/python")
)
$projectPython = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $projectPython) {
    throw "Project virtual environment was not found. Run 'uv sync', then rerun this script."
}
& $projectPython -c $smokeCode
Assert-LastCommand -Operation "TC-CLIP runtime import check"

Write-Host ""
Write-Host "TC-CLIP setup completed. Configure these relative paths in .env:"
Write-Host "TC_CLIP_RUNTIME_ROOT=workspace/third_party/tc_clip/tc-clip"
Write-Host "TC_CLIP_CHECKPOINT_PATH=workspace/models/TC_CLIP/zero_shot_k400_llm_tc_clip.pth"
Write-Host "TC_CLIP_CACHE_DIR=workspace/models/TC_CLIP"
Write-Host "DETECTION_BACKEND=yolo_tcclip"
