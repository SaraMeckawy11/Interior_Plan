$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelDir = Join-Path $ProjectRoot "models\TripoSR"
$ModelPath = Join-Path $ModelDir "model.ckpt"
$PartsDir = Join-Path $ModelDir "_download_parts"
$AssemblingPath = Join-Path $ModelDir "model.ckpt.assembling"
$Url = "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt"
$ExpectedLength = [int64]1677246742
$ExpectedSha256 = "429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
$ParallelParts = 6
$SubPartsPerPart = 3

New-Item -ItemType Directory -Force -Path $ModelDir, $PartsDir | Out-Null
$ResolvedModelDir = (Resolve-Path $ModelDir).Path
if (-not $ResolvedModelDir.StartsWith($ProjectRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside the project workspace."
}

if (Test-Path -LiteralPath $ModelPath) {
    $Existing = Get-Item -LiteralPath $ModelPath
    if ($Existing.Length -eq $ExpectedLength) {
        $Hash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Hash -eq $ExpectedSha256) {
            Write-Output "TripoSR checkpoint is already complete."
            exit 0
        }
    }
}

$PrefixPath = Join-Path $PartsDir "part-000.bin"
if (-not (Test-Path -LiteralPath $PrefixPath)) {
    if (Test-Path -LiteralPath $ModelPath) {
        Move-Item -LiteralPath $ModelPath -Destination $PrefixPath
    } else {
        [IO.File]::WriteAllBytes($PrefixPath, [byte[]]::new(0))
    }
}

$PrefixLength = (Get-Item -LiteralPath $PrefixPath).Length
if ($PrefixLength -ge $ExpectedLength) {
    throw "Existing partial checkpoint is larger than the expected model."
}

$Remaining = $ExpectedLength - $PrefixLength
$PartSize = [int64][Math]::Ceiling($Remaining / [double]$ParallelParts)
$Processes = @()
$SubAssemblyPlans = @()
$AssemblyPlans = @()

for ($Index = 0; $Index -lt $ParallelParts; $Index++) {
    $Start = $PrefixLength + ($Index * $PartSize)
    if ($Start -ge $ExpectedLength) {
        break
    }
    $End = [Math]::Min($ExpectedLength - 1, $Start + $PartSize - 1)
    $PartPath = Join-Path $PartsDir ("part-{0:D3}.bin" -f ($Index + 1))
    $ExpectedPartLength = $End - $Start + 1

    if (-not (Test-Path -LiteralPath $PartPath)) {
        [IO.File]::WriteAllBytes($PartPath, [byte[]]::new(0))
    }
    $ExistingPartLength = (Get-Item -LiteralPath $PartPath).Length
    if ($ExistingPartLength -gt $ExpectedPartLength) {
        throw "A partial checkpoint segment is larger than expected."
    }
    if ($ExistingPartLength -eq $ExpectedPartLength) {
        continue
    }

    $ResumeStart = $Start + $ExistingPartLength
    $ResumeLength = $End - $ResumeStart + 1
    $SubPartSize = [int64][Math]::Ceiling(
        $ResumeLength / [double]$SubPartsPerPart
    )
    $SubPaths = @()

    for ($SubIndex = 0; $SubIndex -lt $SubPartsPerPart; $SubIndex++) {
        $SubStart = $ResumeStart + ($SubIndex * $SubPartSize)
        if ($SubStart -gt $End) {
            break
        }
        $SubEnd = [Math]::Min($End, $SubStart + $SubPartSize - 1)
        $SubPath = Join-Path $PartsDir (
            "part-{0:D3}-sub-{1:D3}.bin" -f ($Index + 1), ($SubIndex + 1)
        )
        $ExpectedSubLength = $SubEnd - $SubStart + 1
        $SubPaths += $SubPath

        if (-not (Test-Path -LiteralPath $SubPath)) {
            [IO.File]::WriteAllBytes($SubPath, [byte[]]::new(0))
        }
        $ExistingSubLength = (Get-Item -LiteralPath $SubPath).Length
        if ($ExistingSubLength -gt $ExpectedSubLength) {
            throw "A partial checkpoint sub-segment is larger than expected."
        }
        if ($ExistingSubLength -eq $ExpectedSubLength) {
            continue
        }

        $TailStart = $SubStart + $ExistingSubLength
        $TailPath = "$SubPath.tail"
        $ExpectedTailLength = $SubEnd - $TailStart + 1
        $Arguments = @(
            "-L", "--fail", "--silent", "--show-error",
            "--retry", "8", "--retry-all-errors", "--retry-delay", "3",
            "--range", "$TailStart-$SubEnd",
            "--output", $TailPath,
            $Url
        )
        if (-not ((Test-Path -LiteralPath $TailPath) -and
                  (Get-Item -LiteralPath $TailPath).Length -eq $ExpectedTailLength)) {
            $Process = Start-Process -FilePath "curl.exe" -ArgumentList $Arguments `
                -PassThru -WindowStyle Hidden
            $Processes += [pscustomobject]@{
                Process = $Process
                Path = $TailPath
                ExpectedLength = $ExpectedTailLength
            }
        }
        $SubAssemblyPlans += [pscustomobject]@{
            Path = $SubPath
            TailPath = $TailPath
            ExpectedLength = $ExpectedSubLength
        }
    }
    $AssemblyPlans += [pscustomobject]@{
        Path = $PartPath
        ExpectedLength = $ExpectedPartLength
        SubPaths = $SubPaths
    }
}

$Failures = @()
foreach ($Item in $Processes) {
    $Item.Process.WaitForExit()
    if ($Item.Process.ExitCode -ne 0) {
        $Failures += "$($Item.Path) (curl $($Item.Process.ExitCode))"
        continue
    }
    if ((Get-Item -LiteralPath $Item.Path).Length -ne $Item.ExpectedLength) {
        $Failures += "$($Item.Path) (wrong size)"
    }
}
if ($Failures.Count -gt 0) {
    throw "Checkpoint segment failures: $($Failures -join ', ')"
}

foreach ($Plan in $SubAssemblyPlans) {
    $SubAssembling = "$($Plan.Path).assembling"
    $SubOutput = [IO.File]::Open(
        $SubAssembling, [IO.FileMode]::Create, [IO.FileAccess]::Write
    )
    try {
        @($Plan.Path, $Plan.TailPath) | ForEach-Object {
            $SubInput = [IO.File]::OpenRead($_)
            try {
                $SubInput.CopyTo($SubOutput)
            } finally {
                $SubInput.Dispose()
            }
        }
    } finally {
        $SubOutput.Dispose()
    }
    if ((Get-Item -LiteralPath $SubAssembling).Length -ne $Plan.ExpectedLength) {
        throw "A resumed checkpoint sub-segment has the wrong size."
    }
    Move-Item -LiteralPath $SubAssembling -Destination $Plan.Path -Force
    Remove-Item -LiteralPath $Plan.TailPath -Force
}

foreach ($Plan in $AssemblyPlans) {
    $PartAssembling = "$($Plan.Path).assembling"
    $PartOutput = [IO.File]::Open(
        $PartAssembling, [IO.FileMode]::Create, [IO.FileAccess]::Write
    )
    try {
        @($Plan.Path) + @($Plan.SubPaths) | ForEach-Object {
            $PartInput = [IO.File]::OpenRead($_)
            try {
                $PartInput.CopyTo($PartOutput)
            } finally {
                $PartInput.Dispose()
            }
        }
    } finally {
        $PartOutput.Dispose()
    }
    if ((Get-Item -LiteralPath $PartAssembling).Length -ne $Plan.ExpectedLength) {
        throw "A resumed checkpoint segment has the wrong assembled size."
    }
    Move-Item -LiteralPath $PartAssembling -Destination $Plan.Path -Force
    $Plan.SubPaths | ForEach-Object {
        Remove-Item -LiteralPath $_ -Force
    }
}

$Output = [IO.File]::Open($AssemblingPath, [IO.FileMode]::Create, [IO.FileAccess]::Write)
try {
    Get-ChildItem -LiteralPath $PartsDir -Filter "part-*.bin" |
        Sort-Object Name |
        ForEach-Object {
            $Input = [IO.File]::OpenRead($_.FullName)
            try {
                $Input.CopyTo($Output)
            } finally {
                $Input.Dispose()
            }
        }
} finally {
    $Output.Dispose()
}

if ((Get-Item -LiteralPath $AssemblingPath).Length -ne $ExpectedLength) {
    throw "Assembled checkpoint has the wrong size."
}
$FinalHash = (Get-FileHash -LiteralPath $AssemblingPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($FinalHash -ne $ExpectedSha256) {
    throw "Checkpoint checksum verification failed."
}

Move-Item -LiteralPath $AssemblingPath -Destination $ModelPath -Force
Get-ChildItem -LiteralPath $PartsDir -Filter "part-*.bin" |
    Remove-Item -Force
Write-Output "TripoSR checkpoint downloaded and verified."
