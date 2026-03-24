param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,

    [double]$Boost,

    [switch]$Normalize,

    [switch]$Test
)

# -----------------------------
# Check FFmpeg
# -----------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "FFmpeg not found in PATH." -ForegroundColor Red
    exit 1
}

# -----------------------------
# Validate Input
# -----------------------------
if (-not (Test-Path $InputFile)) {
    Write-Host "Input file not found." -ForegroundColor Red
    exit 1
}

$InputPath  = Resolve-Path $InputFile
$OutputFile = "louder_" + [System.IO.Path]::GetFileName($InputPath)

# -----------------------------
# Get Audio Levels
# -----------------------------
function Get-AudioLevel {
    Write-Host "Analyzing audio levels..."
    $output = ffmpeg -i $InputPath -af volumedetect -f null NUL 2>&1

    $meanLine = $output | Select-String "mean_volume"
    $maxLine  = $output | Select-String "max_volume"

    $mean = $null
    $max  = $null

    if ($meanLine) {
        $mean = [double]($meanLine -replace '.*mean_volume:\s*', '' -replace ' dB.*', '')
    }

    if ($maxLine) {
        $max = [double]($maxLine -replace '.*max_volume:\s*', '' -replace ' dB.*', '')
    }

    return @{ Mean=$mean; Max=$max }
}

# -----------------------------
# Test Mode
# -----------------------------
if ($Test) {
    $levels = Get-AudioLevel
    Write-Host ""
    Write-Host "Mean Volume : $($levels.Mean) dB"
    Write-Host "Max Volume  : $($levels.Max) dB"
    exit 0
}

# -----------------------------
# Determine Filter
# -----------------------------
$audioFilter = ""

if ($Normalize) {
    Write-Host "Using professional loudness normalization..."
    $audioFilter = "loudnorm=I=-16:TP=-1.5:LRA=11"
}
elseif ($Boost) {
    Write-Host "Using custom boost: $Boost x"
    $audioFilter = "volume=$Boost"
}
else {
    $levels = Get-AudioLevel

    if ($levels.Mean -ne $null) {
        $targetMean = -12
        $neededDb = $targetMean - $levels.Mean

        if ($neededDb -lt 0) {
            Write-Host "Audio already loud enough."
            Write-Host "Copying without change..."
            ffmpeg -i $InputPath -c copy $OutputFile
            exit 0
        }

        if ($neededDb -gt 12) { $neededDb = 12 }

        Write-Host "Auto boost: $neededDb dB"
        $audioFilter = "volume=${neededDb}dB"
    }
    else {
        Write-Host "Could not detect levels. Applying default 6dB boost."
        $audioFilter = "volume=6dB"
    }
}

# -----------------------------
# Process Video
# -----------------------------
Write-Host ""
Write-Host "Processing..."
ffmpeg -i $InputPath `
       -af $audioFilter `
       -c:v copy `
       -c:a aac `
       -b:a 192k `
       $OutputFile

Write-Host ""
Write-Host "Done -> $OutputFile"
