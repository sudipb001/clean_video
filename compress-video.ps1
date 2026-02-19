param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,

    [string]$OutputFile,

    [ValidateSet("h264","h265")]
    [string]$Codec = "h264",

    [int]$CRF = 23,

    [ValidateSet("ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow")]
    [string]$Preset = "medium",

    [double]$TargetSizeMB,

    [switch]$TwoPass
)

# -----------------------------
# Check FFmpeg
# -----------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "FFmpeg not found in PATH." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $InputFile)) {
    Write-Host "Input file not found." -ForegroundColor Red
    exit 1
}

$InputPath = Resolve-Path $InputFile

if (-not $OutputFile) {
    $OutputFile = "compressed_" + [System.IO.Path]::GetFileName($InputPath)
}

$InitialSizeMB = (Get-Item $InputPath).Length / 1MB

Write-Host ""
Write-Host "Original size: $([math]::Round($InitialSizeMB,2)) MB"

# -----------------------------
# Get Duration
# -----------------------------
$Duration = ffprobe -v error -show_entries format=duration `
                     -of default=noprint_wrappers=1:nokey=1 `
                     $InputPath

$Duration = [double]$Duration

# -----------------------------
# Select Codec
# -----------------------------
if ($Codec -eq "h265") {
    $VideoCodec = "libx265"
} else {
    $VideoCodec = "libx264"
}

# -----------------------------
# TARGET SIZE MODE
# -----------------------------
if ($TargetSizeMB) {

    Write-Host "Target size mode: $TargetSizeMB MB"

    $TotalBits = $TargetSizeMB * 8 * 1024 * 1024
    $TotalBitrate = [int](($TotalBits / $Duration) * 0.95)

    $AudioBitrate = 128000
    $VideoBitrate = $TotalBitrate - $AudioBitrate

    if ($TwoPass) {

        Write-Host "Two-pass encoding..."

        ffmpeg -y -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -pass 1 `
               -an `
               -f mp4 NUL

        ffmpeg -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -pass 2 `
               -c:a aac `
               -b:a 128k `
               -movflags +faststart `
               $OutputFile

        Remove-Item "ffmpeg2pass-0.log*" -ErrorAction SilentlyContinue
    }
    else {

        ffmpeg -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -c:a aac `
               -b:a 128k `
               -movflags +faststart `
               $OutputFile
    }
}
else {

    Write-Host "CRF Quality Mode"
    Write-Host "Codec: $Codec  CRF: $CRF  Preset: $Preset"

    ffmpeg -i $InputPath `
           -c:v $VideoCodec `
           -crf $CRF `
           -preset $Preset `
           -c:a aac `
           -b:a 128k `
           -movflags +faststart `
           $OutputFile
}

# -----------------------------
# Results
# -----------------------------
if (Test-Path $OutputFile) {

    $FinalSizeMB = (Get-Item $OutputFile).Length / 1MB
    $Reduction = (1 - ($FinalSizeMB / $InitialSizeMB)) * 100

    Write-Host ""
    Write-Host "Compression Completed"
    Write-Host "-------------------------------------"
    Write-Host "Original Size : $([math]::Round($InitialSizeMB,2)) MB"
    Write-Host "Compressed Size: $([math]::Round($FinalSizeMB,2)) MB"
    Write-Host "Reduction     : $([math]::Round($Reduction,2)) %"
}
else {
    Write-Host "Compression failed." -ForegroundColor Red
}
