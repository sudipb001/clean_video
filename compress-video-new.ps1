# # Normal mode (same behavior as before)
# .\compress-video-fast.ps1 video.mp4

# # Faster processing by copying audio
# .\compress-video-fast.ps1 video.mp4 -CopyAudio

# # H.265 with copied audio
# .\compress-video-fast.ps1 video.mp4 -Codec h265 -CopyAudio

# # Target size mode
# .\compress-video-fast.ps1 video.mp4 -TargetSizeMB 200 -CopyAudio

param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [string]$OutputFile,

    [ValidateSet("h264", "h265")]
    [string]$Codec = "h264",

    [int]$CRF = 23,

    [ValidateSet("ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow")]
    [string]$Preset = "veryfast",

    [double]$TargetSizeMB,

    [switch]$TwoPass,

    [switch]$CopyAudio
)

# -----------------------------
# Check FFmpeg
# -----------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "FFmpeg not found in PATH." -ForegroundColor Red
    exit 1
}

if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Write-Host "FFprobe not found in PATH." -ForegroundColor Red
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
Write-Host "Original size: $([math]::Round($InitialSizeMB, 2)) MB"

# -----------------------------
# Get Duration
# -----------------------------
$Duration = ffprobe -v error `
                     -show_entries format=duration `
                     -of default=noprint_wrappers=1:nokey=1 `
                     $InputPath

$Duration = [double]$Duration

# -----------------------------
# Select Codec
# -----------------------------
if ($Codec -eq "h265") {
    $VideoCodec = "libx265"
}
else {
    $VideoCodec = "libx264"
}

# -----------------------------
# Audio Settings
# -----------------------------
if ($CopyAudio) {
    Write-Host "Audio: Copy mode enabled"
    $AudioArgs = @("-c:a", "copy")
}
else {
    $AudioArgs = @("-c:a", "aac", "-b:a", "128k")
}

# -----------------------------
# TARGET SIZE MODE
# -----------------------------
if ($TargetSizeMB) {

    Write-Host "Target size mode: $TargetSizeMB MB"

    $TotalBits = $TargetSizeMB * 8 * 1024 * 1024
    $TotalBitrate = [int](($TotalBits / $Duration) * 0.95)

    $AudioBitrate = if ($CopyAudio) { 128000 } else { 128000 }
    $VideoBitrate = $TotalBitrate - $AudioBitrate

    if ($TwoPass) {

        Write-Host "Two-pass encoding..."

        ffmpeg -y -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -preset $Preset `
               -threads 0 `
               -pass 1 `
               -an `
               -f mp4 NUL

        ffmpeg -y -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -preset $Preset `
               -threads 0 `
               -pass 2 `
               @AudioArgs `
               -movflags +faststart `
               $OutputFile

        Remove-Item "ffmpeg2pass-0.log*" -ErrorAction SilentlyContinue
    }
    else {

        ffmpeg -y -i $InputPath `
               -c:v $VideoCodec `
               -b:v $VideoBitrate `
               -preset $Preset `
               -threads 0 `
               @AudioArgs `
               -movflags +faststart `
               $OutputFile
    }
}
else {

    Write-Host "CRF Quality Mode"
    Write-Host "Codec: $Codec  CRF: $CRF  Preset: $Preset"

    ffmpeg -y -i $InputPath `
           -c:v $VideoCodec `
           -crf $CRF `
           -preset $Preset `
           -threads 0 `
           @AudioArgs `
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
    Write-Host "Original Size   : $([math]::Round($InitialSizeMB, 2)) MB"
    Write-Host "Compressed Size : $([math]::Round($FinalSizeMB, 2)) MB"
    Write-Host "Reduction       : $([math]::Round($Reduction, 2)) %"
}
else {
    Write-Host "Compression failed." -ForegroundColor Red
}
