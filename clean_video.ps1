<#
.SYNOPSIS
    Advanced video audio cleaning using native FFmpeg filters for Windows.
#>

$ErrorActionPreference = "Stop"

# --- Default Configuration ---
$TARGET_LOUDNESS = "-16.0"
$NOISE_REDUCTION = "0.25"
$PRESERVE_VOICE = "0.6"
$APPLY_COMPRESSION = $true
$APPLY_DEESS = $true
$ENHANCE_EQ = $true
$INFO_ONLY = $false
$INSTALL_DEPS_FLAG = $false
$INPUT_FILE = ""
$OUTPUT_FILE = ""

# --- Helper Functions ---
function Get-Usage {
    @"
Advanced video audio cleaning using native FFmpeg filters (Windows PowerShell).

Usage:
  .\clean_video.ps1 <input_file> [options]
  .\clean_video.ps1 --install-deps

Options:
  -o, --output <file>    Custom output file path
  --target <float>       Target loudness in LUFS (default: -16.0)
  --noise <float>        Noise reduction strength 0.0-1.0 (default: 0.25)
  --voice <float>        Voice preservation 0.0-1.0 (default: 0.6)
  --no-compression       Disable dynamic range compression
  --no-deess             Disable de-esser
  --no-eq                Disable EQ enhancement
  --info                 Show video information and exit
  --install-deps         Check/Install FFmpeg on Windows via winget
"@
}

function Test-Dependencies {
    $ffmpegCheck = Get-Command ffmpeg -ErrorAction SilentlyContinue
    $ffprobeCheck = Get-Command ffprobe -ErrorAction SilentlyContinue
    if (-not $ffmpegCheck -or -not $ffprobeCheck) {
        Write-Error "Error: FFmpeg/FFprobe not found in system PATH. Run with '--install-deps' or install them manually."
        exit 1
    }
}

function Install-Dependencies {
    Write-Host "[*] Checking and installing system dependencies via winget..." -ForegroundColor Cyan
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing FFmpeg..."
        winget install "Gyan.FFmpeg" --silent
        Write-Host "[+] Installation sent via winget. Please restart your PowerShell session to reload the system PATH environment variable." -ForegroundColor Green
    } else {
        Write-Error "Error: winget package manager was not found. Please download and install FFmpeg manually from https://ffmpeg.org/download.html and add it to your system PATH."
        exit 1
    }
}

function Show-VideoInfo {
    param ([string]$file)
    Write-Host "[i] Video Information for: $file" -ForegroundColor Cyan
    $info = & ffprobe -v quiet -print_format json -show_format -show_streams $file | ConvertFrom-Json
    
    if ($info) {
        foreach ($stream in $info.streams) {
            if ($stream.codec_type -eq "video") {
                Write-Host "Video: $($stream.width)x$($stream.height), $($stream.codec_name)"
            }
            elseif ($stream.codec_type -eq "audio") {
                Write-Host "Audio: $($stream.codec_name), $($stream.sample_rate) Hz"
            }
        }
        Write-Host "Duration: $([Math]::Round([double]$info.format.duration, 1)) seconds"
        Write-Host "Size: $([Math]::Round(([double]$info.format.size / 1MB), 1)) MB"
    } else {
        Write-Host "Failed to read properties." -ForegroundColor Red
    }
}

# --- CLI Arguments Parser ---
$argsList = @($args)
for ($i = 0; $i -lt $argsList.Count; $i++) {
    switch ($argsList[$i]) {
        "--install-deps"   { $INSTALL_DEPS_FLAG = $true }
        "--info"           { $INFO_ONLY = $true }
        "--no-compression" { $APPLY_COMPRESSION = $false }
        "--no-deess"       { $APPLY_DEESS = $false }
        "--no-eq"          { $ENHANCE_EQ = $false }
        "--target"         { $TARGET_LOUDNESS = $argsList[++$i] }
        "--noise"          { $NOISE_REDUCTION = $argsList[++$i] }
        "--voice"          { $PRESERVE_VOICE = $argsList[++$i] }
        "-o"               { $OUTPUT_FILE = $argsList[++$i] }
        "--output"         { $OUTPUT_FILE = $argsList[++$i] }
        "-h"               { Get-Usage; exit 0 }
        "--help"           { Get-Usage; exit 0 }
        { $_.StartsWith("-") } { Write-Error "Error: Unknown option: $_"; Get-Usage; exit 1 }
        Default            { $INPUT_FILE = $argsList[$i] }
    }
}

# Execute install-deps early if called
if ($INSTALL_DEPS_FLAG) {
    Install-Dependencies
    exit 0
}

# Input validation
if ([string]::IsNullOrEmpty($INPUT_FILE)) {
    Write-Error "Error: Missing input video file."
    Get-Usage
    exit 1
}

if (-not (Test-Path -Path $INPUT_FILE -PathType Leaf)) {
    Write-Error "Error: File not found -> $INPUT_FILE"
    exit 1
}

Test-Dependencies

if ($INFO_ONLY) {
    Show-VideoInfo $INPUT_FILE
    exit 0
}

# Auto-generate output filename if missing
if ([string]::IsNullOrEmpty($OUTPUT_FILE)) {
    $item = Get-Item $INPUT_FILE
    $dir = $item.DirectoryName
    $base = $item.Name
    $OUTPUT_FILE = Join-Path $dir "cleaned_$base"
}

# --- Core Audio Engine (FFmpeg Filter-Graph Construction) ---
$AUDIO_FILTERS = "highpass=f=80"

# Pure mathematical evaluation replacement for awk
$noise_reduction_db = 10 + ([double]$NOISE_REDUCTION * 30)
$AUDIO_FILTERS += ",afftdn=nr=$noise_reduction_db"

if ($ENHANCE_EQ) {
    $AUDIO_FILTERS += ",equalizer=f=300:width_type=h:width=200:g=2"
    $AUDIO_FILTERS += ",equalizer=f=1000:width_type=h:width=300:g=1.5"
}

if ($APPLY_DEESS) {
    $AUDIO_FILTERS += ",deesser=f=0.5:s=0:i=0.5"
}

$AUDIO_FILTERS += ",lowpass=f=12000"

if ($APPLY_COMPRESSION) {
    $AUDIO_FILTERS += ",compand=attacks=0.1:decays=0.2:points=-50/-50|-40/-30|-20/-20|0/-10"
}

$AUDIO_FILTERS += ",alimiter=level_out=0.95:limit=0.95"
$AUDIO_FILTERS += ",loudnorm=I=${TARGET_LOUDNESS}:TP=-1.5:LRA=11"

# --- Main Engine Processing ---
$startTime = [DateTimeOffset]::Now.ToUnixTimeSeconds()

Write-Host "[*] Processing: $INPUT_FILE" -ForegroundColor Cyan
Write-Host "[*] Output:     $OUTPUT_FILE" -ForegroundColor Cyan
Write-Host "--------------------------------------------------"
Write-Host "[*] Running combined filter pipeline..." -ForegroundColor Yellow

# Combined processing execution loop
& ffmpeg -y -i $INPUT_FILE `
    -af $AUDIO_FILTERS `
    -c:v copy `
    -map 0:v:0 -map 0:a:0? `
    -shortest `
    -stats `
    $OUTPUT_FILE

if ($LASTEXITCODE -eq 0 -and (Test-Path -Path $OUTPUT_FILE)) {
    $endTime = [DateTimeOffset]::Now.ToUnixTimeSeconds()
    $elapsed = $endTime - $startTime
    $fileSize = "$([Math]::Round(((Get-Item $OUTPUT_FILE).Length / 1MB), 1))M"
    
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "[+] Success! Output created: $OUTPUT_FILE ($fileSize)" -ForegroundColor Green
    Write-Host "[+] Total processing time: $elapsed seconds" -ForegroundColor Green
    exit 0
} else {
    Write-Error "Error: Final file merge operation failed."
    exit 1
}