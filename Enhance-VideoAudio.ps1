#Requires -Version 5.1
<#
.SYNOPSIS
    Enhances voice and removes noise from MP4 video files.

.PARAMETER InputVideo
    Path to the input MP4 video file.

.PARAMETER OutputVideo
    (Optional) Path to save the output video. Defaults to improved_<name>.mp4

.PARAMETER NoEchoCancel
    Switch to disable echo cancellation.

.PARAMETER NoNoiseReduction
    Switch to disable noise reduction.

.PARAMETER UsePython
    Switch to use the companion audio_enhancer.py instead of FFmpeg filters.

.EXAMPLE
    .\Enhance-VideoAudio.ps1 -InputVideo "recording.mp4"

.EXAMPLE
    .\Enhance-VideoAudio.ps1 -InputVideo "recording.mp4" -OutputVideo "clean.mp4" -NoEchoCancel
#>

[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateScript({ Test-Path $_ -PathType Leaf })]
    [string]$InputVideo,

    [Parameter(Mandatory = $false)]
    [string]$OutputVideo,

    [Parameter(Mandatory = $false)]
    [switch]$NoEchoCancel,

    [Parameter(Mandatory = $false)]
    [switch]$NoNoiseReduction,

    [Parameter(Mandatory = $false)]
    [switch]$UsePython
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Helpers ──────────────────────────────────────────────────────────────────

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Step {
    param([int]$Number, [int]$Total, [string]$Text)
    Write-Host ""
    Write-Host "[Step $Number/$Total] $Text" -ForegroundColor Yellow
}

function Get-MediaDuration {
    param([string]$FilePath)
    try {
        $out = & ffprobe -v error -show_entries format=duration `
                         -of default=noprint_wrappers=1:nokey=1 "$FilePath" 2>&1
        $num = ($out | Where-Object { $_ -match '^\d' } | Select-Object -First 1)
        return [double]$num
    } catch {
        return 0.0
    }
}

# Run FFmpeg with a live progress bar reading stderr line-by-line.
# Returns the process exit code (int).
function Invoke-FFmpeg {
    param(
        [string[]]$Arguments,
        [string]$Activity,
        [double]$DurationSeconds
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName               = "ffmpeg"
    $psi.Arguments              = ($Arguments -join " ")
    $psi.UseShellExecute        = $false
    $psi.RedirectStandardError  = $true
    $psi.RedirectStandardOutput = $true
    $psi.CreateNoWindow         = $true

    $proc = [System.Diagnostics.Process]::new()
    $proc.StartInfo = $psi
    $null = $proc.Start()

    $spinner   = @('|', '/', '-', '\')
    $spinIdx   = 0
    $lastPct   = 0

    while (-not $proc.StandardError.EndOfStream) {
        $line = $proc.StandardError.ReadLine()

        # Parse "time=HH:MM:SS.xx" from ffmpeg progress output
        if ($line -match 'time=(\d+):(\d+):([\d.]+)') {
            $secs    = [int]$Matches[1] * 3600 + [int]$Matches[2] * 60 + [double]$Matches[3]
            $lastPct = if ($DurationSeconds -gt 0) {
                [Math]::Min([int](($secs / $DurationSeconds) * 100), 99)
            } else { $lastPct }
        }

        Write-Progress -Activity $Activity `
                       -Status "$($spinner[$spinIdx % 4]) ${lastPct}%" `
                       -PercentComplete $lastPct
        $spinIdx++
    }

    $proc.WaitForExit()
    Write-Progress -Activity $Activity -Status "Done" -PercentComplete 100 -Completed
    return $proc.ExitCode
}

# ─── Step 1: Extract audio ────────────────────────────────────────────────────

function Extract-Audio {
    param([string]$VideoPath)

    $tmpWav  = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), ".wav")
    $duration = Get-MediaDuration -FilePath $VideoPath

    Write-Host "  Temp WAV : $tmpWav" -ForegroundColor Gray
    Write-Host "  Duration : $duration s" -ForegroundColor Gray

    $args = @(
        "-y", "-i", "`"$VideoPath`"",
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "2",
        "`"$tmpWav`""
    )

    $code = Invoke-FFmpeg -Arguments $args -Activity "Extracting audio" -DurationSeconds $duration

    if ($code -ne 0) { throw "Audio extraction failed (FFmpeg exit code: $code)." }

    Write-Host "  Audio extracted successfully." -ForegroundColor Green
    return $tmpWav
}

# ─── Step 2a: Enhance via FFmpeg filter chain ─────────────────────────────────

function Enhance-AudioFFmpeg {
    param(
        [string]$AudioPath,
        [bool]$ApplyNoiseReduction
    )

    $tmpOut = [System.IO.Path]::ChangeExtension(
        [System.IO.Path]::GetTempFileName(),
        "_enhanced.wav"
    )

    $filters = [System.Collections.Generic.List[string]]::new()

    # Remove low-frequency rumble (fan noise, table vibration, mic handling noise)
    $filters.Add("highpass=f=70")

    # Remove excessive high-frequency hiss
    $filters.Add("lowpass=f=10000")

    # Gentle broadband noise reduction
    if ($ApplyNoiseReduction) {
        $filters.Add("afftdn=nr=10:nf=-30")
    }

    # Very gentle noise gate to suppress room noise
    $filters.Add("agate=threshold=0.005:ratio=1.5:attack=20:release=300")

    # Add presence and speech clarity
    $filters.Add("equalizer=f=2500:width_type=h:width=1500:g=2")

    # Reduce muddiness in deep male voices
    $filters.Add("equalizer=f=180:width_type=h:width=120:g=-2")

    # Gentle dynamic compression for consistent volume
    $filters.Add(
        "acompressor=threshold=-20dB:ratio=1.8:attack=30:release=250:makeup=1"
    )

    # Normalize loudness for YouTube/Udemy tutorials
    $filters.Add("loudnorm=I=-16:TP=-1.5:LRA=11")

    $chain = $filters -join ","

    Write-Host "  Filter chain: $chain" -ForegroundColor Gray

    $args = @(
        "-y",
        "-i", "`"$AudioPath`"",
        "-af", "`"$chain`"",
        "-ar", "44100",
        "-acodec", "pcm_s16le",
        "`"$tmpOut`""
    )

    $duration = Get-MediaDuration -FilePath $AudioPath

    $code = Invoke-FFmpeg `
        -Arguments $args `
        -Activity "Enhancing audio" `
        -DurationSeconds $duration

    if ($code -ne 0) {
        throw "Audio enhancement failed (FFmpeg exit code: $code)."
    }

    Write-Host "  Audio enhanced successfully." -ForegroundColor Green

    return $tmpOut
}


# ─── Step 2b: Enhance via Python script ──────────────────────────────────────

function Enhance-AudioPython {
    param([string]$AudioPath, [bool]$ApplyEchoCancel)

    $pyScript = Join-Path $PSScriptRoot "audio_enhancer.py"
    if (-not (Test-Path $pyScript)) {
        throw "audio_enhancer.py not found at $pyScript."
    }

    $tmpOut  = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), "_py.wav")
    $pyArgs  = @("`"$pyScript`"", "`"$AudioPath`"", "--output", "`"$tmpOut`"")
    if (-not $ApplyEchoCancel) { $pyArgs += "--no-echo-cancel" }

    Write-Host "  Running Python enhancer..." -ForegroundColor Gray
    $proc = Start-Process python -ArgumentList $pyArgs -NoNewWindow -PassThru
    $proc.WaitForExit()

    if ($proc.ExitCode -ne 0) { throw "Python enhancer failed (exit $($proc.ExitCode))." }
    Write-Host "  Python enhancement complete." -ForegroundColor Green
    return $tmpOut
}

# ─── Step 3: Merge audio back into video ─────────────────────────────────────

function Merge-AudioVideo {
    param(
        [string]$VideoPath,
        [string]$EnhancedAudioPath,
        [string]$OutputPath
    )

    $duration = Get-MediaDuration -FilePath $VideoPath
    Write-Host "  Output: $OutputPath" -ForegroundColor Gray

    $args = @(
        "-y",
        "-i", "`"$VideoPath`"",
        "-i", "`"$EnhancedAudioPath`"",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "320k",
        "-map", "0:v:0", "-map", "1:a:0",
        "`"$OutputPath`""
    )

    $code = Invoke-FFmpeg -Arguments $args -Activity "Combining audio and video" -DurationSeconds $duration

    if ($code -ne 0) { throw "Video merge failed (FFmpeg exit code: $code)." }
    Write-Host "  Video saved: $OutputPath" -ForegroundColor Green
}

# ─── Check FFmpeg ─────────────────────────────────────────────────────────────

function Test-FFmpeg {
    try   { $null = & ffmpeg -version 2>&1; if ($LASTEXITCODE -ne 0) { throw } }
    catch {
        Write-Host "ERROR: FFmpeg not found. Get it from https://ffmpeg.org/download.html" -ForegroundColor Red
        exit 1
    }
    Write-Host "  FFmpeg found." -ForegroundColor Green
}

# =============================================================================
#  MAIN
# =============================================================================

Write-Header "MP4 Audio Enhancer - Voice Clarity and Noise Removal"

$InputVideo = Resolve-Path $InputVideo | Select-Object -ExpandProperty Path

if (-not $OutputVideo) {
    $dir         = Split-Path $InputVideo -Parent
    $base        = [System.IO.Path]::GetFileNameWithoutExtension($InputVideo)
    $ext         = [System.IO.Path]::GetExtension($InputVideo)
    $OutputVideo = Join-Path $dir "improved_${base}${ext}"
}

Write-Host "  Input : $InputVideo"
Write-Host "  Output: $OutputVideo"
Write-Host ""
Write-Host "  Settings:" -ForegroundColor White
Write-Host "    Noise Reduction : $(if ($NoNoiseReduction) { 'DISABLED' } else { 'ENABLED' })"
Write-Host "    Mode            : $(if ($UsePython) { 'Python' } else { 'FFmpeg filters' })"


Write-Step -Number 0 -Total 3 -Text "Checking dependencies"
Test-FFmpeg

$tempFiles = [System.Collections.Generic.List[string]]::new()
$startTime = [datetime]::Now

try {
    Write-Step -Number 1 -Total 3 -Text "Extracting audio from video"
    $tempWav = Extract-Audio -VideoPath $InputVideo
    $tempFiles.Add($tempWav)

    Write-Step -Number 2 -Total 3 -Text "Enhancing audio"
    if ($UsePython) {
        $enhancedWav = Enhance-AudioPython -AudioPath $tempWav -ApplyEchoCancel (-not $NoEchoCancel)
    } else {
        $enhancedWav = Enhance-AudioFFmpeg `
            -AudioPath $tempWav `
            -ApplyNoiseReduction (-not $NoNoiseReduction)
    }
    $tempFiles.Add($enhancedWav)

    Write-Step -Number 3 -Total 3 -Text "Combining enhanced audio with video"
    Merge-AudioVideo -VideoPath $InputVideo -EnhancedAudioPath $enhancedWav -OutputPath $OutputVideo

    $elapsed = [Math]::Round(([datetime]::Now - $startTime).TotalSeconds, 1)

    Write-Header "Enhancement Complete"
    Write-Host "  Time  : ${elapsed}s"  -ForegroundColor Green
    Write-Host "  Output: $OutputVideo" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Improvements applied:" -ForegroundColor White
    if (-not $NoNoiseReduction) {
        Write-Host "    [+] Noise reduction (FFT denoiser + noise gate)"
    }
    Write-Host "    [+] Low-frequency rumble removal"
    Write-Host "    [+] High-frequency hiss reduction"
    Write-Host "    [+] Speech clarity enhancement"
    Write-Host "    [+] Deep voice mud reduction"
    Write-Host "    [+] Gentle dynamic range compression"
    Write-Host "    [+] Loudness normalization (-16 LUFS)"
    Write-Host ""

} catch {
    Write-Host ""
    Write-Host "ERROR: $_" -ForegroundColor Red
    exit 1
} finally {
    foreach ($f in $tempFiles) {
        if ($f -and (Test-Path $f)) { Remove-Item $f -Force -ErrorAction SilentlyContinue }
    }
}