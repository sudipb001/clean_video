#requires -version 5.1
<#
.SYNOPSIS
    Remove a segment from an MP4 file using ffmpeg.

.DESCRIPTION
    PowerShell equivalent of the remove_segment bash script. Removes a time
    range [RemoveStart, RemoveEnd) from an input MP4, either via a fast
    copy/remux+concat approach (may produce black frames if not keyframe
    aligned) or a safe re-encode approach using filter_complex.

.PARAMETER InputFile
    Input MP4 file.

.PARAMETER RemoveStart
    Start time to remove. Accepts HH:MM:SS, MM:SS, or seconds (e.g. 982).

.PARAMETER RemoveEnd
    End time to remove. Accepts HH:MM:SS, MM:SS, or seconds (e.g. 1270).

.PARAMETER Output
    Output filename. Default: output.mp4

.PARAMETER Fast
    Fast mode: try lossless copy/remux + concat instead of re-encoding.

.PARAMETER Crf
    CRF value for re-encode mode. Default: 18

.PARAMETER Preset
    x264 preset for re-encode mode. Default: veryfast

.PARAMETER AudioBitrate
    AAC audio bitrate for re-encode mode. Default: 192k

.PARAMETER Force
    Overwrite output if it already exists.

.EXAMPLE
    .\Remove-Segment.ps1 input.mp4 00:16:22 00:21:10 -Output output.mp4

.EXAMPLE
    .\Remove-Segment.ps1 -Fast input.mp4 982 1270 -Output out_fast.mp4
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputFile,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$RemoveStart,

    [Parameter(Mandatory = $true, Position = 2)]
    [string]$RemoveEnd,

    [Alias("o")]
    [string]$Output = "output.mp4",

    [switch]$Fast,

    [double]$Crf = 18,

    [string]$Preset = "veryfast",

    [string]$AudioBitrate = "192k",

    [Alias("y")]
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper: convert HH:MM:SS, MM:SS, or seconds to a double number of seconds
# ---------------------------------------------------------------------------
function ConvertTo-Seconds {
    param([string]$TimeString)

    # Plain number (integer or float)
    if ($TimeString -match '^[0-9]+(\.[0-9]+)?$') {
        return [double]$TimeString
    }

    $parts = $TimeString -split ':'

    switch ($parts.Count) {
        2 {
            # MM:SS
            $minutes = [double]$parts[0]
            $seconds = [double]$parts[1]
            return ($minutes * 60) + $seconds
        }
        3 {
            # HH:MM:SS
            $hours   = [double]$parts[0]
            $minutes = [double]$parts[1]
            $seconds = [double]$parts[2]
            return ($hours * 3600) + ($minutes * 60) + $seconds
        }
        default {
            throw "ERR: invalid time format '$TimeString'"
        }
    }
}

# ---------------------------------------------------------------------------
# Validate prerequisites
# ---------------------------------------------------------------------------
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: ffmpeg not found in PATH. Install ffmpeg first."
    exit 2
}

if (-not (Test-Path -LiteralPath $InputFile -PathType Leaf)) {
    Write-Error "ERROR: input file not found: $InputFile"
    exit 2
}

# ---------------------------------------------------------------------------
# Convert and validate times
# ---------------------------------------------------------------------------
try {
    $removeStartSec = ConvertTo-Seconds -TimeString $RemoveStart
}
catch {
    Write-Error "Bad start time: $RemoveStart"
    exit 2
}

try {
    $removeEndSec = ConvertTo-Seconds -TimeString $RemoveEnd
}
catch {
    Write-Error "Bad end time: $RemoveEnd"
    exit 2
}

if ($removeEndSec -le $removeStartSec) {
    Write-Error "ERROR: RemoveEnd must be greater than RemoveStart"
    exit 2
}

# ---------------------------------------------------------------------------
# ffmpeg overwrite flag
# ---------------------------------------------------------------------------
$ffYFlag = if ($Force) { "-y" } else { "-n" }

# ---------------------------------------------------------------------------
# Temp files (cleaned up on exit, including on error)
# ---------------------------------------------------------------------------
$tmpPrefix = Join-Path ([System.IO.Path]::GetTempPath()) ("remove_segment." + [System.IO.Path]::GetRandomFileName().Substring(0, 8))
$tempFiles = @()

function Cleanup {
    foreach ($f in $tempFiles) {
        if (Test-Path -LiteralPath $f) {
            Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
        }
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host "Input: $InputFile"
Write-Host "Removing: $RemoveStart -> $RemoveEnd (seconds: $removeStartSec -> $removeEndSec)"
Write-Host "Output: $Output"
if ($Fast) {
    Write-Host "Mode: FAST (copy/remux)"
} else {
    Write-Host "Mode: RE-ENCODE (safe)"
}

try {
    if ($Fast) {
        # -----------------------------------------------------------------
        # Fast path: remux input, cut at nearest keyframes, concat
        # -----------------------------------------------------------------
        $remux = "${tmpPrefix}_remux.mp4"
        $part1 = "${tmpPrefix}_part1.mp4"
        $part2 = "${tmpPrefix}_part2.mp4"
        $filesList = "${tmpPrefix}_files.txt"
        $tempFiles += @($remux, $part1, $part2, $filesList)

        Write-Host "Remuxing to clean container: $remux"
        & ffmpeg $ffYFlag -i $InputFile -c copy -map 0 -movflags +faststart $remux
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg remux failed with exit code $LASTEXITCODE" }

        Write-Host "Extracting part1 (0 -> $removeStartSec) to $part1 (keyframe-seek)"
        & ffmpeg $ffYFlag -ss 0 -i $remux -t $removeStartSec -c copy -avoid_negative_ts 1 -fflags +genpts $part1
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg part1 extraction failed with exit code $LASTEXITCODE" }

        Write-Host "Extracting part2 ($removeEndSec -> end) to $part2 (keyframe-seek)"
        & ffmpeg $ffYFlag -ss $removeEndSec -i $remux -c copy -avoid_negative_ts 1 -fflags +genpts $part2
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg part2 extraction failed with exit code $LASTEXITCODE" }

        # concat list (single quotes around paths, as ffmpeg concat demuxer expects)
        $part1Escaped = $part1 -replace "'", "'\\''"
        $part2Escaped = $part2 -replace "'", "'\\''"
        @(
            "file '$part1Escaped'"
            "file '$part2Escaped'"
        ) | Set-Content -LiteralPath $filesList -Encoding ascii

        Write-Host "Concatenating into $Output"
        & ffmpeg $ffYFlag -f concat -safe 0 -i $filesList -c copy -fflags +genpts $Output
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg concat failed with exit code $LASTEXITCODE" }

        Write-Host "Done (fast mode). Verify output for micro-flash or sync issues."
    }
    else {
        # -----------------------------------------------------------------
        # Re-encode path using filter_complex (guaranteed clean)
        # -----------------------------------------------------------------
        Write-Host "Running re-encode remove (this will re-encode video/audio; may take time)"

        $filterComplex = "[0:v]trim=0:$removeStartSec,setpts=PTS-STARTPTS[v0];" +
                          "[0:a]atrim=0:$removeStartSec,asetpts=PTS-STARTPTS[a0];" +
                          "[0:v]trim=$removeEndSec,setpts=PTS-STARTPTS[v1];" +
                          "[0:a]atrim=$removeEndSec,asetpts=PTS-STARTPTS[a1];" +
                          "[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]"

        $ffArgs = @(
            $ffYFlag,
            "-i", $InputFile,
            "-filter_complex", $filterComplex,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", $Crf, "-preset", $Preset,
            "-c:a", "aac", "-b:a", $AudioBitrate,
            "-movflags", "+faststart",
            $Output
        )

        Write-Host ("ffmpeg " + ($ffArgs -join " "))
        & ffmpeg @ffArgs
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg re-encode failed with exit code $LASTEXITCODE" }

        Write-Host "Done (re-encode)."
    }
}
finally {
    Cleanup
}
