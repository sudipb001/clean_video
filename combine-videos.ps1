# combine-videos.ps1
# Usage:
# .\combine-videos.ps1 file1.mp4 file2.mp4
#
# Output:
# combined_output.mp4
#
# Requires ffmpeg.exe in PATH

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$file1,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$file2
)

$output   = "combined_output.mp4"
$listFile = "concat_list.txt"

# Check input files
if (!(Test-Path $file1)) {
    Write-Host "Missing file: $file1"
    exit 1
}

if (!(Test-Path $file2)) {
    Write-Host "Missing file: $file2"
    exit 1
}

# Create concat list for FFmpeg
@"
file '$file1'
file '$file2'
"@ | Set-Content -Encoding ASCII $listFile

# Combine without re-encoding (no quality loss)
ffmpeg -f concat -safe 0 -i $listFile -c copy $output

# Remove temp file
Remove-Item $listFile -ErrorAction SilentlyContinue

Write-Host "Done. Output created: $output"