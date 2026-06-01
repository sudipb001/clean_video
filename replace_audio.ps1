# .\replace-audio.ps1 `
#     -Video lesson01.mp4 `
#     -Audio lesson01.wav

param(
    [string]$Video,
    [string]$Audio
)

$Output = [System.IO.Path]::GetFileNameWithoutExtension($Video) + "_clean.mp4"

ffmpeg `
    -i $Video `
    -i $Audio `
    -map 0:v:0 `
    -map 1:a:0 `
    -c:v copy `
    -c:a aac `
    -b:a 192k `
    -shortest `
    $Output

Write-Host "Created: $Output"