@echo off
if "%~1"=="" (
    echo Usage: smart_louder.bat "video.mp4"
    exit /b 1
)

set INPUT=%~1
set OUTPUT=louder_%~nx1

echo Analyzing audio...
for /f "tokens=2 delims=:" %%A in ('ffmpeg -i "%INPUT%" -af volumedetect -f null NUL 2^>^&1 ^| find "mean_volume"') do (
    set MEAN=%%A
)

echo Mean volume is %MEAN%

echo Applying 6dB boost...
ffmpeg -i "%INPUT%" -af "volume=6dB" -c:v copy -c:a aac -b:a 192k "%OUTPUT%"

echo Done.
