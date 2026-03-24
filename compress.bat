@echo off
setlocal

set "INPUT=%~1"
set "OUTPUT=%~2"

if "%INPUT%"=="" (
  echo Usage: compress input.mp4 output.mp4
  exit /b 1
)

if "%OUTPUT%"=="" (
  set "OUTPUT=compressed_%~nx1"
)

ffmpeg -y -i "%INPUT%" ^
  -c:v libx264 ^
  -crf 23 ^
  -preset medium ^
  -c:a aac -b:a 128k ^
  -movflags +faststart ^
  "%OUTPUT%"

endlocal
