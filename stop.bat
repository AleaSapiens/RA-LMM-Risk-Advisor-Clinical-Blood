@echo off
set CONTAINER=ra-lmm-clinical-blood-check

echo Stopping %CONTAINER%...
docker stop "%CONTAINER%"
if errorlevel 1 (
    echo Failed to stop container, or it is not running.
    pause
    exit /b 1
)

echo Container stopped.
pause
