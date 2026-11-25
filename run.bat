@echo off
set IMAGE=blythe0827/ra-lmm-risk-advisor-variants:clinical-blood-v1.0.0
set CONTAINER=ra-lmm-clinical-blood-check
set URL=http://127.0.0.1:3840

echo Pulling %IMAGE%...
docker pull %IMAGE%
if errorlevel 1 (
    echo Failed to pull image.
    pause
    exit /b 1
)

docker stop "%CONTAINER%" >nul 2>&1

echo Starting %CONTAINER%...
docker run --rm -d -p 127.0.0.1:3840:7860 --name "%CONTAINER%" "%IMAGE%"
if errorlevel 1 (
    echo Failed to start container.
    pause
    exit /b 1
)

start "" "%URL%"
echo Application started at %URL%
pause
