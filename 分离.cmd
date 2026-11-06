@echo off
setlocal enabledelayedexpansion
set "HF_HOME=D:\AI\audiosep\hf_cache"
set "ROOT=D:\AI\audiosep"
cd /d "%ROOT%"

set "INPUT=%~1"
set "PROMPT=%~2"

if "%INPUT%"=="" (
  echo.
  echo   ============================================================
  echo    音源分离工具 ^(SAM-Audio^)
  echo   ============================================================
  echo.
  echo    把音频/视频文件直接拖到本窗口，然后回车。
  echo    也可以把文件拖到本 cmd 文件图标上。
  echo.
  set /p "INPUT=  文件路径: "
)

if "%INPUT%"=="" (
  echo   没有输入文件，退出。
  pause
  exit /b 1
)

rem 去掉可能存在的引号
set "INPUT=!INPUT:"=!"

if "%PROMPT%"=="" set "PROMPT=car engine"

echo.
echo   文件   : !INPUT!
echo   提示词 : %PROMPT%
echo.
echo   模型加载约需 20 秒。长音频会自动切段处理，每 15 秒音频约需 40 秒，请勿关闭窗口。
echo.

for %%F in ("!INPUT!") do (
  set "OUTDIR=%%~dpFseparated"
)

sam-audio\.venv\Scripts\python.exe batch.py "!INPUT!" -p "%PROMPT%" -o "!OUTDIR!"

echo.
if errorlevel 1 (
  echo   处理失败，请把上面的报错发给助手。
) else (
  echo   完成，输出目录：
  echo   !OUTDIR!
  echo.
  if exist "!OUTDIR!" explorer "!OUTDIR!"
)
pause
