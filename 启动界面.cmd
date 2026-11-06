@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 音源分离工作台
echo.
echo   ============================================================
echo    音源分离工作台
echo   ============================================================
echo.
echo    正在启动，浏览器会自动打开 http://127.0.0.1:8899
echo    模型加载约 35 秒，界面右上角会显示状态。
echo.
echo    关闭本窗口即停止服务。
echo.
sam-audio\.venv\Scripts\python.exe ui.py
if errorlevel 1 (
  echo.
  echo   启动失败，请把上面的报错发给助手。
  pause
)
