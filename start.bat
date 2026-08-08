@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
D:\anaconda3\envs\use\python.exe backend\server.py
pause
