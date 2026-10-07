@echo off
title Fake News Detection - Frontend Server
echo ============================================================
echo Starting Frontend Web Server at http://127.0.0.1:3000 ...
echo ============================================================
echo Opening browser...
start http://127.0.0.1:3000
python -m http.server 3000 --directory "Front End"
pause
