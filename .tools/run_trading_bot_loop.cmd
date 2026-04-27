@echo off
cd /d E:\Trading\v02
set PYTHONPATH=E:\Trading\v02\.py312pkgs;E:\Trading\v02\src
set PYTHONIOENCODING=utf-8
set LOKY_MAX_CPU_COUNT=4
"C:\Program Files\ANSYS Inc\ANSYS Student\v261\CEI\apex261\machines\win64\Python-3.12.11\python.exe" src\trading_bot.py --loop --poll-seconds 60 --symbols BTCUSDT,ETHUSDT,SOLUSDT --timeframe 1h --paper-initial-cash 10000 --sync-latest-from-binance --refresh-features --target-accepted-models 3 --model-maintenance-interval-seconds 3600 --log-level INFO 1>> logs\trading_bot_scheduled.stdout.log 2>> logs\trading_bot_scheduled.stderr.log
