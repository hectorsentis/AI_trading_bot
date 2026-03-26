
DATA DOWLOAD EXAMPLES

python src/download_data.py
python src/download_data.py --mode recent --symbols BTCUSDT ETHUSDT --timeframe 1h --recent-bars 500
python src/download_data.py --mode range --symbols BTCUSDT --timeframe 15m --start-date 2024-01-01 --end-date 2024-03-31
python src/download_data.py --mode full --symbols SOLUSDT --timeframe 4h
python src/download_data.py --dry-run


# Trading Bot (Binance)

Pipeline de datos para trading cuantitativo:

## Estructura

- download_data.py → descarga histórico
- data_loader.py → carga a SQLite
- data_check.py → detecta gaps
- data_gap_fill.py → rellena gaps

## Flujo

```bash
python src/download_data.py
python src/data_loader.py
python src/data_check.py
python src/data_gap_fill.py

