# Skill: download-data

## When to use
First-time setup, or to refresh historical/incremental OHLCV before training or trading.

## Preconditions
- DB initialized: `python src/db_utils.py --init --check-schema`
- Symbols/timeframes configured in `.env` (`SYMBOLS`, `TIMEFRAME`/`TIMEFRAMES`).

## Commands
```bash
# Full historical backfill
python src/download_data.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --mode full

# Verify coverage and gaps
python src/data_loader.py --gap-check --no-prompt
python src/data_quality_service.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h

# Fill detected gaps
python src/data_gap_fill.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h
```

## Verification
`data_coverage` is current per symbol/timeframe; `data_gaps` has no unexplained gaps; latest
candle timestamp is recent.

## Red-lines
UTC only; no duplicates per `(symbol,timeframe,datetime_utc)`; idempotent; never delete history.
