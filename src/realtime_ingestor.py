import argparse
import json
import time

from config import BOT_POLL_SECONDS, SYMBOLS, TIMEFRAME
from data_quality_service import run_quality_checks
from trading_bot import sync_latest_from_binance


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime/incremental Binance kline ingestor into SQLite.")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--recent-bars", type=int, default=200)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=BOT_POLL_SECONDS)
    return parser.parse_args()


def _symbols(raw):
    return [s.upper() for s in raw] if raw else [s.upper() for s in SYMBOLS]


def run_once(symbols: list[str], timeframe: str, recent_bars: int) -> dict:
    sync = sync_latest_from_binance(symbols=symbols, timeframe=timeframe, recent_bars=recent_bars)
    quality = run_quality_checks(symbols=symbols, timeframe=timeframe)
    return {"sync": sync, "quality": quality}


def main():
    args = parse_args()
    symbols = _symbols(args.symbols)
    if not args.loop:
        print(json.dumps(run_once(symbols, args.timeframe, args.recent_bars), ensure_ascii=True, indent=2))
        return
    while True:
        print(json.dumps(run_once(symbols, args.timeframe, args.recent_bars), ensure_ascii=True, indent=2))
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
