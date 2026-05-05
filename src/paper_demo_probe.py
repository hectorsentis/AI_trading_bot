"""Place a tiny Binance Spot Demo/Testnet paper order through the real bot stack.

This is an operational smoke test for paper trading only. It refuses to use
real-account clients and still routes the order through RiskManager,
ExecutionEngine, PortfolioManager and SQLite audit tables.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from broker_client import BinanceSpotClient
from config import (
    ACCOUNT_MODE_LOCAL_PAPER,
    ACCOUNT_MODE_TESTNET_PAPER,
    DRY_RUN,
    ENABLE_LIVE_TRADING,
    ENABLE_REAL_BINANCE_ACCOUNT,
    ENABLE_REAL_ORDER_EXECUTION,
    ENABLE_TESTNET_PAPER_TRADING,
    PAPER_INITIAL_CASH_USDT,
    TIMEFRAME,
)
from db_utils import init_research_tables
from execution_engine import ExecutionEngine
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe paper/demo order smoke test.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--side", choices=["BUY", "FLAT"], default="BUY")
    parser.add_argument("--initial-cash", type=float, default=PAPER_INITIAL_CASH_USDT)
    parser.add_argument("--local-simulated", action="store_true", help="Do not call Binance; use local simulated fill.")
    return parser.parse_args()


def assert_never_real() -> None:
    if ENABLE_LIVE_TRADING or ENABLE_REAL_ORDER_EXECUTION or ENABLE_REAL_BINANCE_ACCOUNT or not DRY_RUN:
        raise RuntimeError(
            "Paper demo probe refuses to run unless real trading flags are disabled and DRY_RUN=true."
        )


def run_probe(args: argparse.Namespace) -> dict:
    assert_never_real()
    init_research_tables()

    symbol = args.symbol.upper().strip()
    model_id = args.model_id or f"paper_demo_probe_{symbol.lower()}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    account_mode = ACCOUNT_MODE_LOCAL_PAPER if args.local_simulated else ACCOUNT_MODE_TESTNET_PAPER
    broker = None
    if account_mode == ACCOUNT_MODE_TESTNET_PAPER:
        if not ENABLE_TESTNET_PAPER_TRADING:
            raise RuntimeError("ENABLE_TESTNET_PAPER_TRADING=false; demo/testnet paper order is blocked.")
        broker = BinanceSpotClient.testnet_execution_client()

    market_client = BinanceSpotClient.market_data_client()
    price_payload = market_client.ticker_price(symbol)
    market_price = float(price_payload["price"])

    signal_position = 1 if args.side == "BUY" else 0
    signal_payload = {
        "datetime_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "final_signal_position": signal_position,
        "final_signal_label": "LONG" if signal_position > 0 else "FLAT",
        "research_signal_label": "LONG" if signal_position > 0 else "FLAT",
        "probabilities": {"short": 0.0, "flat": 0.01 if signal_position else 0.99, "long": 0.99 if signal_position else 0.01},
        "confidence": 0.99,
        "accepted": True,
        "reasons": ["paper_demo_probe"],
        "spot_policy": "Spot demo probe is long/flat only and cannot open real or short exposure.",
    }

    portfolio = PortfolioManager(
        timeframe=args.timeframe,
        dry_run=True,
        initial_cash=float(args.initial_cash),
        model_id=model_id,
        account_mode=account_mode,
    )
    risk = RiskManager(model_id=model_id, account_mode=account_mode)
    engine = ExecutionEngine(
        portfolio_manager=portfolio,
        risk_manager=risk,
        account_mode=account_mode,
        broker_client=broker,
    )
    execution = engine.execute_signal(
        model_id=model_id,
        symbol=symbol,
        timeframe=args.timeframe,
        signal_payload=signal_payload,
        market_price=market_price,
    )
    state = portfolio.snapshot(price_by_symbol={symbol: market_price})

    return {
        "ok": execution.get("status") == "FILLED",
        "symbol": symbol,
        "timeframe": args.timeframe,
        "model_id": model_id,
        "account_mode": account_mode,
        "market_price": market_price,
        "execution": execution,
        "portfolio": {
            "cash": state.cash,
            "equity": state.equity,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": state.unrealized_pnl,
            "positions": state.positions,
        },
        "safety": {
            "DRY_RUN": DRY_RUN,
            "ENABLE_LIVE_TRADING": ENABLE_LIVE_TRADING,
            "ENABLE_REAL_ORDER_EXECUTION": ENABLE_REAL_ORDER_EXECUTION,
            "ENABLE_REAL_BINANCE_ACCOUNT": ENABLE_REAL_BINANCE_ACCOUNT,
            "ENABLE_TESTNET_PAPER_TRADING": ENABLE_TESTNET_PAPER_TRADING,
        },
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(run_probe(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
