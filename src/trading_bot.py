import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from broker_client import BinanceCredentialsError, BinanceSpotClient
from config import (
    DB_FILE,
    DRY_RUN,
    ENABLE_TRADING,
    FEATURES_TABLE,
    LOGS_DIR,
    LOOKAHEAD_BARS,
    ENABLE_MODEL_POOL_MAINTENANCE,
    MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS,
    MODEL_POOL_TRAINING_ENABLED_IN_BOT,
    MODEL_POOL_VALIDATION_MAX_FOLDS,
    MODEL_SELECTION_ACCEPTANCE_ORDER,
    PAPER_INITIAL_CASH_USDT,
    PREFER_ACTIVE_MODEL,
    REPORTS_DIR,
    SL_MULTIPLIER,
    SIGNALS_TABLE,
    SYMBOLS,
    TARGET_ACCEPTED_MODELS,
    TIMEFRAME,
    TP_MULTIPLIER,
)
from data_loader import (
    compute_gaps_for_symbol,
    replace_gaps_for_symbol,
    update_price_coverage,
    upsert_prices,
)
from db_utils import init_research_tables, ensure_project_directories, refresh_coverage_from_table
from download_data import normalize_klines, save_raw_snapshot
from download_data import get_latest_datetime_from_db
from execution_engine import ExecutionEngine
from feature_store import run_feature_store
from model_maintenance import maintain_model_pool
from model_registry import get_latest_model, get_model_by_id, list_accepted_models, select_model_for_inference
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from signal_engine import generate_signal_from_probabilities


LOGGER = logging.getLogger("trading_bot")


def _parse_symbols(raw: list[str] | None) -> list[str]:
    if not raw:
        return [s.upper() for s in SYMBOLS]
    out: list[str] = []
    for item in raw:
        out.extend(s.strip().upper() for s in item.split(",") if s.strip())
    return sorted(set(out))


def parse_args():
    parser = argparse.ArgumentParser(description="Safe local autonomous Binance Spot paper-trading bot.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-once", action="store_true", help="Run one full paper-trading cycle.")
    mode.add_argument("--loop", action="store_true", help="Run continuously until Ctrl+C.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--symbols", nargs="*", default=None, help="Comma or space separated symbols.")
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--paper-initial-cash", type=float, default=PAPER_INITIAL_CASH_USDT)
    parser.add_argument("--sync-latest-from-binance", action="store_true")
    parser.add_argument("--sync-recent-bars", type=int, default=200)
    parser.add_argument("--refresh-features", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--target-accepted-models", type=int, default=TARGET_ACCEPTED_MODELS)
    parser.add_argument("--maintain-model-pool", action="store_true", default=ENABLE_MODEL_POOL_MAINTENANCE)
    parser.add_argument("--skip-model-maintenance", action="store_true")
    parser.add_argument("--model-maintenance-max-attempts", type=int, default=MODEL_POOL_MAX_TRAINING_ATTEMPTS_PER_CYCLE)
    parser.add_argument("--model-maintenance-validation-folds", type=int, default=MODEL_POOL_VALIDATION_MAX_FOLDS)
    parser.add_argument("--model-maintenance-interval-seconds", type=int, default=MODEL_POOL_MAINTENANCE_INTERVAL_SECONDS)
    parser.add_argument("--disable-ensemble", action="store_true", help="Use one selected model instead of accepted-model ensemble.")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "trading_bot.log", encoding="utf-8"),
        ],
    )


def _assert_safety() -> None:
    if ENABLE_TRADING:
        raise RuntimeError("ENABLE_TRADING is True. This bot refuses to run live in this step.")
    if not DRY_RUN:
        raise RuntimeError("DRY_RUN is False. This bot only supports dry-run/paper execution in this step.")


def _resolve_model(model_id_arg: str | None, timeframe: str) -> tuple[str, Path]:
    if model_id_arg:
        record = get_model_by_id(model_id_arg)
        if not record:
            raise ValueError(f"Model id not found in registry: {model_id_arg}")
        model_path = Path(record["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        return str(record["model_id"]), model_path

    preferred = select_model_for_inference(
        timeframe=timeframe,
        acceptance_order=MODEL_SELECTION_ACCEPTANCE_ORDER,
        prefer_active=PREFER_ACTIVE_MODEL,
    )
    if preferred:
        model_path = Path(preferred["model_path"])
        if model_path.exists():
            return str(preferred["model_id"]), model_path

    latest = get_latest_model(timeframe=timeframe, prefer_active=PREFER_ACTIVE_MODEL)
    if latest:
        model_path = Path(latest["model_path"])
        if model_path.exists():
            return str(latest["model_id"]), model_path

    raise FileNotFoundError("No valid model found for trading_bot.")


def _resolve_model_pool(args) -> list[tuple[str, Path]]:
    if args.model_id or args.disable_ensemble:
        model_id, model_path = _resolve_model(args.model_id, args.timeframe)
        return [(model_id, model_path)]

    pool: list[tuple[str, Path]] = []
    for record in list_accepted_models(timeframe=args.timeframe, limit=max(1, int(args.target_accepted_models))):
        path = Path(str(record["model_path"]))
        if path.exists():
            pool.append((str(record["model_id"]), path))

    if pool:
        return pool

    model_id, model_path = _resolve_model(args.model_id, args.timeframe)
    return [(model_id, model_path)]


def _load_latest_features(symbol: str, timeframe: str, base_feature_columns: list[str]) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    try:
        not_null = " AND ".join([f"{col} IS NOT NULL" for col in base_feature_columns])
        df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, close, {", ".join(base_feature_columns)}
            FROM {FEATURES_TABLE}
            WHERE symbol = ? AND timeframe = ? AND {not_null}
            ORDER BY datetime_utc DESC
            LIMIT 1
            """,
            conn,
            params=(symbol, timeframe),
        )
    finally:
        conn.close()

    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    for col in base_feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime_utc", "close"] + base_feature_columns).copy()


def _save_signal(
    model_id: str,
    symbol: str,
    timeframe: str,
    datetime_utc: pd.Timestamp,
    signal_position: int,
    prob_short: float,
    prob_flat: float,
    prob_long: float,
) -> None:
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {SIGNALS_TABLE} (
                model_id, symbol, timeframe, datetime_utc, signal_position, signal_label,
                prob_short, prob_flat, prob_long, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                symbol,
                timeframe,
                datetime_utc.isoformat(),
                int(signal_position),
                {1: "LONG", 0: "FLAT", -1: "SHORT"}[int(signal_position)],
                float(prob_short),
                float(prob_flat),
                float(prob_long),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def sync_latest_from_binance(symbols: list[str], timeframe: str, recent_bars: int) -> dict:
    client = BinanceSpotClient.market_data_client()
    summary: dict[str, dict] = {}

    try:
        health = client.healthcheck()
        LOGGER.info("Binance public healthcheck ok: %s", health)
    except Exception as exc:
        LOGGER.error("Binance public healthcheck failed: %s", exc)
        return {symbol: {"status": "ERROR", "reason": str(exc)} for symbol in symbols}

    for symbol in symbols:
        try:
            raw = client.recent_klines(symbol=symbol, interval=timeframe, limit=recent_bars)
            df = normalize_klines(symbol=symbol, timeframe=timeframe, raw_klines=raw)
            if df.empty:
                summary[symbol] = {"status": "EMPTY", "rows": 0}
                continue

            latest_dt = get_latest_datetime_from_db(symbol=symbol, timeframe=timeframe)
            if latest_dt is not None:
                df_new = df[df["datetime_utc"] > latest_dt].copy()
            else:
                df_new = df.copy()

            if df_new.empty:
                summary[symbol] = {
                    "status": "NO_NEW_BARS",
                    "rows_fetched": int(len(df)),
                    "latest_db_datetime_utc": latest_dt.isoformat() if latest_dt is not None else None,
                }
                continue

            snapshot_path = save_raw_snapshot(symbol=symbol, timeframe=timeframe, df=df_new, dry_run=False)
            rows_upserted = upsert_prices(df_new)
            update_price_coverage(symbol, timeframe)
            gaps_df = compute_gaps_for_symbol(symbol, timeframe)
            replace_gaps_for_symbol(symbol, timeframe, gaps_df)
            refresh_coverage_from_table("prices", "prices", symbol, timeframe)

            summary[symbol] = {
                "status": "OK",
                "rows_fetched": int(len(df)),
                "new_rows": int(len(df_new)),
                "rows_upserted": int(rows_upserted),
                "snapshot_path": str(snapshot_path),
                "min_datetime_utc": df_new["datetime_utc"].min().isoformat(),
                "max_datetime_utc": df_new["datetime_utc"].max().isoformat(),
                "gaps_after_sync": int(len(gaps_df)),
                "missing_bars_after_sync": int(gaps_df["missing_bars"].sum()) if not gaps_df.empty else 0,
            }
            LOGGER.info("[%s] synced latest candles: %s", symbol, summary[symbol])
        except Exception as exc:
            LOGGER.exception("[%s] sync failed", symbol)
            summary[symbol] = {"status": "ERROR", "reason": str(exc)}
    return summary


def refresh_recent_features(symbols: list[str], timeframe: str) -> None:
    args = SimpleNamespace(
        lookahead_bars=LOOKAHEAD_BARS,
        tp_multiplier=TP_MULTIPLIER,
        sl_multiplier=SL_MULTIPLIER,
        recalc_overlap_bars=240,
        warmup_bars=240,
        full_rebuild=False,
    )
    run_feature_store(symbols=symbols, timeframe=timeframe, args=args)


def run_once(args) -> dict:
    _assert_safety()
    ensure_project_directories()
    init_research_tables()

    symbols = _parse_symbols(args.symbols)
    sync_summary = {}
    if args.sync_latest_from_binance:
        sync_summary = sync_latest_from_binance(symbols=symbols, timeframe=args.timeframe, recent_bars=args.sync_recent_bars)

    should_refresh_features = bool(args.refresh_features)
    if args.sync_latest_from_binance and args.refresh_features:
        should_refresh_features = any(
            row.get("status") == "OK" and int(row.get("new_rows", 0)) > 0
            for row in sync_summary.values()
            if isinstance(row, dict)
        )
        if not should_refresh_features:
            LOGGER.info("Skipping feature refresh: no new Binance bars were inserted.")

    if should_refresh_features:
        refresh_recent_features(symbols=symbols, timeframe=args.timeframe)

    maintenance_summary = {}
    if (
        not args.model_id
        and args.maintain_model_pool
        and not args.skip_model_maintenance
        and MODEL_POOL_TRAINING_ENABLED_IN_BOT
    ):
        LOGGER.info(
            "Maintaining accepted model pool target=%s max_attempts=%s",
            args.target_accepted_models,
            args.model_maintenance_max_attempts,
        )
        maintenance_summary = maintain_model_pool(
            symbols=symbols,
            timeframe=args.timeframe,
            target_accepted_models=args.target_accepted_models,
            max_attempts=args.model_maintenance_max_attempts,
            validation_max_folds=args.model_maintenance_validation_folds,
        )
        LOGGER.info("Model maintenance summary: %s", maintenance_summary)

    model_pool = _resolve_model_pool(args)
    artifacts = []
    for model_id, model_path in model_pool:
        artifact = joblib.load(model_path)
        artifacts.append((model_id, model_path, artifact))
    execution_model_id = "ensemble:" + ",".join(model_id for model_id, _, _ in artifacts)

    portfolio_manager = PortfolioManager(
        timeframe=args.timeframe,
        dry_run=True,
        initial_cash=float(args.paper_initial_cash),
    )
    risk_manager = RiskManager()
    execution_engine = ExecutionEngine(portfolio_manager=portfolio_manager, risk_manager=risk_manager)

    price_map: dict[str, float] = {}
    symbol_reports: list[dict] = []

    for symbol in symbols:
        model_votes = []
        close_price = None
        dt = None

        for model_id, model_path, artifact in artifacts:
            model = artifact["model"]
            feature_columns = artifact["feature_columns"]
            base_feature_columns = artifact["base_feature_columns"]
            symbol_mapping = artifact["symbol_mapping"]

            if symbol not in symbol_mapping:
                continue

            latest = _load_latest_features(symbol=symbol, timeframe=args.timeframe, base_feature_columns=base_feature_columns)
            if latest.empty:
                continue

            latest = latest.copy()
            latest["symbol_code"] = int(symbol_mapping[symbol])
            X = latest[feature_columns]
            probas = model.predict_proba(X)

            vote_dt = latest.iloc[0]["datetime_utc"]
            vote_close = float(latest.iloc[0]["close"])
            model_votes.append(
                {
                    "model_id": model_id,
                    "model_path": str(model_path),
                    "datetime_utc": vote_dt.isoformat(),
                    "close_price": vote_close,
                    "probabilities": [float(probas[0, 0]), float(probas[0, 1]), float(probas[0, 2])],
                }
            )
            if dt is None or vote_dt > dt:
                dt = vote_dt
                close_price = vote_close

        if not model_votes:
            symbol_reports.append({"symbol": symbol, "status": "SKIPPED", "reason": "symbol_not_in_model_mapping"})
            continue

        avg_probas = np.array([vote["probabilities"] for vote in model_votes], dtype=float).mean(axis=0)
        prob_short = float(avg_probas[0])
        prob_flat = float(avg_probas[1])
        prob_long = float(avg_probas[2])
        price_map[symbol] = close_price

        signal = generate_signal_from_probabilities(
            prob_short=prob_short,
            prob_flat=prob_flat,
            prob_long=prob_long,
        )
        signal["datetime_utc"] = dt.isoformat()

        _save_signal(
            model_id=execution_model_id,
            symbol=symbol,
            timeframe=args.timeframe,
            datetime_utc=dt,
            signal_position=signal["final_signal_position"],
            prob_short=prob_short,
            prob_flat=prob_flat,
            prob_long=prob_long,
        )

        execution = execution_engine.execute_signal(
            model_id=execution_model_id,
            symbol=symbol,
            timeframe=args.timeframe,
            signal_payload=signal,
            market_price=close_price,
        )

        symbol_reports.append(
            {
                "symbol": symbol,
                "status": "OK",
                "datetime_utc": dt.isoformat(),
                "close_price": close_price,
                "model_votes": model_votes,
                "signal": signal,
                "execution": execution,
            }
        )

    portfolio_state = portfolio_manager.snapshot(price_by_symbol=price_map)

    run_report = {
        "mode": "paper_dry_run",
        "model_id": execution_model_id,
        "model_pool": [{"model_id": model_id, "model_path": str(model_path)} for model_id, model_path, _ in artifacts],
        "timeframe": args.timeframe,
        "dry_run": DRY_RUN,
        "enable_trading": ENABLE_TRADING,
        "symbols": symbols,
        "sync_summary": sync_summary,
        "maintenance_summary": maintenance_summary,
        "symbol_reports": symbol_reports,
        "portfolio_state": {
            "cash": portfolio_state.cash,
            "equity": portfolio_state.equity,
            "realized_pnl": portfolio_state.realized_pnl,
            "unrealized_pnl": portfolio_state.unrealized_pnl,
            "daily_realized_pnl": portfolio_state.daily_realized_pnl,
            "exposure_by_symbol": portfolio_state.exposure_by_symbol,
            "positions": portfolio_state.positions,
        },
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.report_path) if args.report_path else REPORTS_DIR / f"trading_bot_run_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_report, ensure_ascii=True, indent=2), encoding="utf-8")
    run_report["report_path"] = str(report_path)
    LOGGER.info("trading_bot run completed: %s", report_path)
    return run_report


def main():
    args = parse_args()
    configure_logging(args.log_level)

    if not args.run_once and not args.loop:
        args.run_once = True

    if args.run_once:
        report = run_once(args)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return

    LOGGER.info("Starting continuous paper loop. poll_seconds=%s", args.poll_seconds)
    next_model_maintenance_ts = 0.0
    try:
        while True:
            cycle_args = SimpleNamespace(**vars(args))
            now = time.time()
            if now < next_model_maintenance_ts:
                cycle_args.skip_model_maintenance = True
            else:
                cycle_args.skip_model_maintenance = bool(args.skip_model_maintenance)
                next_model_maintenance_ts = now + max(60, int(args.model_maintenance_interval_seconds))
            try:
                report = run_once(cycle_args)
                LOGGER.info("Loop cycle done. report=%s", report.get("report_path"))
            except Exception:
                LOGGER.exception("Loop cycle failed; bot will continue after poll interval.")
            time.sleep(max(1, int(args.poll_seconds)))
    except KeyboardInterrupt:
        LOGGER.info("Ctrl+C received. trading_bot stopped cleanly.")


if __name__ == "__main__":
    main()
