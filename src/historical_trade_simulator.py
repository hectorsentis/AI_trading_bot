from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from capital_allocator import CapitalAllocator
from config import (
    DB_FILE,
    DEFAULT_QUOTE_SIZE_USDT,
    FILLS_TABLE,
    MIN_ORDER_NOTIONAL_USDT,
    ORDERS_TABLE,
    PAPER_FEE_RATE,
    PAPER_INITIAL_CASH_USDT,
    PAPER_POSITION_STEP_SIZE,
    PAPER_SLIPPAGE_BPS,
    PORTFOLIO_SNAPSHOTS_TABLE,
    PRICES_TABLE,
    REPORTS_DIR,
    SYMBOLS,
    TIMEFRAME,
    TRADES_TABLE,
    VALIDATION_PREDICTIONS_TABLE,
)
from db_utils import ensure_project_directories, init_research_tables, save_portfolio_snapshot
from ledger import create_trade_record, mark_trade_failed, mark_trade_open, refresh_model_performance
from portfolio_manager import PortfolioManager
from prediction_engine import build_prediction_from_probabilities, build_prediction_from_row_fields, persist_prediction
from risk_manager import RiskManager
from trade_builder import BuiltTrade, build_trade_from_allocation
from trade_proposal_engine import build_trade_proposal, persist_trade_proposal
from validation_funnel_report import ValidationFunnelRecorder, write_funnel_report


HISTORICAL_ACCOUNT_MODE_PREFIX = "historical_sim"


@dataclass
class SimulatedOpenTrade:
    built_trade: BuiltTrade
    entry_time_utc: pd.Timestamp
    entry_fill_price: float
    entry_fee_usdt: float
    entry_slippage_usdt: float
    order_id: str
    fill_id: str
    bars_held: int = 0
    max_unrealized_pnl_usdt: float = 0.0
    min_unrealized_pnl_usdt: float = 0.0
    max_drawdown_usdt: float = 0.0


@dataclass
class HistoricalSimulationResult:
    simulation_run_id: str
    model_id: str
    timeframe: str
    account_mode: str
    start_datetime_utc: str | None
    end_datetime_utc: str | None
    metrics: dict
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    artifacts: dict = field(default_factory=dict)
    funnel_report: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "simulation_run_id": self.simulation_run_id,
            "model_id": self.model_id,
            "timeframe": self.timeframe,
            "account_mode": self.account_mode,
            "start_datetime_utc": self.start_datetime_utc,
            "end_datetime_utc": self.end_datetime_utc,
            "metrics": self.metrics,
            "funnel_report": self.funnel_report,
            "artifacts": self.artifacts,
        }


def _to_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _order_id(prefix: str = "hist") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _fill_id() -> str:
    return f"fill_{uuid.uuid4().hex}"


def _simulated_fill_price(side: str, reference_price: float) -> float:
    slip = float(PAPER_SLIPPAGE_BPS) / 10_000.0
    if side.upper() == "BUY":
        return float(reference_price) * (1.0 + slip)
    return float(reference_price) * (1.0 - slip)


def load_market_ohlc(
    *,
    timeframe: str,
    symbols: list[str],
    start_datetime_utc: str | None = None,
    end_datetime_utc: str | None = None,
) -> pd.DataFrame:
    init_research_tables()
    where = ["timeframe = ?"]
    params: list[object] = [timeframe]
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        where.append(f"symbol IN ({placeholders})")
        params.extend([s.upper() for s in symbols])
    if start_datetime_utc:
        where.append("datetime_utc >= ?")
        params.append(_to_utc(start_datetime_utc).isoformat())
    if end_datetime_utc:
        where.append("datetime_utc <= ?")
        params.append(_to_utc(end_datetime_utc).isoformat())

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT symbol, timeframe, datetime_utc, open, high, low, close, volume
            FROM {PRICES_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY datetime_utc, symbol
            """,
            conn,
            params=tuple(params),
        )
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["symbol", "timeframe", "datetime_utc", "open", "high", "low", "close"]).sort_values(
        ["datetime_utc", "symbol"]
    )


def load_validation_predictions_for_simulation(
    *,
    model_id: str,
    timeframe: str,
    validation_run_id: str | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    init_research_tables()
    where = ["model_id = ?", "timeframe = ?"]
    params: list[object] = [model_id, timeframe]
    if validation_run_id:
        where.append("validation_run_id = ?")
        params.append(validation_run_id)
    if symbols:
        placeholders = ", ".join(["?"] * len(symbols))
        where.append(f"symbol IN ({placeholders})")
        params.extend([s.upper() for s in symbols])

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT model_id, validation_run_id, symbol, timeframe, datetime_utc,
                   prob_short, prob_flat, prob_long, signal_position, fold_id,
                   expected_return_pct, expected_move_pct, expected_adverse_move_pct,
                   q05_return_pct, q25_return_pct, q50_return_pct, q75_return_pct, q95_return_pct,
                   expected_mfe_pct, expected_mae_pct, native_horizon_bars, has_native_prediction
            FROM {VALIDATION_PREDICTIONS_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY datetime_utc, symbol, fold_id
            """,
            conn,
            params=tuple(params),
        )
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    numeric_cols = [
        "prob_short", "prob_flat", "prob_long", "signal_position", "fold_id",
        "expected_return_pct", "expected_move_pct", "expected_adverse_move_pct",
        "q05_return_pct", "q25_return_pct", "q50_return_pct", "q75_return_pct", "q95_return_pct",
        "expected_mfe_pct", "expected_mae_pct", "native_horizon_bars", "has_native_prediction",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["datetime_utc", "prob_short", "prob_flat", "prob_long"]).sort_values(
        ["datetime_utc", "symbol", "fold_id"]
    )


def _insert_historical_order(
    *,
    built_trade: BuiltTrade,
    account_mode: str,
    side: str,
    status: str,
    reason: str,
    requested_price: float,
    fill_price: float | None,
    quantity: float,
    timestamp_utc: pd.Timestamp,
    raw: dict | None = None,
) -> str:
    order_id = _order_id()
    notional = abs(float(quantity) * float(fill_price if fill_price is not None else requested_price))
    ts = _to_utc(timestamp_utc).isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {ORDERS_TABLE} (
                order_id, client_order_id, exchange_order_id, model_id, prediction_id, proposal_id,
                allocation_id, trade_id, symbol, timeframe, timestamp_utc, signal_datetime_utc,
                account_mode, side, executable_action, order_type, type, quantity, requested_price,
                price_requested, fill_price, price_filled, notional, status, reason,
                take_profit_price, stop_loss_price, protection_required, protection_status,
                decision_json, dry_run, created_at_utc, updated_at_utc, filled_at_utc,
                raw_exchange_response_json
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'MARKET', 'MARKET', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'HISTORICAL_VIRTUAL_TP_SL', ?, 1, ?, ?, ?, ?)
            """,
            (
                order_id,
                f"{built_trade.trade_id}_{side.lower()}",
                built_trade.model_id,
                built_trade.prediction_id,
                built_trade.proposal_id,
                built_trade.allocation_id,
                built_trade.trade_id,
                built_trade.symbol,
                built_trade.timeframe,
                ts,
                ts,
                account_mode,
                side.upper(),
                side.upper(),
                float(quantity),
                float(requested_price),
                float(requested_price),
                float(fill_price) if fill_price is not None else None,
                float(fill_price) if fill_price is not None else None,
                float(notional),
                status,
                reason,
                float(built_trade.tp_price),
                float(built_trade.sl_price),
                json.dumps(raw or {}, ensure_ascii=True, sort_keys=True),
                ts,
                ts,
                ts if fill_price is not None and status == "FILLED" else None,
                json.dumps(raw or {}, ensure_ascii=True, sort_keys=True),
            ),
        )
        conn.commit()
    return order_id


def _insert_historical_fill(
    *,
    built_trade: BuiltTrade,
    order_id: str,
    account_mode: str,
    quantity: float,
    price: float,
    commission: float,
    slippage_usdt: float,
    timestamp_utc: pd.Timestamp,
    raw: dict | None = None,
) -> str:
    fill_id = _fill_id()
    ts = _to_utc(timestamp_utc).isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {FILLS_TABLE} (
                fill_id, order_id, model_id, prediction_id, proposal_id, allocation_id, trade_id,
                symbol, timeframe, account_mode, quantity, price, commission, fee_usdt,
                commission_asset, slippage_usdt, timestamp_utc, raw_exchange_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'USDT', ?, ?, ?)
            """,
            (
                fill_id,
                order_id,
                built_trade.model_id,
                built_trade.prediction_id,
                built_trade.proposal_id,
                built_trade.allocation_id,
                built_trade.trade_id,
                built_trade.symbol,
                built_trade.timeframe,
                account_mode,
                float(quantity),
                float(price),
                float(commission),
                float(commission),
                float(slippage_usdt),
                ts,
                json.dumps(raw or {}, ensure_ascii=True, sort_keys=True),
            ),
        )
        conn.commit()
    return fill_id


def _close_trade_record(
    *,
    open_trade: SimulatedOpenTrade,
    exit_time_utc: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    exit_fee_usdt: float,
    exit_slippage_usdt: float,
    realized_pnl_usdt: float,
) -> None:
    bt = open_trade.built_trade
    ts = _to_utc(exit_time_utc).isoformat()
    total_fees = float(open_trade.entry_fee_usdt) + float(exit_fee_usdt)
    total_slippage = float(open_trade.entry_slippage_usdt) + float(exit_slippage_usdt)
    raw = {
        "source": "historical_trade_simulator",
        "entry_order_id": open_trade.order_id,
        "entry_fill_id": open_trade.fill_id,
        "bars_held": int(open_trade.bars_held),
        "max_unrealized_pnl_usdt": float(open_trade.max_unrealized_pnl_usdt),
        "min_unrealized_pnl_usdt": float(open_trade.min_unrealized_pnl_usdt),
        "trade_max_drawdown_usdt": float(open_trade.max_drawdown_usdt),
    }
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            f"""
            UPDATE {TRADES_TABLE}
            SET status='CLOSED', exit_price=?, avg_exit_price=?, closed_at_utc=?,
                exit_reason=?, realized_pnl_usdt=?, unrealized_pnl_usdt=0,
                fees_usdt=?, slippage_usdt=?, raw_json=?, updated_at_utc=?
            WHERE trade_id=?
            """,
            (
                float(exit_price),
                float(exit_price),
                ts,
                exit_reason,
                float(realized_pnl_usdt),
                total_fees,
                total_slippage,
                json.dumps(raw, ensure_ascii=True, sort_keys=True),
                ts,
                bt.trade_id,
            ),
        )
        conn.commit()


class HistoricalTradeSimulator:
    """Historical mirror of the paper/live trade lifecycle.

    The simulator intentionally routes every prediction through the same
    proposal, allocator, trade-builder, risk-manager and ledger contracts used
    by paper/live trading. Only the execution venue is replaced by deterministic
    historical fills from OHLC candles with configured fee/slippage assumptions.
    """

    def __init__(
        self,
        *,
        model_id: str,
        timeframe: str,
        account_mode: str | None = None,
        initial_cash_usdt: float = PAPER_INITIAL_CASH_USDT,
        requested_notional_usdt: float = DEFAULT_QUOTE_SIZE_USDT,
        simulation_run_id: str | None = None,
    ):
        init_research_tables()
        self.model_id = model_id
        self.timeframe = timeframe
        self.simulation_run_id = simulation_run_id or f"hist_{model_id}_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.account_mode = account_mode or f"{HISTORICAL_ACCOUNT_MODE_PREFIX}:{self.simulation_run_id}"
        self.initial_cash_usdt = float(initial_cash_usdt)
        self.requested_notional_usdt = float(requested_notional_usdt)
        self.portfolio_manager = PortfolioManager(
            timeframe=timeframe,
            dry_run=True,
            initial_cash=float(initial_cash_usdt),
            model_id="historical_shared_capital_pool",
            account_mode=self.account_mode,
        )
        self.allocator = CapitalAllocator(account_mode=self.account_mode)
        self.risk_manager = RiskManager(model_id=model_id, account_mode=self.account_mode)
        self.open_trades: list[SimulatedOpenTrade] = []
        self.closed_rows: list[dict] = []
        self.equity_rows: list[dict] = []
        self.funnel = ValidationFunnelRecorder(
            model_id=self.model_id,
            validation_run_id=None,
            simulation_run_id=self.simulation_run_id,
            timeframe=self.timeframe,
        )
        save_portfolio_snapshot(
            cash=self.portfolio_manager.cash,
            equity=self.portfolio_manager.equity,
            realized_pnl=self.portfolio_manager.realized_pnl,
            unrealized_pnl=0.0,
            exposure_by_symbol={},
            dry_run=True,
            model_id="historical_shared_capital_pool",
            account_mode=self.account_mode,
        )

    def _execute_entry(self, built_trade: BuiltTrade, entry_reference_price: float, timestamp_utc: pd.Timestamp) -> SimulatedOpenTrade | None:
        create_trade_record(built_trade, account_mode=self.account_mode, status="APPROVED")
        state = self.portfolio_manager.get_state(price_by_symbol={built_trade.symbol: entry_reference_price})
        risk = self.risk_manager.validate_trade_plan(
            built_trade,
            portfolio_state=state,
            latest_market_timestamp_utc=_to_utc(timestamp_utc).isoformat(),
            evaluation_timestamp_utc=timestamp_utc,
        )
        self.funnel.observe_risk(risk)
        if not risk["approved"]:
            reason = ";".join(risk["reasons"])
            mark_trade_failed(built_trade.trade_id, reason, raw={"risk": risk, "source": "historical_trade_simulator"})
            _insert_historical_order(
                built_trade=built_trade,
                account_mode=self.account_mode,
                side="BUY",
                status="REJECTED",
                reason=reason,
                requested_price=entry_reference_price,
                fill_price=None,
                quantity=built_trade.quantity,
                timestamp_utc=timestamp_utc,
                raw={"risk": risk},
            )
            self.funnel.observe_order(side="BUY", status="REJECTED")
            self.funnel.observe_entry_unfilled("risk_manager_rejected_entry")
            return None

        fill_price = _simulated_fill_price("BUY", entry_reference_price)
        fee = float(built_trade.quantity) * fill_price * float(PAPER_FEE_RATE)
        slippage = abs(fill_price - float(built_trade.entry_reference_price)) * float(built_trade.quantity)
        self.portfolio_manager.apply_fill(
            symbol=built_trade.symbol,
            side="BUY",
            quantity=built_trade.quantity,
            fill_price=fill_price,
            fee_rate=PAPER_FEE_RATE,
        )
        order_id = _insert_historical_order(
            built_trade=built_trade,
            account_mode=self.account_mode,
            side="BUY",
            status="FILLED",
            reason="historical_entry_fill",
            requested_price=entry_reference_price,
            fill_price=fill_price,
            quantity=built_trade.quantity,
            timestamp_utc=timestamp_utc,
            raw={"simulation_run_id": self.simulation_run_id},
        )
        self.funnel.observe_order(side="BUY", status="FILLED")
        fill_id = _insert_historical_fill(
            built_trade=built_trade,
            order_id=order_id,
            account_mode=self.account_mode,
            quantity=built_trade.quantity,
            price=fill_price,
            commission=fee,
            slippage_usdt=slippage,
            timestamp_utc=timestamp_utc,
            raw={"simulation_run_id": self.simulation_run_id},
        )
        self.funnel.observe_fill()
        mark_trade_open(
            trade_id=built_trade.trade_id,
            avg_entry_price=fill_price,
            qty=built_trade.quantity,
            fees_usdt=fee,
            slippage_usdt=slippage,
            raw_update={"order_id": order_id, "fill_id": fill_id, "source": "historical_trade_simulator"},
        )
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute(
                f"UPDATE {TRADES_TABLE} SET opened_at_utc=?, updated_at_utc=? WHERE trade_id=?",
                (_to_utc(timestamp_utc).isoformat(), _to_utc(timestamp_utc).isoformat(), built_trade.trade_id),
            )
            conn.commit()
        self.funnel.observe_entry_filled()
        return SimulatedOpenTrade(
            built_trade=built_trade,
            entry_time_utc=_to_utc(timestamp_utc),
            entry_fill_price=float(fill_price),
            entry_fee_usdt=float(fee),
            entry_slippage_usdt=float(slippage),
            order_id=order_id,
            fill_id=fill_id,
        )

    def _close_open_trade(
        self,
        open_trade: SimulatedOpenTrade,
        *,
        timestamp_utc: pd.Timestamp,
        requested_exit_price: float,
        exit_reason: str,
    ) -> None:
        bt = open_trade.built_trade
        exit_fill = _simulated_fill_price("SELL", requested_exit_price)
        exit_fee = float(bt.quantity) * exit_fill * float(PAPER_FEE_RATE)
        exit_slippage = abs(exit_fill - float(requested_exit_price)) * float(bt.quantity)
        self.portfolio_manager.apply_fill(
            symbol=bt.symbol,
            side="SELL",
            quantity=bt.quantity,
            fill_price=exit_fill,
            fee_rate=PAPER_FEE_RATE,
        )
        order_id = _insert_historical_order(
            built_trade=bt,
            account_mode=self.account_mode,
            side="SELL",
            status="FILLED",
            reason=exit_reason,
            requested_price=requested_exit_price,
            fill_price=exit_fill,
            quantity=bt.quantity,
            timestamp_utc=timestamp_utc,
            raw={"simulation_run_id": self.simulation_run_id, "exit_reason": exit_reason},
        )
        self.funnel.observe_order(side="SELL", status="FILLED")
        fill_id = _insert_historical_fill(
            built_trade=bt,
            order_id=order_id,
            account_mode=self.account_mode,
            quantity=bt.quantity,
            price=exit_fill,
            commission=exit_fee,
            slippage_usdt=exit_slippage,
            timestamp_utc=timestamp_utc,
            raw={"simulation_run_id": self.simulation_run_id, "exit_reason": exit_reason},
        )
        self.funnel.observe_fill()
        realized = float(bt.quantity) * (exit_fill - open_trade.entry_fill_price) - open_trade.entry_fee_usdt - exit_fee
        _close_trade_record(
            open_trade=open_trade,
            exit_time_utc=timestamp_utc,
            exit_price=exit_fill,
            exit_reason=exit_reason,
            exit_fee_usdt=exit_fee,
            exit_slippage_usdt=exit_slippage,
            realized_pnl_usdt=realized,
        )
        self.closed_rows.append(
            {
                "simulation_run_id": self.simulation_run_id,
                "model_id": bt.model_id,
                "prediction_id": bt.prediction_id,
                "proposal_id": bt.proposal_id,
                "allocation_id": bt.allocation_id,
                "trade_id": bt.trade_id,
                "entry_order_id": open_trade.order_id,
                "entry_fill_id": open_trade.fill_id,
                "exit_order_id": order_id,
                "exit_fill_id": fill_id,
                "symbol": bt.symbol,
                "timeframe": bt.timeframe,
                "entry_time_utc": open_trade.entry_time_utc.isoformat(),
                "exit_time_utc": _to_utc(timestamp_utc).isoformat(),
                "entry_price": open_trade.entry_fill_price,
                "exit_price": exit_fill,
                "qty": bt.quantity,
                "approved_notional_usdt": bt.approved_notional_usdt,
                "tp_price": bt.tp_price,
                "sl_price": bt.sl_price,
                "exit_reason": exit_reason,
                "realized_pnl_usdt": realized,
                "fees_usdt": open_trade.entry_fee_usdt + exit_fee,
                "slippage_usdt": open_trade.entry_slippage_usdt + exit_slippage,
                "bars_held": int(open_trade.bars_held),
                "max_unrealized_pnl_usdt": open_trade.max_unrealized_pnl_usdt,
                "min_unrealized_pnl_usdt": open_trade.min_unrealized_pnl_usdt,
                "trade_max_drawdown_usdt": open_trade.max_drawdown_usdt,
            }
        )
        self.funnel.observe_trade_closed(exit_reason)

    def _update_open_trades(self, candle: pd.Series, deterioration_by_symbol: dict[str, dict]) -> None:
        remaining: list[SimulatedOpenTrade] = []
        ts = _to_utc(candle["datetime_utc"])
        symbol = str(candle["symbol"])
        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])
        for ot in self.open_trades:
            bt = ot.built_trade
            if bt.symbol != symbol:
                remaining.append(ot)
                continue
            ot.bars_held += 1
            unrealized = float(bt.quantity) * (close - ot.entry_fill_price)
            ot.max_unrealized_pnl_usdt = max(ot.max_unrealized_pnl_usdt, unrealized)
            ot.min_unrealized_pnl_usdt = min(ot.min_unrealized_pnl_usdt, unrealized)
            ot.max_drawdown_usdt = min(ot.max_drawdown_usdt, unrealized - ot.max_unrealized_pnl_usdt)

            exit_reason = None
            exit_price = None
            # Conservative OHLC path: if TP and SL are both touched in a candle,
            # assume the adverse stop was hit first.
            if low <= float(bt.sl_price):
                exit_reason = "virtual_stop_loss_reached"
                exit_price = float(bt.sl_price)
            elif high >= float(bt.tp_price):
                exit_reason = "virtual_take_profit_reached"
                exit_price = float(bt.tp_price)
            elif ot.bars_held >= int(bt.horizon_bars) or ts >= _to_utc(bt.valid_until_utc):
                exit_reason = "signal_horizon_expired"
                exit_price = close
            else:
                deterioration = deterioration_by_symbol.get(symbol)
                if deterioration and deterioration.get("deteriorated"):
                    exit_reason = deterioration.get("reason", "signal_deterioration_exit")
                    exit_price = close

            if exit_reason and exit_price is not None:
                self._close_open_trade(
                    ot,
                    timestamp_utc=ts,
                    requested_exit_price=float(exit_price),
                    exit_reason=exit_reason,
                )
            else:
                remaining.append(ot)
        self.open_trades = remaining

    def _apply_deterioration_exits_at_close(self, candles: pd.DataFrame, deterioration_by_symbol: dict[str, dict]) -> None:
        if not deterioration_by_symbol or candles.empty or not self.open_trades:
            return
        close_by_symbol = {str(r["symbol"]): float(r["close"]) for _, r in candles.iterrows()}
        ts = _to_utc(candles["datetime_utc"].iloc[0])
        remaining: list[SimulatedOpenTrade] = []
        for ot in self.open_trades:
            deterioration = deterioration_by_symbol.get(ot.built_trade.symbol)
            close = close_by_symbol.get(ot.built_trade.symbol)
            if deterioration and deterioration.get("deteriorated") and close is not None:
                self._close_open_trade(
                    ot,
                    timestamp_utc=ts,
                    requested_exit_price=close,
                    exit_reason=deterioration.get("reason") or "signal_deterioration_exit",
                )
            else:
                remaining.append(ot)
        self.open_trades = remaining

    def _record_equity(self, timestamp_utc: pd.Timestamp, latest_prices: dict[str, float]) -> None:
        state = self.portfolio_manager.snapshot(price_by_symbol=latest_prices)
        self.equity_rows.append(
            {
                "simulation_run_id": self.simulation_run_id,
                "timestamp_utc": _to_utc(timestamp_utc).isoformat(),
                "cash": float(state.cash),
                "equity": float(state.equity),
                "realized_pnl": float(state.realized_pnl),
                "unrealized_pnl": float(state.unrealized_pnl),
                "open_trades": int(len(self.open_trades)),
            }
        )

    def _prediction_from_row(self, row: pd.Series, close_price: float) -> object:
        # Prefer the native, calibrated, cost-adjusted prediction the model produced for this
        # row (persisted per fold). This keeps validation/backtest acceptance consistent with the
        # live paper path instead of recomputing a synthetic distribution.
        native = build_prediction_from_row_fields(
            row,
            model_id=str(row.get("model_id") or self.model_id),
            symbol=str(row["symbol"]),
            timeframe=str(row.get("timeframe") or self.timeframe),
            timestamp_utc=row["datetime_utc"],
        )
        if native is not None:
            return native
        return build_prediction_from_probabilities(
            model_id=str(row.get("model_id") or self.model_id),
            symbol=str(row["symbol"]),
            timeframe=str(row.get("timeframe") or self.timeframe),
            timestamp_utc=row["datetime_utc"],
            close_price=float(close_price),
            prob_short=float(row["prob_short"]),
            prob_flat=float(row["prob_flat"]),
            prob_long=float(row["prob_long"]),
            latest_features=row.to_dict(),
            raw={
                "source": "historical_trade_simulator",
                "validation_run_id": row.get("validation_run_id"),
                "fold_id": int(row["fold_id"]) if "fold_id" in row and not pd.isna(row["fold_id"]) else None,
            },
        )

    @staticmethod
    def _deterioration_from_prediction(prediction) -> dict:
        deteriorated = (
            prediction.direction != "LONG"
            or float(prediction.expected_return_pct) <= 0.0
            or float(prediction.prob_down) > float(prediction.prob_up)
        )
        reason = None
        if prediction.direction != "LONG":
            reason = "signal_direction_no_longer_long"
        elif float(prediction.expected_return_pct) <= 0.0:
            reason = "expected_value_deterioration_exit"
        elif float(prediction.prob_down) > float(prediction.prob_up):
            reason = "adverse_probability_deterioration_exit"
        return {"deteriorated": bool(deteriorated), "reason": reason}

    def run(self, predictions_df: pd.DataFrame, market_df: pd.DataFrame, *, persist_artifacts: bool = True) -> HistoricalSimulationResult:
        if predictions_df.empty:
            raise ValueError("predictions_df is empty")
        if market_df.empty:
            raise ValueError("market_df is empty")

        predictions = predictions_df.copy()
        market = market_df.copy()
        validation_run_id = None
        if "validation_run_id" in predictions.columns and not predictions["validation_run_id"].dropna().empty:
            validation_run_id = str(predictions["validation_run_id"].dropna().astype(str).iloc[0])
        self.funnel.validation_run_id = validation_run_id
        predictions["datetime_utc"] = pd.to_datetime(predictions["datetime_utc"], utc=True, errors="coerce")
        market["datetime_utc"] = pd.to_datetime(market["datetime_utc"], utc=True, errors="coerce")
        predictions = predictions.dropna(subset=["datetime_utc", "symbol"]).sort_values(["datetime_utc", "symbol"])
        self.funnel.observe_validation_predictions(predictions)
        market = market.dropna(subset=["datetime_utc", "symbol", "open", "high", "low", "close"]).sort_values(["datetime_utc", "symbol"])
        close_by_key = {
            (str(r.symbol), _to_utc(r.datetime_utc)): float(r.close)
            for r in market[["symbol", "datetime_utc", "close"]].itertuples(index=False)
        }
        market_by_symbol = {symbol: df.reset_index(drop=True) for symbol, df in market.groupby("symbol", sort=False)}
        all_times = sorted(market["datetime_utc"].dropna().unique().tolist())
        predictions_by_time: dict[pd.Timestamp, list[pd.Series]] = {}
        for _, row in predictions.iterrows():
            predictions_by_time.setdefault(_to_utc(row["datetime_utc"]), []).append(row)

        latest_prices: dict[str, float] = {}
        pending_entries: list[dict] = []

        for raw_ts in all_times:
            ts = _to_utc(raw_ts)
            current_candles = market[market["datetime_utc"] == ts]
            deterioration_by_symbol: dict[str, dict] = {}

            # Fill entries scheduled by predictions from the previous completed
            # candle at the current candle's open. This prevents next-bar leakage.
            still_pending: list[dict] = []
            for pending in pending_entries:
                symbol_df = market_by_symbol.get(pending["symbol"])
                if symbol_df is None:
                    continue
                entry_row = symbol_df[symbol_df["datetime_utc"] == ts]
                if entry_row.empty:
                    still_pending.append(pending)
                    continue
                entry_open = float(entry_row.iloc[0]["open"])
                built_trade = pending["built_trade"]
                entered = self._execute_entry(built_trade, entry_reference_price=entry_open, timestamp_utc=ts)
                if entered is not None:
                    self.open_trades.append(entered)
            pending_entries = still_pending

            # Update exits using this candle after entries. The simulator applies
            # the same virtual TP/SL/horizon concepts used by exit_manager.
            for _, candle in current_candles.iterrows():
                latest_prices[str(candle["symbol"])] = float(candle["close"])
                self._update_open_trades(candle, deterioration_by_symbol)

            # Build predictions/proposals after this candle's close. Accepted
            # trades are scheduled for next bar open.
            for row in predictions_by_time.get(ts, []):
                symbol = str(row["symbol"])
                close = close_by_key.get((symbol, ts))
                if close is None:
                    continue
                prediction = self._prediction_from_row(row, close)
                persist_prediction(prediction)
                self.funnel.observe_model_prediction(prediction)
                deterioration_by_symbol[symbol] = self._deterioration_from_prediction(prediction)
                proposal = build_trade_proposal(
                    prediction=prediction,
                    entry_reference_price=float(close),
                    requested_notional_usdt=self.requested_notional_usdt,
                )
                persist_trade_proposal(proposal)
                self.funnel.observe_proposal_initial(proposal)
                allocation = self.allocator.allocate(proposal, evaluation_timestamp_utc=ts.isoformat())
                self.funnel.observe_allocation(proposal, allocation)
                if allocation.get("decision") not in {"ACCEPT", "RESIZE"}:
                    continue
                built_trade, build_reasons = build_trade_from_allocation(
                    proposal,
                    allocation,
                    step_size=float(PAPER_POSITION_STEP_SIZE),
                )
                self.funnel.observe_trade_builder(built_trade, build_reasons)
                if built_trade is None:
                    # Persist a failed trade shell so proposal/allocation rejection
                    # remains attributable in the operational ledger.
                    failed_trade = SimpleNamespace(trade_id=f"trade_{uuid.uuid4().hex}")
                    with sqlite3.connect(DB_FILE) as conn:
                        conn.execute(
                            f"""
                            INSERT OR REPLACE INTO {TRADES_TABLE} (
                                trade_id, proposal_id, allocation_id, prediction_id, model_id, symbol,
                                timeframe, side, status, account_mode, requested_notional_usdt,
                                approved_notional_usdt, exit_reason, realized_pnl_usdt,
                                unrealized_pnl_usdt, fees_usdt, slippage_usdt, raw_json,
                                created_at_utc, updated_at_utc
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'LONG', 'FAILED', ?, ?, ?, ?, 0, 0, 0, 0, ?, ?, ?)
                            """,
                            (
                                failed_trade.trade_id,
                                proposal.proposal_id,
                                allocation["allocation_id"],
                                proposal.prediction_id,
                                proposal.model_id,
                                proposal.symbol,
                                proposal.timeframe,
                                self.account_mode,
                                proposal.requested_notional_usdt,
                                allocation.get("approved_notional_usdt", 0.0),
                                ";".join(build_reasons),
                                json.dumps({"build_reasons": build_reasons, "source": "historical_trade_simulator"}, ensure_ascii=True),
                                ts.isoformat(),
                                ts.isoformat(),
                            ),
                        )
                        conn.commit()
                    continue
                pending_entries.append({"symbol": symbol, "built_trade": built_trade})
                self.funnel.observe_entry_scheduled()

            self._apply_deterioration_exits_at_close(current_candles, deterioration_by_symbol)
            self._record_equity(ts, latest_prices)

        # Force-close anything still open at the last available close. This is a
        # risk-managed end-of-simulation horizon exit, not a model sell signal.
        if pending_entries:
            for _pending in pending_entries:
                self.funnel.observe_entry_unfilled("no_next_bar_available_for_scheduled_entry")
            pending_entries = []

        if not market.empty and self.open_trades:
            last_ts = _to_utc(market["datetime_utc"].max())
            for ot in list(self.open_trades):
                price = latest_prices.get(ot.built_trade.symbol)
                if price is None:
                    continue
                self._close_open_trade(
                    ot,
                    timestamp_utc=last_ts,
                    requested_exit_price=float(price),
                    exit_reason="simulation_end_forced_flat",
                )
            self.open_trades = []
            self._record_equity(last_ts, latest_prices)

        equity_curve = pd.DataFrame(self.equity_rows)
        trades_df = pd.DataFrame(self.closed_rows)
        metrics = calculate_lifecycle_metrics(
            trades_df=trades_df,
            equity_curve=equity_curve,
            initial_cash_usdt=self.initial_cash_usdt,
            market_df=market,
        )
        funnel_report = self.funnel.final_report(lifecycle_metrics=metrics)
        refresh_model_performance(account_mode=self.account_mode)
        result = HistoricalSimulationResult(
            simulation_run_id=self.simulation_run_id,
            model_id=self.model_id,
            timeframe=self.timeframe,
            account_mode=self.account_mode,
            start_datetime_utc=market["datetime_utc"].min().isoformat() if not market.empty else None,
            end_datetime_utc=market["datetime_utc"].max().isoformat() if not market.empty else None,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades_df,
            funnel_report=funnel_report,
        )
        if persist_artifacts:
            result.artifacts.update(write_simulation_artifacts(result))
            funnel_path = write_funnel_report(funnel_report, prefix="trade_lifecycle_funnel")
            funnel_report["artifact_path"] = str(funnel_path)
            funnel_path.write_text(json.dumps(funnel_report, ensure_ascii=True, indent=2), encoding="utf-8")
            result.funnel_report = funnel_report
            result.artifacts["funnel_json"] = str(funnel_path)
        return result


def calculate_lifecycle_metrics(
    *,
    trades_df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_cash_usdt: float,
    market_df: pd.DataFrame | None = None,
) -> dict:
    initial_cash = float(max(initial_cash_usdt, 1e-9))
    if trades_df.empty:
        return {
            "realized_pnl_usdt": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "expectancy_usdt": 0.0,
            "trade_count": 0,
            "fees_usdt": 0.0,
            "slippage_usdt": 0.0,
            "sharpe": 0.0,
            "buy_hold_return": _buy_hold_return(market_df),
        }
    pnl = pd.to_numeric(trades_df["realized_pnl_usdt"], errors="coerce").fillna(0.0)
    fees = pd.to_numeric(trades_df.get("fees_usdt", 0.0), errors="coerce").fillna(0.0)
    slippage = pd.to_numeric(trades_df.get("slippage_usdt", 0.0), errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(abs(pnl[pnl < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    expectancy = float(pnl.mean()) if len(pnl) else 0.0

    if equity_curve.empty or "equity" not in equity_curve:
        equity = pd.Series([initial_cash + float(pnl.sum())])
    else:
        equity = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna()
        if equity.empty:
            equity = pd.Series([initial_cash + float(pnl.sum())])
    peak = equity.cummax()
    drawdowns = (equity / peak.replace(0, np.nan)) - 1.0
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0
    returns = equity.pct_change().dropna()
    sharpe = float((returns.mean() / returns.std()) * np.sqrt(max(len(returns), 1))) if len(returns) > 1 and returns.std() > 0 else 0.0
    total_return = float((equity.iloc[-1] / initial_cash) - 1.0) if len(equity) else 0.0

    return {
        "realized_pnl_usdt": float(pnl.sum()),
        "total_return": total_return,
        "strategy_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_factor": float(profit_factor),
        "win_rate": win_rate,
        "expectancy_usdt": expectancy,
        "trade_count": int(len(pnl)),
        "fees_usdt": float(fees.sum()),
        "slippage_usdt": float(slippage.sum()),
        "sharpe": sharpe,
        "buy_hold_return": _buy_hold_return(market_df),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict() if "exit_reason" in trades_df else {},
    }


def _buy_hold_return(market_df: pd.DataFrame | None) -> float:
    if market_df is None or market_df.empty:
        return 0.0
    returns = []
    for _, df in market_df.sort_values("datetime_utc").groupby("symbol"):
        if len(df) < 2:
            continue
        first = float(df.iloc[0]["close"])
        last = float(df.iloc[-1]["close"])
        if first > 0:
            returns.append((last / first) - 1.0)
    return float(np.mean(returns)) if returns else 0.0


def write_simulation_artifacts(result: HistoricalSimulationResult) -> dict:
    ensure_project_directories()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"historical_lifecycle_{result.model_id}_{result.simulation_run_id}"
    trades_path = REPORTS_DIR / f"{prefix}_trades.csv"
    equity_path = REPORTS_DIR / f"{prefix}_equity.csv"
    summary_path = REPORTS_DIR / f"{prefix}_summary.json"
    result.trades.to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, index=False)
    summary = result.to_summary()
    summary["artifacts"] = {
        "trades_csv": str(trades_path),
        "equity_csv": str(equity_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary["artifacts"]


def run_historical_trade_lifecycle(
    *,
    predictions_df: pd.DataFrame,
    market_df: pd.DataFrame,
    model_id: str,
    timeframe: str,
    simulation_run_id: str | None = None,
    requested_notional_usdt: float = DEFAULT_QUOTE_SIZE_USDT,
    persist_artifacts: bool = True,
) -> HistoricalSimulationResult:
    simulator = HistoricalTradeSimulator(
        model_id=model_id,
        timeframe=timeframe,
        requested_notional_usdt=requested_notional_usdt,
        simulation_run_id=simulation_run_id,
    )
    return simulator.run(predictions_df=predictions_df, market_df=market_df, persist_artifacts=persist_artifacts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run historical trade-lifecycle simulation from persisted OOS predictions.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--validation-run-id", default=None)
    parser.add_argument("--timeframe", default=TIMEFRAME)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--requested-notional-usdt", type=float, default=DEFAULT_QUOTE_SIZE_USDT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.upper().strip() for s in args.symbols] if args.symbols else [s.upper() for s in SYMBOLS]
    preds = load_validation_predictions_for_simulation(
        model_id=args.model_id,
        timeframe=args.timeframe,
        validation_run_id=args.validation_run_id,
        symbols=symbols,
    )
    if args.start_date:
        preds = preds[preds["datetime_utc"] >= _to_utc(args.start_date)].copy()
    if args.end_date:
        preds = preds[preds["datetime_utc"] <= _to_utc(args.end_date)].copy()
    if preds.empty:
        raise ValueError("No validation predictions found for simulation.")
    market = load_market_ohlc(
        timeframe=args.timeframe,
        symbols=symbols,
        start_datetime_utc=preds["datetime_utc"].min().isoformat(),
        end_datetime_utc=None,
    )
    result = run_historical_trade_lifecycle(
        predictions_df=preds,
        market_df=market,
        model_id=args.model_id,
        timeframe=args.timeframe,
        requested_notional_usdt=args.requested_notional_usdt,
    )
    print(json.dumps(result.to_summary(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
