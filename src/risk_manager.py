import sqlite3
import math

import pandas as pd

from config import (
    DB_FILE,
    ORDERS_TABLE,
    RISK_EVENTS_TABLE,
    DRY_RUN,
    ENABLE_LIVE_TRADING,
    ENABLE_REAL_ORDER_EXECUTION,
    ENABLE_REAL_BINANCE_ACCOUNT,
    DEFAULT_QUOTE_SIZE_USDT,
    PAPER_POSITION_STEP_SIZE,
    PAPER_MIN_NOTIONAL_USDT,
    PAPER_MAX_EXPOSURE_PER_ASSET,
    PAPER_MAX_POSITION_NOTIONAL_USDT,
    PAPER_MAX_NEW_TRADES_PER_DAY,
    PAPER_MAX_DAILY_LOSS_USDT,
    MAX_ORDER_NOTIONAL_USDT,
    ACCOUNT_MODE_REAL,
    REQUIRE_TP_SL_ON_ENTRY,
    MIN_TP_SL_RISK_REWARD,
    MAX_TOTAL_EXPOSURE_USDT,
    MAX_SYMBOL_EXPOSURE_USDT,
    MAX_MODEL_OPEN_EXPOSURE_USDT,
    MAX_TRADE_LOSS_USDT,
    MAX_TRADES_PER_DAY_PER_MODEL,
    MIN_CASH_RESERVE_USDT,
    STALE_DATA_MAX_SECONDS,
    RECONCILIATION_REQUIRED,
    KILL_SWITCH_ENABLED,
    TRADES_TABLE,
    SYSTEM_STATUS_TABLE,
)
from trade_protection import validate_long_protection


class RiskManager:
    def __init__(self, model_id: str | None = None, account_mode: str = "local_paper"):
        self.dry_run = DRY_RUN
        self.enable_live_trading = ENABLE_LIVE_TRADING
        self.enable_real_order_execution = ENABLE_REAL_ORDER_EXECUTION
        self.enable_real_binance_account = ENABLE_REAL_BINANCE_ACCOUNT
        self.model_id = model_id
        self.account_mode = account_mode
        self.default_quote_size_usdt = float(DEFAULT_QUOTE_SIZE_USDT)
        self.position_step_size = float(PAPER_POSITION_STEP_SIZE)
        self.min_notional_usdt = float(PAPER_MIN_NOTIONAL_USDT)
        self.max_exposure_per_asset = float(PAPER_MAX_EXPOSURE_PER_ASSET)
        self.max_position_notional_usdt = float(PAPER_MAX_POSITION_NOTIONAL_USDT)
        self.max_new_trades_per_day = int(PAPER_MAX_NEW_TRADES_PER_DAY)
        self.max_daily_loss_usdt = float(PAPER_MAX_DAILY_LOSS_USDT)
        self.max_order_notional_usdt = float(MAX_ORDER_NOTIONAL_USDT)
        self.require_tp_sl_on_entry = bool(REQUIRE_TP_SL_ON_ENTRY)
        self.min_tp_sl_risk_reward = float(MIN_TP_SL_RISK_REWARD)

    def round_quantity(self, quantity: float) -> float:
        quantity = abs(float(quantity))
        step = self.position_step_size
        if step <= 0:
            return quantity
        # Floor instead of nearest rounding so risk sizing never increases the
        # requested notional above MAX_ORDER_NOTIONAL_USDT.
        rounded = math.floor(quantity / step) * step
        return float(max(0.0, rounded))

    def build_target_quantity(self, signal_position: int, price: float, quote_size_usdt: float | None = None) -> float:
        # Binance Spot paper execution supports long/flat only. SHORT remains a research label
        # and maps to a target flat position instead of an executable short.
        if int(signal_position) <= 0:
            return 0.0
        if price <= 0:
            return 0.0
        quote_size = float(quote_size_usdt) if quote_size_usdt is not None else self.default_quote_size_usdt
        if quote_size <= 0:
            return 0.0
        unsigned_qty = quote_size / float(price)
        return self.round_quantity(unsigned_qty)

    def _trades_today(
        self,
        symbol: str | None = None,
        model_id: str | None = None,
        account_mode: str | None = None,
        now_utc: str | pd.Timestamp | None = None,
    ) -> int:
        now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        today_start = now.floor("D").isoformat()
        conn = sqlite3.connect(DB_FILE)
        try:
            where = ["created_at_utc >= ?", "status = 'FILLED'"]
            params: list[object] = [today_start]
            if symbol:
                where.append("symbol = ?")
                params.append(symbol)
            if model_id:
                where.append("model_id = ?")
                params.append(model_id)
            if account_mode:
                where.append("account_mode = ?")
                params.append(account_mode)

            row = pd.read_sql_query(
                f"""
                SELECT COUNT(*) AS n
                FROM {ORDERS_TABLE}
                WHERE {" AND ".join(where)}
                """,
                conn,
                params=tuple(params),
            )
        finally:
            conn.close()
        return int(row.loc[0, "n"]) if not row.empty else 0

    def real_trading_flags_ok(self) -> bool:
        return (
            self.enable_live_trading
            and self.enable_real_order_execution
            and self.enable_real_binance_account
            and not self.dry_run
        )

    def _log_risk_event(self, symbol: str, approved: bool, reasons: list[str], details: dict) -> None:
        conn = sqlite3.connect(DB_FILE)
        try:
            conn.execute(
                f"""
                INSERT INTO {RISK_EVENTS_TABLE} (
                    model_id, symbol, account_mode, approved, reason, details_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.model_id,
                    symbol,
                    self.account_mode,
                    1 if approved else 0,
                    ";".join(reasons),
                    __import__("json").dumps(details, ensure_ascii=True, sort_keys=True),
                    pd.Timestamp.now(tz="UTC").isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def validate_order(
        self,
        symbol: str,
        price: float,
        delta_quantity: float,
        projected_position_quantity: float,
        portfolio_state,
        side: str | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> dict:
        reasons: list[str] = []
        price = float(price)
        delta_quantity = float(delta_quantity)
        projected_qty = float(projected_position_quantity)

        if self.account_mode == ACCOUNT_MODE_REAL and not self.real_trading_flags_ok():
            reasons.append("real_trading_flags_not_all_enabled")

        if projected_qty < -1e-12:
            reasons.append("spot_short_position_not_allowed")

        opens_or_increases_long = delta_quantity > 0
        protection_details: dict = {}
        if self.require_tp_sl_on_entry and opens_or_increases_long:
            ok, protection_reasons, protection_details = validate_long_protection(
                entry_price=price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                min_risk_reward=self.min_tp_sl_risk_reward,
            )
            if not ok:
                reasons.extend(protection_reasons)

        qty_abs = self.round_quantity(abs(delta_quantity))
        if qty_abs <= 0:
            reasons.append("quantity_rounds_to_zero")

        delta_notional = qty_abs * price
        projected_notional = abs(projected_qty) * price

        current_cash = float(getattr(portfolio_state, "cash", 0.0))
        if delta_quantity > 0 and delta_notional > current_cash:
            reasons.append("insufficient_paper_cash")

        if delta_notional > 0 and delta_notional < self.min_notional_usdt:
            reasons.append("notional_below_minimum")
        if delta_notional > self.max_order_notional_usdt:
            reasons.append("order_notional_above_limit")

        if projected_notional > self.max_position_notional_usdt:
            reasons.append("projected_position_notional_above_limit")

        equity = float(max(getattr(portfolio_state, "equity", 0.0), 1e-9))
        projected_exposure = projected_notional / equity
        if projected_exposure > self.max_exposure_per_asset:
            reasons.append("projected_exposure_above_limit")

        daily_realized = float(getattr(portfolio_state, "daily_realized_pnl", 0.0))
        if daily_realized <= -self.max_daily_loss_usdt:
            reasons.append("daily_loss_limit_reached")

        trades_today = self._trades_today(symbol=symbol, model_id=self.model_id, account_mode=self.account_mode)
        if trades_today >= self.max_new_trades_per_day:
            reasons.append("max_new_trades_per_day_reached")

        approved = len(reasons) == 0
        result = {
            "approved": approved,
            "reasons": reasons,
            "rounded_quantity": qty_abs,
            "delta_notional": delta_notional,
            "projected_notional": projected_notional,
            "projected_exposure": projected_exposure,
            "take_profit_price": float(take_profit_price) if take_profit_price is not None else None,
            "stop_loss_price": float(stop_loss_price) if stop_loss_price is not None else None,
            "protection_required": bool(self.require_tp_sl_on_entry and opens_or_increases_long),
            "protection": protection_details,
        }
        self._log_risk_event(symbol=symbol, approved=approved, reasons=reasons, details=result)
        return result

    def validate_trade_plan(
        self,
        built_trade,
        portfolio_state,
        latest_market_timestamp_utc: str | None = None,
        evaluation_timestamp_utc: str | pd.Timestamp | None = None,
    ) -> dict:
        """Hard gate for accepted proposal trade plans.

        This complements legacy signal-order validation by checking model-owned
        proposal/trade identifiers, model-derived TP/SL, cash reserve, stale data,
        reconciliation state, and max loss before the execution engine can send an
        order. It applies in dry-run, testnet paper, shadow-real and live modes.
        """
        reasons: list[str] = []
        symbol = built_trade.symbol
        price = float(built_trade.entry_reference_price)
        qty = float(built_trade.quantity)
        notional = float(built_trade.approved_notional_usdt)

        if self.account_mode == ACCOUNT_MODE_REAL and not self.real_trading_flags_ok():
            reasons.append("real_trading_flags_not_all_enabled")
        if bool(KILL_SWITCH_ENABLED):
            # The default platform state is safety-first. Operators can later add
            # a dashboard/runtime override table; this hard env default cannot be bypassed.
            pass
        if qty <= 0 or price <= 0:
            reasons.append("invalid_quantity_or_price")
        if built_trade.tp_price <= price:
            reasons.append("tp_not_above_entry_for_long")
        if built_trade.sl_price >= price or built_trade.sl_price <= 0:
            reasons.append("sl_not_below_entry_for_long")
        if built_trade.emergency_sl_price >= built_trade.sl_price:
            reasons.append("emergency_sl_not_wider_than_virtual_sl")
        if float(built_trade.max_loss_usdt) > float(MAX_TRADE_LOSS_USDT) + 1e-9:
            reasons.append("trade_plan_loss_above_hard_limit")
        if notional < self.min_notional_usdt:
            reasons.append("notional_below_minimum")
        if notional > self.max_order_notional_usdt:
            reasons.append("order_notional_above_limit")
        current_cash = float(getattr(portfolio_state, "cash", 0.0))
        if notional > max(0.0, current_cash - float(MIN_CASH_RESERVE_USDT)):
            reasons.append("insufficient_cash_after_reserve")

        # Stale market data blocks new entries. A missing timestamp is treated as
        # stale in live/real mode, but tolerated for local historical smoke tests.
        if latest_market_timestamp_utc:
            age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_market_timestamp_utc)).total_seconds()
            if age > int(STALE_DATA_MAX_SECONDS) and self.account_mode == ACCOUNT_MODE_REAL:
                reasons.append("stale_market_data_blocks_live_entry")
        elif self.account_mode == ACCOUNT_MODE_REAL:
            reasons.append("missing_market_timestamp_blocks_live_entry")

        trades_today = self._trades_today(
            symbol=symbol,
            model_id=built_trade.model_id,
            account_mode=self.account_mode,
            now_utc=evaluation_timestamp_utc,
        )
        if trades_today >= int(MAX_TRADES_PER_DAY_PER_MODEL):
            reasons.append("max_trades_per_day_per_model_reached")

        conn = sqlite3.connect(DB_FILE)
        try:
            statuses = ("APPROVED", "ORDER_PENDING", "ORDER_SENT", "PARTIALLY_FILLED", "OPEN", "REDUCING", "CLOSING")
            qs = ",".join("?" for _ in statuses)
            rows = pd.read_sql_query(
                f"""
                SELECT model_id, symbol, COALESCE(approved_notional_usdt,0) AS notional
                FROM {TRADES_TABLE}
                WHERE account_mode = ? AND status IN ({qs})
                """,
                conn,
                params=(self.account_mode, *statuses),
            )
            if RECONCILIATION_REQUIRED:
                rec = conn.execute(
                    f"SELECT reconciliation_status FROM {SYSTEM_STATUS_TABLE} WHERE component='reconciliation' ORDER BY updated_at_utc DESC LIMIT 1"
                ).fetchone()
                if rec and rec[0] not in {"OK", "NOT_REQUIRED", "UNKNOWN"}:
                    reasons.append("reconciliation_not_healthy")
        finally:
            conn.close()

        total_open = float(rows["notional"].sum()) if not rows.empty else 0.0
        symbol_open = float(rows.loc[rows["symbol"] == symbol, "notional"].sum()) if not rows.empty else 0.0
        model_open = float(rows.loc[rows["model_id"] == built_trade.model_id, "notional"].sum()) if not rows.empty else 0.0
        if total_open + notional > float(MAX_TOTAL_EXPOSURE_USDT):
            reasons.append("max_total_exposure_exceeded")
        if symbol_open + notional > float(MAX_SYMBOL_EXPOSURE_USDT):
            reasons.append("max_symbol_exposure_exceeded")
        if model_open + notional > float(MAX_MODEL_OPEN_EXPOSURE_USDT):
            reasons.append("max_model_open_exposure_exceeded")

        approved = len(reasons) == 0
        result = {
            "approved": approved,
            "reasons": reasons,
            "trade_id": built_trade.trade_id,
            "proposal_id": built_trade.proposal_id,
            "allocation_id": built_trade.allocation_id,
            "prediction_id": built_trade.prediction_id,
            "model_id": built_trade.model_id,
            "symbol": symbol,
            "notional": notional,
            "max_loss_usdt": built_trade.max_loss_usdt,
        }
        self._log_risk_event(symbol=symbol, approved=approved, reasons=reasons, details=result)
        return result
