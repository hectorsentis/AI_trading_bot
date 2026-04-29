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
)


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

    def _trades_today(self, symbol: str | None = None, model_id: str | None = None, account_mode: str | None = None) -> int:
        today_start = pd.Timestamp.now(tz="UTC").floor("D").isoformat()
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
    ) -> dict:
        reasons: list[str] = []
        price = float(price)
        delta_quantity = float(delta_quantity)
        projected_qty = float(projected_position_quantity)

        if self.account_mode == ACCOUNT_MODE_REAL and not self.real_trading_flags_ok():
            reasons.append("real_trading_flags_not_all_enabled")

        if projected_qty < -1e-12:
            reasons.append("spot_short_position_not_allowed")

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
        }
        self._log_risk_event(symbol=symbol, approved=approved, reasons=reasons, details=result)
        return result
