import sqlite3
from dataclasses import dataclass

import pandas as pd

from config import (
    DB_FILE,
    POSITIONS_TABLE,
    PORTFOLIO_SNAPSHOTS_TABLE,
    TIMEFRAME,
    PAPER_INITIAL_CASH_USDT,
    ACCOUNT_MODE_LOCAL_PAPER,
)
from db_utils import init_research_tables, save_portfolio_snapshot


@dataclass
class PortfolioState:
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    exposure_by_symbol: dict[str, float]
    positions: dict[str, dict]
    daily_realized_pnl: float


class PortfolioManager:
    def __init__(
        self,
        timeframe: str = TIMEFRAME,
        dry_run: bool = True,
        initial_cash: float = PAPER_INITIAL_CASH_USDT,
        model_id: str = "legacy",
        account_mode: str = ACCOUNT_MODE_LOCAL_PAPER,
    ):
        init_research_tables()
        self.timeframe = timeframe
        self.dry_run = dry_run
        self.initial_cash = float(initial_cash)
        self.model_id = model_id
        self.account_mode = account_mode

        latest = self._load_latest_snapshot()
        if latest is None:
            self.cash = float(initial_cash)
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0
            self.equity = float(initial_cash)
        else:
            self.cash = float(latest["cash"])
            self.realized_pnl = float(latest["realized_pnl"])
            self.unrealized_pnl = float(latest["unrealized_pnl"])
            self.equity = float(latest["equity"])

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(DB_FILE)

    def _dry_run_int(self) -> int:
        return 1 if self.dry_run else 0

    def _load_latest_snapshot(self) -> dict | None:
        conn = self._connect()
        try:
            df = pd.read_sql_query(
                f"""
                SELECT datetime_utc, cash, equity, realized_pnl, unrealized_pnl
                FROM {PORTFOLIO_SNAPSHOTS_TABLE}
                WHERE dry_run = ? AND model_id = ? AND account_mode = ?
                ORDER BY snapshot_id DESC
                LIMIT 1
                """,
                conn,
                params=(self._dry_run_int(), self.model_id, self.account_mode),
            )
        finally:
            conn.close()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def _load_positions_df(self) -> pd.DataFrame:
        conn = self._connect()
        try:
            df = pd.read_sql_query(
                f"""
                SELECT model_id, symbol, timeframe, account_mode, quantity, avg_price, realized_pnl, unrealized_pnl, updated_at_utc, dry_run
                FROM {POSITIONS_TABLE}
                WHERE model_id = ? AND timeframe = ? AND account_mode = ?
                ORDER BY symbol
                """,
                conn,
                params=(self.model_id, self.timeframe, self.account_mode),
            )
        finally:
            conn.close()
        if df.empty:
            return df
        for col in ["quantity", "avg_price", "realized_pnl", "unrealized_pnl"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    def _upsert_position(self, symbol: str, quantity: float, avg_price: float, realized_pnl: float, unrealized_pnl: float):
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {POSITIONS_TABLE} (
                    model_id, symbol, timeframe, account_mode, quantity, avg_price, realized_pnl, unrealized_pnl, updated_at_utc, dry_run
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.model_id,
                    symbol,
                    self.timeframe,
                    self.account_mode,
                    float(quantity),
                    float(avg_price),
                    float(realized_pnl),
                    float(unrealized_pnl),
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    self._dry_run_int(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _daily_realized_pnl(self) -> float:
        today_start = pd.Timestamp.now(tz="UTC").floor("D")
        conn = self._connect()
        try:
            df = pd.read_sql_query(
                f"""
                SELECT realized_pnl
                FROM {PORTFOLIO_SNAPSHOTS_TABLE}
                WHERE dry_run = ? AND model_id = ? AND account_mode = ? AND datetime_utc >= ?
                ORDER BY snapshot_id ASC
                LIMIT 1
                """,
                conn,
                params=(self._dry_run_int(), self.model_id, self.account_mode, today_start.isoformat()),
            )
        finally:
            conn.close()
        if df.empty:
            return 0.0
        realized_at_day_start = float(df.loc[0, "realized_pnl"])
        return float(self.realized_pnl - realized_at_day_start)

    def get_state(self, price_by_symbol: dict[str, float] | None = None) -> PortfolioState:
        positions_df = self._load_positions_df()
        positions: dict[str, dict] = {}
        exposure_notional = {}
        total_unrealized = 0.0

        for _, row in positions_df.iterrows():
            symbol = str(row["symbol"])
            qty = float(row["quantity"])
            avg = float(row["avg_price"])
            realized = float(row["realized_pnl"])
            current_price = float(price_by_symbol[symbol]) if price_by_symbol and symbol in price_by_symbol else avg
            unrealized = qty * (current_price - avg)
            notional = abs(qty * current_price)

            total_unrealized += unrealized
            exposure_notional[symbol] = notional
            positions[symbol] = {
                "quantity": qty,
                "avg_price": avg,
                "current_price": current_price,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "notional_abs": notional,
            }

        equity = float(self.cash + self.realized_pnl + total_unrealized)
        if equity <= 0:
            equity = 1e-9
        exposure_by_symbol = {symbol: float(notional / equity) for symbol, notional in exposure_notional.items()}

        return PortfolioState(
            cash=float(self.cash),
            equity=equity,
            realized_pnl=float(self.realized_pnl),
            unrealized_pnl=float(total_unrealized),
            exposure_by_symbol=exposure_by_symbol,
            positions=positions,
            daily_realized_pnl=self._daily_realized_pnl(),
        )

    def apply_fill(self, symbol: str, side: str, quantity: float, fill_price: float, fee_rate: float) -> dict:
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if fill_price <= 0:
            raise ValueError("fill_price must be > 0")

        side = side.upper().strip()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        signed_qty = float(quantity) if side == "BUY" else -float(quantity)
        positions_df = self._load_positions_df()
        existing = positions_df[positions_df["symbol"] == symbol]

        prev_qty = float(existing.iloc[0]["quantity"]) if not existing.empty else 0.0
        prev_avg = float(existing.iloc[0]["avg_price"]) if not existing.empty else 0.0
        prev_realized = float(existing.iloc[0]["realized_pnl"]) if not existing.empty else 0.0

        realized_delta = 0.0
        new_qty = prev_qty + signed_qty

        if prev_qty == 0 or (prev_qty > 0 and signed_qty > 0) or (prev_qty < 0 and signed_qty < 0):
            new_avg = (
                (abs(prev_qty) * prev_avg + abs(signed_qty) * float(fill_price)) / abs(new_qty)
                if abs(new_qty) > 0
                else 0.0
            )
        else:
            closing_qty = min(abs(prev_qty), abs(signed_qty))
            if prev_qty > 0 and signed_qty < 0:
                realized_delta += closing_qty * (float(fill_price) - prev_avg)
            elif prev_qty < 0 and signed_qty > 0:
                realized_delta += closing_qty * (prev_avg - float(fill_price))

            if new_qty == 0:
                new_avg = 0.0
            elif (prev_qty > 0 and new_qty > 0) or (prev_qty < 0 and new_qty < 0):
                new_avg = prev_avg
            else:
                new_avg = float(fill_price)

        fee = abs(float(quantity) * float(fill_price)) * float(fee_rate)
        realized_delta -= fee

        self.cash -= signed_qty * float(fill_price)
        self.realized_pnl += realized_delta
        updated_realized = prev_realized + realized_delta

        self._upsert_position(
            symbol=symbol,
            quantity=new_qty,
            avg_price=new_avg,
            realized_pnl=updated_realized,
            unrealized_pnl=0.0,
        )

        return {
            "symbol": symbol,
            "side": side,
            "quantity": float(quantity),
            "fill_price": float(fill_price),
            "fee": fee,
            "realized_delta": realized_delta,
            "new_quantity": new_qty,
            "new_avg_price": new_avg,
        }

    def snapshot(self, price_by_symbol: dict[str, float] | None = None) -> PortfolioState:
        state = self.get_state(price_by_symbol=price_by_symbol)
        self.equity = state.equity
        self.unrealized_pnl = state.unrealized_pnl
        save_portfolio_snapshot(
            cash=state.cash,
            equity=state.equity,
            realized_pnl=state.realized_pnl,
            unrealized_pnl=state.unrealized_pnl,
            exposure_by_symbol=state.exposure_by_symbol,
            dry_run=self.dry_run,
            model_id=self.model_id,
            account_mode=self.account_mode,
        )
        return state
