import json
import sqlite3
import uuid

import pandas as pd

from config import (
    DB_FILE,
    ORDERS_TABLE,
    FILLS_TABLE,
    DEFAULT_ORDER_TYPE,
    DRY_RUN,
    PAPER_SLIPPAGE_BPS,
    PAPER_FEE_RATE,
    ACCOUNT_MODE_LOCAL_PAPER,
    ACCOUNT_MODE_TESTNET_PAPER,
)
from broker_client import BinanceCredentialsError, BinanceSpotClient, LiveTradingBlockedError
from db_utils import init_research_tables


class ExecutionEngine:
    def __init__(self, portfolio_manager, risk_manager, account_mode: str = ACCOUNT_MODE_LOCAL_PAPER, broker_client=None):
        init_research_tables()
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.dry_run = DRY_RUN
        self.slippage_bps = float(PAPER_SLIPPAGE_BPS)
        self.account_mode = account_mode
        self.broker_client = broker_client

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(DB_FILE)

    def _order_id(self) -> str:
        return f"dry_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:10]}"

    def _simulated_fill_price(self, side: str, market_price: float) -> float:
        slip = self.slippage_bps / 10_000.0
        if side == "BUY":
            return float(market_price) * (1.0 + slip)
        return float(market_price) * (1.0 - slip)

    def _insert_order(
        self,
        order_id: str,
        model_id: str,
        symbol: str,
        timeframe: str,
        signal_datetime_utc: str | None,
        side: str,
        executable_action: str,
        quantity: float,
        requested_price: float,
        fill_price: float | None,
        status: str,
        reason: str | None,
        signal_position: int,
        research_signal_label: str,
        decision_payload: dict | None = None,
        exchange_order_id: str | None = None,
        raw_exchange_response: dict | None = None,
    ) -> None:
        notional = abs(quantity * (fill_price if fill_price is not None else requested_price))
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {ORDERS_TABLE} (
                    order_id, model_id, symbol, timeframe, signal_datetime_utc,
                    exchange_order_id, account_mode, side, executable_action, order_type, type, quantity,
                    requested_price, price_requested, fill_price, price_filled, notional, status, reason, signal_position,
                    research_signal_label, decision_json, dry_run, created_at_utc, updated_at_utc, filled_at_utc,
                    raw_exchange_response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    model_id,
                    symbol,
                    timeframe,
                    signal_datetime_utc,
                    exchange_order_id,
                    self.account_mode,
                    side,
                    executable_action,
                    DEFAULT_ORDER_TYPE,
                    DEFAULT_ORDER_TYPE,
                    float(quantity),
                    float(requested_price),
                    float(requested_price),
                    float(fill_price) if fill_price is not None else None,
                    float(fill_price) if fill_price is not None else None,
                    float(notional),
                    status,
                    reason,
                    int(signal_position),
                    research_signal_label,
                    json.dumps(decision_payload or {}, ensure_ascii=True, sort_keys=True),
                    1 if self.dry_run else 0,
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    pd.Timestamp.now(tz="UTC").isoformat() if fill_price is not None and status == "FILLED" else None,
                    json.dumps(raw_exchange_response or {}, ensure_ascii=True, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _already_processed_signal(
        self,
        model_id: str,
        symbol: str,
        timeframe: str,
        signal_datetime_utc: str | None,
    ) -> bool:
        if not signal_datetime_utc:
            return False
        conn = self._connect()
        try:
            row = pd.read_sql_query(
                f"""
                SELECT COUNT(*) AS n
                FROM {ORDERS_TABLE}
                WHERE model_id = ?
                  AND symbol = ?
                  AND timeframe = ?
                  AND signal_datetime_utc = ?
                  AND dry_run = ?
                  AND account_mode = ?
                  AND status IN ('FILLED', 'SKIPPED', 'REJECTED')
                """,
                conn,
                params=(model_id, symbol, timeframe, signal_datetime_utc, 1 if self.dry_run else 0, self.account_mode),
            )
        finally:
            conn.close()
        return int(row.loc[0, "n"]) > 0 if not row.empty else False

    def _insert_fill(self, order_id: str, model_id: str, symbol: str, timeframe: str, quantity: float, price: float, commission: float = 0.0, commission_asset: str | None = "USDT", exchange_trade_id: str | None = None, raw: dict | None = None) -> str:
        fill_id = f"fill_{uuid.uuid4().hex}"
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {FILLS_TABLE} (
                    fill_id, order_id, exchange_trade_id, model_id, symbol, timeframe, account_mode,
                    quantity, price, commission, commission_asset, timestamp_utc, raw_exchange_response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    order_id,
                    exchange_trade_id,
                    model_id,
                    symbol,
                    timeframe,
                    self.account_mode,
                    float(quantity),
                    float(price),
                    float(commission),
                    commission_asset,
                    pd.Timestamp.now(tz="UTC").isoformat(),
                    json.dumps(raw or {}, ensure_ascii=True, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return fill_id

    def execute_signal(
        self,
        model_id: str,
        symbol: str,
        timeframe: str,
        signal_payload: dict,
        market_price: float,
    ) -> dict:
        signal_datetime_utc = signal_payload.get("datetime_utc")
        if hasattr(signal_datetime_utc, "isoformat"):
            signal_datetime_utc = signal_datetime_utc.isoformat()
        elif signal_datetime_utc is not None:
            signal_datetime_utc = str(signal_datetime_utc)

        research_position = int(signal_payload["final_signal_position"])
        research_label = str(signal_payload.get("final_signal_label", {1: "LONG", 0: "FLAT", -1: "SHORT"}[research_position]))

        if self._already_processed_signal(model_id, symbol, timeframe, signal_datetime_utc):
            return {
                "status": "SKIPPED",
                "reason": "signal_datetime_already_processed",
                "signal_datetime_utc": signal_datetime_utc,
            }

        state = self.portfolio_manager.get_state(price_by_symbol={symbol: market_price})
        current_qty = float(state.positions.get(symbol, {}).get("quantity", 0.0))

        # Spot executable target is long/flat only. Research SHORT maps to target flat.
        target_qty = self.risk_manager.build_target_quantity(signal_position=research_position, price=market_price)
        delta_qty = float(target_qty - current_qty)
        executable_action = "BUY" if delta_qty > 0 else ("SELL" if delta_qty < 0 else "HOLD")

        if abs(delta_qty) <= max(self.risk_manager.position_step_size / 2.0, 1e-12):
            order_id = self._order_id()
            self._insert_order(
                order_id=order_id,
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                signal_datetime_utc=signal_datetime_utc,
                side="HOLD",
                executable_action=executable_action,
                quantity=0.0,
                requested_price=market_price,
                fill_price=None,
                status="SKIPPED",
                reason="already_at_target_position",
                signal_position=research_position,
                research_signal_label=research_label,
                decision_payload=signal_payload,
            )
            return {
                "order_id": order_id,
                "status": "SKIPPED",
                "reason": "already_at_target_position",
                "target_qty": target_qty,
                "current_qty": current_qty,
                "executable_action": executable_action,
            }

        side = executable_action
        risk = self.risk_manager.validate_order(
            symbol=symbol,
            price=market_price,
            delta_quantity=delta_qty,
            projected_position_quantity=current_qty + delta_qty,
            portfolio_state=state,
        )

        order_id = self._order_id()
        if not risk["approved"]:
            reason = ";".join(risk["reasons"])
            self._insert_order(
                order_id=order_id,
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                signal_datetime_utc=signal_datetime_utc,
                side=side,
                executable_action=executable_action,
                quantity=abs(delta_qty),
                requested_price=market_price,
                fill_price=None,
                status="REJECTED",
                reason=reason,
                signal_position=research_position,
                research_signal_label=research_label,
                decision_payload=signal_payload,
            )
            return {
                "order_id": order_id,
                "status": "REJECTED",
                "reason": reason,
                "risk": risk,
                "target_qty": target_qty,
                "current_qty": current_qty,
                "executable_action": executable_action,
            }

        qty = float(risk["rounded_quantity"])
        if qty <= 0:
            self._insert_order(
                order_id=order_id,
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                signal_datetime_utc=signal_datetime_utc,
                side=side,
                executable_action=executable_action,
                quantity=0.0,
                requested_price=market_price,
                fill_price=None,
                status="SKIPPED",
                reason="rounded_quantity_zero",
                signal_position=research_position,
                research_signal_label=research_label,
                decision_payload=signal_payload,
            )
            return {
                "order_id": order_id,
                "status": "SKIPPED",
                "reason": "rounded_quantity_zero",
                "risk": risk,
                "target_qty": target_qty,
                "current_qty": current_qty,
                "executable_action": executable_action,
            }

        raw_exchange_response = {}
        exchange_order_id = None
        fill_price = self._simulated_fill_price(side=side, market_price=market_price)
        reason = "dry_run_fill"

        if self.account_mode == ACCOUNT_MODE_TESTNET_PAPER and self.broker_client is not None:
            try:
                raw_exchange_response = self.broker_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=DEFAULT_ORDER_TYPE,
                    quantity=qty,
                )
                exchange_order_id = str(raw_exchange_response.get("orderId")) if raw_exchange_response.get("orderId") is not None else None
                fills = raw_exchange_response.get("fills") or []
                if fills:
                    fill_price = float(fills[0].get("price", fill_price))
                elif raw_exchange_response.get("price"):
                    fill_price = float(raw_exchange_response.get("price"))
                reason = "binance_testnet_order"
            except (BinanceCredentialsError, LiveTradingBlockedError, Exception) as exc:
                raw_exchange_response = {"error": str(exc)}
                if self.account_mode == ACCOUNT_MODE_TESTNET_PAPER:
                    # Never fall through to real; caller may choose local_paper fallback by using a local engine.
                    self._insert_order(
                        order_id=order_id,
                        model_id=model_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        signal_datetime_utc=signal_datetime_utc,
                        side=side,
                        executable_action=executable_action,
                        quantity=qty,
                        requested_price=market_price,
                        fill_price=None,
                        status="REJECTED",
                        reason=f"testnet_order_failed:{exc}",
                        signal_position=research_position,
                        research_signal_label=research_label,
                        decision_payload=signal_payload,
                        raw_exchange_response=raw_exchange_response,
                    )
                    return {
                        "order_id": order_id,
                        "status": "REJECTED",
                        "reason": f"testnet_order_failed:{exc}",
                        "raw_exchange_response": raw_exchange_response,
                    }

        fill = self.portfolio_manager.apply_fill(
            symbol=symbol,
            side=side,
            quantity=qty,
            fill_price=fill_price,
            fee_rate=PAPER_FEE_RATE,
        )
        self._insert_fill(
            order_id=order_id,
            model_id=model_id,
            symbol=symbol,
            timeframe=timeframe,
            quantity=qty,
            price=fill_price,
            commission=float(fill.get("fee", 0.0)),
            raw=raw_exchange_response,
        )

        self._insert_order(
            order_id=order_id,
            model_id=model_id,
            symbol=symbol,
            timeframe=timeframe,
            signal_datetime_utc=signal_datetime_utc,
            side=side,
            executable_action=executable_action,
            quantity=qty,
            requested_price=market_price,
            fill_price=fill_price,
            status="FILLED",
            reason=reason,
            signal_position=research_position,
            research_signal_label=research_label,
            decision_payload=signal_payload,
            exchange_order_id=exchange_order_id,
            raw_exchange_response=raw_exchange_response,
        )

        return {
            "order_id": order_id,
            "status": "FILLED",
            "target_qty": target_qty,
            "current_qty": current_qty,
            "delta_qty_requested": delta_qty,
            "delta_qty_filled": qty if side == "BUY" else -qty,
            "executable_action": executable_action,
            "fill_price": fill_price,
            "reason": reason,
            "exchange_order_id": exchange_order_id,
            "raw_exchange_response": raw_exchange_response,
            "risk": risk,
            "fill": fill,
        }

