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
    REQUIRE_TP_SL_ON_ENTRY,
    MIN_TP_SL_RISK_REWARD,
    TP_MULTIPLIER,
    SL_MULTIPLIER,
)
from broker_client import BinanceCredentialsError, BinanceSpotClient, LiveTradingBlockedError
from db_utils import init_research_tables
from trade_protection import build_long_protection, validate_long_protection
from trade_builder import BuiltTrade
from ledger import create_trade_record, mark_trade_failed, mark_trade_open


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
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        risk_reward: float | None = None,
        protection_required: bool = True,
        protection_status: str | None = None,
        linked_exit_order_id: str | None = None,
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
                    research_signal_label, take_profit_price, stop_loss_price, risk_reward, protection_required,
                    protection_status, linked_exit_order_id, decision_json, dry_run, created_at_utc, updated_at_utc,
                    filled_at_utc, raw_exchange_response_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    float(take_profit_price) if take_profit_price is not None else None,
                    float(stop_loss_price) if stop_loss_price is not None else None,
                    float(risk_reward) if risk_reward is not None else None,
                    1 if protection_required else 0,
                    protection_status,
                    linked_exit_order_id,
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

    def _protection_from_payload(self, signal_payload: dict, entry_price: float) -> dict:
        tp = signal_payload.get("take_profit_price")
        sl = signal_payload.get("stop_loss_price")
        rr = signal_payload.get("risk_reward")
        atr = signal_payload.get("atr_14")

        if (tp is None or sl is None) and atr is not None:
            protection = build_long_protection(
                entry_price=entry_price,
                atr=float(atr),
                tp_multiplier=float(signal_payload.get("tp_multiplier", TP_MULTIPLIER)),
                sl_multiplier=float(signal_payload.get("sl_multiplier", SL_MULTIPLIER)),
            )
            tp = protection.take_profit_price
            sl = protection.stop_loss_price
            rr = protection.risk_reward

        if rr is None and tp is not None and sl is not None:
            _, _, details = validate_long_protection(
                entry_price=entry_price,
                take_profit_price=float(tp),
                stop_loss_price=float(sl),
                min_risk_reward=MIN_TP_SL_RISK_REWARD,
            )
            rr = details.get("risk_reward")
        return {
            "take_profit_price": float(tp) if tp is not None else None,
            "stop_loss_price": float(sl) if sl is not None else None,
            "risk_reward": float(rr) if rr is not None else None,
        }

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
        protection = self._protection_from_payload(signal_payload, market_price) if delta_qty > 0 else {
            "take_profit_price": signal_payload.get("take_profit_price"),
            "stop_loss_price": signal_payload.get("stop_loss_price"),
            "risk_reward": signal_payload.get("risk_reward"),
        }
        risk = self.risk_manager.validate_order(
            symbol=symbol,
            price=market_price,
            delta_quantity=delta_qty,
            projected_position_quantity=current_qty + delta_qty,
            portfolio_state=state,
            side=side,
            take_profit_price=protection.get("take_profit_price"),
            stop_loss_price=protection.get("stop_loss_price"),
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
                take_profit_price=protection.get("take_profit_price"),
                stop_loss_price=protection.get("stop_loss_price"),
                risk_reward=protection.get("risk_reward"),
                protection_required=bool(REQUIRE_TP_SL_ON_ENTRY and delta_qty > 0),
                protection_status="REJECTED_BEFORE_ENTRY" if delta_qty > 0 else None,
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
                take_profit_price=protection.get("take_profit_price"),
                stop_loss_price=protection.get("stop_loss_price"),
                risk_reward=protection.get("risk_reward"),
                protection_required=bool(REQUIRE_TP_SL_ON_ENTRY and delta_qty > 0),
                protection_status="SKIPPED_BEFORE_ENTRY" if delta_qty > 0 else None,
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
        linked_exit_order_id = None
        fill_price = self._simulated_fill_price(side=side, market_price=market_price)
        reason = "dry_run_fill"
        protection_status = "NOT_REQUIRED"

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

        if side == "BUY" and REQUIRE_TP_SL_ON_ENTRY:
            # Re-anchor TP/SL to actual fill price so slippage cannot leave a
            # filled entry with stale or invalid protection.
            if signal_payload.get("atr_14") is not None:
                fill_protection = self._protection_from_payload(signal_payload, fill_price)
                protection.update(fill_protection)
            ok, protection_reasons, protection_details = validate_long_protection(
                entry_price=fill_price,
                take_profit_price=protection.get("take_profit_price"),
                stop_loss_price=protection.get("stop_loss_price"),
                min_risk_reward=MIN_TP_SL_RISK_REWARD,
            )
            if not ok:
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
                    reason="invalid_tp_sl_after_fill_anchor:" + ";".join(protection_reasons),
                    signal_position=research_position,
                    research_signal_label=research_label,
                    decision_payload=signal_payload,
                    raw_exchange_response=raw_exchange_response,
                    take_profit_price=protection.get("take_profit_price"),
                    stop_loss_price=protection.get("stop_loss_price"),
                    risk_reward=protection.get("risk_reward"),
                    protection_required=True,
                    protection_status="REJECTED_INVALID_BRACKET",
                )
                return {
                    "order_id": order_id,
                    "status": "REJECTED",
                    "reason": "invalid_tp_sl_after_fill_anchor:" + ";".join(protection_reasons),
                    "protection": protection_details,
                }
            protection["risk_reward"] = protection_details.get("risk_reward")

            if self.account_mode == ACCOUNT_MODE_TESTNET_PAPER and self.broker_client is not None:
                try:
                    oco_response = self.broker_client.place_oco_sell(
                        symbol=symbol,
                        quantity=qty,
                        take_profit_price=float(protection["take_profit_price"]),
                        stop_loss_price=float(protection["stop_loss_price"]),
                        list_client_order_id=f"bracket_{uuid.uuid4().hex[:20]}",
                    )
                    raw_exchange_response = {"entry_order": raw_exchange_response, "protective_oco": oco_response}
                    linked_exit_order_id = str(oco_response.get("orderListId") or oco_response.get("listClientOrderId") or "")
                    protection_status = "BINANCE_OCO_ACTIVE"
                except Exception as exc:
                    emergency_flatten_response = None
                    try:
                        emergency_flatten_response = self.broker_client.place_order(
                            symbol=symbol,
                            side="SELL",
                            order_type=DEFAULT_ORDER_TYPE,
                            quantity=qty,
                        )
                    except Exception as flatten_exc:
                        emergency_flatten_response = {"error": str(flatten_exc)}
                    raw_exchange_response = {
                        "entry_order": raw_exchange_response,
                        "protective_oco_error": str(exc),
                        "emergency_flatten_order": emergency_flatten_response,
                    }
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
                        status="REJECTED",
                        reason=f"protective_oco_failed_after_entry:{exc}",
                        signal_position=research_position,
                        research_signal_label=research_label,
                        decision_payload=signal_payload,
                        exchange_order_id=exchange_order_id,
                        raw_exchange_response=raw_exchange_response,
                        take_profit_price=protection.get("take_profit_price"),
                        stop_loss_price=protection.get("stop_loss_price"),
                        risk_reward=protection.get("risk_reward"),
                        protection_required=True,
                        protection_status="BRACKET_FAILED_REVIEW_REQUIRED",
                    )
                    return {
                        "order_id": order_id,
                        "status": "REJECTED",
                        "reason": f"protective_oco_failed_after_entry:{exc}",
                        "raw_exchange_response": raw_exchange_response,
                    }
            else:
                protection_status = "LOCAL_BRACKET_RECORDED"

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
            take_profit_price=protection.get("take_profit_price"),
            stop_loss_price=protection.get("stop_loss_price"),
            risk_reward=protection.get("risk_reward"),
            protection_required=bool(REQUIRE_TP_SL_ON_ENTRY and side == "BUY"),
            protection_status=protection_status,
            linked_exit_order_id=linked_exit_order_id,
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
            "linked_exit_order_id": linked_exit_order_id,
            "take_profit_price": protection.get("take_profit_price"),
            "stop_loss_price": protection.get("stop_loss_price"),
            "risk_reward": protection.get("risk_reward"),
            "protection_status": protection_status,
            "raw_exchange_response": raw_exchange_response,
            "risk": risk,
            "fill": fill,
        }


    def execute_built_trade(self, built_trade: BuiltTrade, latest_market_timestamp_utc: str | None = None) -> dict:
        """Central execution path for model-owned proposals.

        The caller must have already persisted prediction, proposal and allocation.
        This method reruns risk gates, logs order/fill rows with full attribution,
        and updates the trade ledger. Models never call broker endpoints directly.
        """
        create_trade_record(built_trade, account_mode=self.account_mode, status="APPROVED")
        state = self.portfolio_manager.get_state(price_by_symbol={built_trade.symbol: built_trade.entry_reference_price})
        risk = self.risk_manager.validate_trade_plan(
            built_trade,
            portfolio_state=state,
            latest_market_timestamp_utc=latest_market_timestamp_utc,
        )
        if not risk["approved"]:
            reason = ";".join(risk["reasons"])
            mark_trade_failed(built_trade.trade_id, reason, raw={"risk": risk})
            self._insert_trade_order(
                built_trade=built_trade,
                side="BUY",
                status="REJECTED",
                reason=reason,
                fill_price=None,
                raw_exchange_response={"risk": risk},
            )
            return {"trade_id": built_trade.trade_id, "status": "REJECTED", "reason": reason, "risk": risk}

        fill_price = self._simulated_fill_price(side="BUY", market_price=built_trade.entry_reference_price)
        raw_exchange_response = {}
        exchange_order_id = None
        reason = "dry_run_trade_plan_fill"
        if self.account_mode == ACCOUNT_MODE_TESTNET_PAPER and self.broker_client is not None:
            try:
                raw_exchange_response = self.broker_client.place_order(
                    symbol=built_trade.symbol,
                    side="BUY",
                    order_type=DEFAULT_ORDER_TYPE,
                    quantity=built_trade.quantity,
                    newClientOrderId=built_trade.trade_id[:30],
                )
                exchange_order_id = str(raw_exchange_response.get("orderId")) if raw_exchange_response.get("orderId") is not None else None
                fills = raw_exchange_response.get("fills") or []
                if fills:
                    fill_price = float(fills[0].get("price", fill_price))
                reason = "binance_testnet_trade_plan_order"
            except Exception as exc:
                raw_exchange_response = {"error": str(exc)}
                mark_trade_failed(built_trade.trade_id, f"testnet_order_failed:{exc}", raw=raw_exchange_response)
                self._insert_trade_order(
                    built_trade=built_trade,
                    side="BUY",
                    status="REJECTED",
                    reason=f"testnet_order_failed:{exc}",
                    fill_price=None,
                    raw_exchange_response=raw_exchange_response,
                )
                return {"trade_id": built_trade.trade_id, "status": "REJECTED", "reason": f"testnet_order_failed:{exc}"}

        fill = self.portfolio_manager.apply_fill(
            symbol=built_trade.symbol,
            side="BUY",
            quantity=built_trade.quantity,
            fill_price=fill_price,
            fee_rate=PAPER_FEE_RATE,
        )
        order_id = self._insert_trade_order(
            built_trade=built_trade,
            side="BUY",
            status="FILLED",
            reason=reason,
            fill_price=fill_price,
            exchange_order_id=exchange_order_id,
            raw_exchange_response=raw_exchange_response,
        )
        fill_id = self._insert_trade_fill(
            built_trade=built_trade,
            order_id=order_id,
            price=fill_price,
            commission=float(fill.get("fee", 0.0)),
            raw=raw_exchange_response,
        )
        slippage_usdt = abs(fill_price - built_trade.entry_reference_price) * built_trade.quantity
        mark_trade_open(
            trade_id=built_trade.trade_id,
            avg_entry_price=fill_price,
            qty=built_trade.quantity,
            fees_usdt=float(fill.get("fee", 0.0)),
            slippage_usdt=slippage_usdt,
            raw_update={"order_id": order_id, "fill_id": fill_id, "reason": reason},
        )
        return {
            "trade_id": built_trade.trade_id,
            "order_id": order_id,
            "fill_id": fill_id,
            "status": "FILLED",
            "reason": reason,
            "fill_price": fill_price,
            "tp_price": built_trade.tp_price,
            "sl_price": built_trade.sl_price,
            "emergency_sl_price": built_trade.emergency_sl_price,
        }

    def _insert_trade_order(
        self,
        *,
        built_trade: BuiltTrade,
        side: str,
        status: str,
        reason: str,
        fill_price: float | None,
        exchange_order_id: str | None = None,
        raw_exchange_response: dict | None = None,
    ) -> str:
        order_id = self._order_id()
        requested_price = float(built_trade.entry_reference_price)
        notional = float(built_trade.quantity) * float(fill_price if fill_price is not None else requested_price)
        now = pd.Timestamp.now(tz="UTC").isoformat()
        conn = self._connect()
        try:
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id, built_trade.trade_id, exchange_order_id, built_trade.model_id, built_trade.prediction_id,
                    built_trade.proposal_id, built_trade.allocation_id, built_trade.trade_id, built_trade.symbol, built_trade.timeframe,
                    now, built_trade.timestamp_utc if hasattr(built_trade, "timestamp_utc") else built_trade.valid_until_utc,
                    self.account_mode, side, side, DEFAULT_ORDER_TYPE, DEFAULT_ORDER_TYPE, float(built_trade.quantity),
                    requested_price, requested_price, float(fill_price) if fill_price is not None else None,
                    float(fill_price) if fill_price is not None else None, float(notional), status, reason,
                    float(built_trade.tp_price), float(built_trade.sl_price), 1, "VIRTUAL_TP_SL_RECORDED",
                    json.dumps(built_trade.raw_json, ensure_ascii=True, sort_keys=True), 1 if self.dry_run else 0,
                    now, now, now if fill_price is not None and status == "FILLED" else None,
                    json.dumps(raw_exchange_response or {}, ensure_ascii=True, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return order_id

    def _insert_trade_fill(
        self,
        *,
        built_trade: BuiltTrade,
        order_id: str,
        price: float,
        commission: float = 0.0,
        raw: dict | None = None,
    ) -> str:
        fill_id = f"fill_{uuid.uuid4().hex}"
        now = pd.Timestamp.now(tz="UTC").isoformat()
        slippage_usdt = abs(float(price) - float(built_trade.entry_reference_price)) * float(built_trade.quantity)
        conn = self._connect()
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {FILLS_TABLE} (
                    fill_id, order_id, model_id, prediction_id, proposal_id, allocation_id, trade_id,
                    symbol, timeframe, account_mode, quantity, price, commission, fee_usdt,
                    commission_asset, slippage_usdt, timestamp_utc, raw_exchange_response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id, order_id, built_trade.model_id, built_trade.prediction_id, built_trade.proposal_id,
                    built_trade.allocation_id, built_trade.trade_id, built_trade.symbol, built_trade.timeframe, self.account_mode,
                    float(built_trade.quantity), float(price), float(commission), float(commission), "USDT",
                    float(slippage_usdt), now, json.dumps(raw or {}, ensure_ascii=True, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return fill_id
