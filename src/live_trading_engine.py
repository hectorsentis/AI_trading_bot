import argparse
import json

from broker_client import BinanceSpotClient, LiveTradingBlockedError
from config import (
    ALLOW_AUTO_PROMOTE_TO_REAL,
    DRY_RUN,
    ENABLE_LIVE_TRADING,
    ENABLE_REAL_BINANCE_ACCOUNT,
    ENABLE_REAL_ORDER_EXECUTION,
)
from kill_switch import KillSwitch
from model_registry import get_model_by_id, mark_model_real_active
from risk_manager import RiskManager


def real_trading_flags_ok() -> bool:
    return ENABLE_LIVE_TRADING and ENABLE_REAL_ORDER_EXECUTION and ENABLE_REAL_BINANCE_ACCOUNT and not DRY_RUN


class LiveTradingEngine:
    """Real Binance Spot execution gate. No order path bypasses risk and kill switch."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.kill_switch = KillSwitch()
        self.risk_manager = RiskManager(model_id=model_id, account_mode="real")
        self.client = BinanceSpotClient.real_execution_client()

    def assert_can_trade_real(self) -> dict:
        model = get_model_by_id(self.model_id)
        reasons: list[str] = []
        if not model:
            reasons.append("model_not_found")
        elif model.get("status") not in {"real_ready", "real_active"}:
            reasons.append(f"model_status_not_real_ready:{model.get('status')}")
        if not real_trading_flags_ok():
            reasons.append("real_trading_flags_not_all_enabled")
        ks = self.kill_switch.check(self.model_id, "real")
        if not ks["ok"]:
            reasons.extend([f"kill_switch:{r}" for r in ks["reasons"]])
        return {"ok": len(reasons) == 0, "reasons": reasons}

    def activate_if_allowed(self) -> dict:
        gate = self.assert_can_trade_real()
        if not ALLOW_AUTO_PROMOTE_TO_REAL:
            gate["ok"] = False
            gate.setdefault("reasons", []).append("ALLOW_AUTO_PROMOTE_TO_REAL_false")
        if gate["ok"]:
            mark_model_real_active(self.model_id, reason="auto_promote_all_real_flags_enabled")
        return gate

    def place_real_order(self, symbol: str, side: str, quantity: float, price: float, order_type: str = "MARKET", portfolio_state=None) -> dict:
        gate = self.assert_can_trade_real()
        if not gate["ok"]:
            raise LiveTradingBlockedError("Real order blocked: " + ";".join(gate["reasons"]))
        if portfolio_state is None:
            raise LiveTradingBlockedError("portfolio_state required for live risk validation")
        risk = self.risk_manager.validate_order(
            symbol=symbol,
            price=price,
            delta_quantity=quantity if side.upper() == "BUY" else -quantity,
            projected_position_quantity=quantity,
            portfolio_state=portfolio_state,
        )
        if not risk["approved"]:
            raise LiveTradingBlockedError("Risk manager rejected real order: " + ";".join(risk["reasons"]))
        return self.client.place_order(symbol=symbol, side=side, order_type=order_type, quantity=risk["rounded_quantity"])


def main():
    parser = argparse.ArgumentParser(description="Inspect real trading safety gates; does not place orders.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--activate-if-allowed", action="store_true")
    args = parser.parse_args()
    engine = LiveTradingEngine(args.model_id)
    result = engine.activate_if_allowed() if args.activate_if_allowed else engine.assert_can_trade_real()
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
