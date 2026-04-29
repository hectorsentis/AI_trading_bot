import argparse
import json
import tempfile
from pathlib import Path

from config import DRY_RUN, ENABLE_LIVE_TRADING, ENABLE_REAL_BINANCE_ACCOUNT, ENABLE_REAL_ORDER_EXECUTION
from db_utils import assert_required_schema, init_research_tables
from kill_switch import assert_real_trading_not_default
from model_registry import activate_model_for_paper, register_model, get_model_by_id, mark_model_paper_rejected


def run_checks() -> dict:
    init_research_tables()
    safety = assert_real_trading_not_default()
    schema = assert_required_schema()
    pool = {"ok": True, "details": []}
    model_id = "check_model_do_not_trade"
    existing = get_model_by_id(model_id)
    if not existing:
        try:
            register_model(
                model_id=model_id,
                symbol_scope="BTCUSDT",
                timeframe="1h",
                train_start=None,
                train_end=None,
                test_start=None,
                test_end=None,
                feature_version="check",
                label_version="check",
                model_path="models/check_model_do_not_trade.joblib",
                metrics={},
                params={},
                status="backtest_accepted",
                acceptance_status="accepted",
                rejection_reasons=[],
                evaluation_scope="check",
            )
        except Exception as exc:
            pool = {"ok": False, "details": [str(exc)]}
    try:
        activate_model_for_paper(model_id, account_mode="local_paper")
        mark_model_paper_rejected(model_id, reason="platform_check_rejection_path")
    except Exception as exc:
        pool = {"ok": False, "details": [str(exc)]}
    return {"safety": safety, "schema": schema, "model_pool_smoke": pool}


if __name__ == "__main__":
    print(json.dumps(run_checks(), ensure_ascii=True, indent=2))
