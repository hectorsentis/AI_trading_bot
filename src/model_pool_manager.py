import json
from dataclasses import dataclass
from pathlib import Path

from config import (
    ACCOUNT_MODE_TESTNET_PAPER,
    AUTO_REPLACE_REJECTED_MODELS,
    MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    TARGET_ACCEPTED_MODELS,
    TIMEFRAME,
    SYMBOLS,
    TRAINING_SCOPE,
)
from model_registry import (
    activate_model_for_paper,
    list_backtest_accepted_models,
    list_paper_active_models,
)
from model_maintenance import maintain_model_pool as _legacy_train_maintain


@dataclass
class PoolSummary:
    target: int
    training_scope: str
    paper_active_before: list[str]
    activated: list[str]
    training_summary: dict
    paper_active_after: list[str]
    target_met: bool


def maintain_paper_model_pool(
    symbols: list[str] | None = None,
    timeframe: str = TIMEFRAME,
    target_accepted_models: int = TARGET_ACCEPTED_MODELS,
    max_attempts: int = MAX_TRAINING_ATTEMPTS_PER_CYCLE,
    account_mode: str = ACCOUNT_MODE_TESTNET_PAPER,
    training_scope: str = TRAINING_SCOPE,
) -> dict:
    symbols = [s.upper() for s in (symbols or SYMBOLS)]
    training_scope = (training_scope or "multi_symbol").replace("-", "_")
    if training_scope == "both":
        multi = maintain_paper_model_pool(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=target_accepted_models,
            max_attempts=max_attempts,
            account_mode=account_mode,
            training_scope="multi_symbol",
        )
        per = maintain_paper_model_pool(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=target_accepted_models,
            max_attempts=max_attempts,
            account_mode=account_mode,
            training_scope="per_symbol",
        )
        return {
            "target": int(target_accepted_models),
            "training_scope": "both",
            "multi_symbol": multi,
            "per_symbol": per,
            "target_met": bool(multi.get("target_met")) and bool(per.get("target_met")),
        }
    if training_scope == "per_symbol" and len(symbols) > 1:
        per_symbol = {}
        for symbol in symbols:
            per_symbol[symbol] = maintain_paper_model_pool(
                symbols=[symbol],
                timeframe=timeframe,
                target_accepted_models=target_accepted_models,
                max_attempts=max_attempts,
                account_mode=account_mode,
                training_scope=training_scope,
            )
        return {
            "target": int(target_accepted_models),
            "training_scope": training_scope,
            "per_symbol": per_symbol,
            "target_met": all(item.get("target_met") for item in per_symbol.values()),
        }
    active_before = list_paper_active_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols)
    activated: list[str] = []

    while len(active_before) + len(activated) < int(target_accepted_models):
        available = list_backtest_accepted_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols)
        if not available:
            break
        candidate = available[0]
        activate_model_for_paper(candidate["model_id"], account_mode=account_mode)
        activated.append(candidate["model_id"])

    training_summary: dict = {}
    active_after_activation = list_paper_active_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols)
    if len(active_after_activation) < int(target_accepted_models) and int(max_attempts) > 0:
        training_summary = _legacy_train_maintain(
            symbols=symbols,
            timeframe=timeframe,
            target_accepted_models=int(target_accepted_models),
            max_attempts=int(max_attempts),
            training_scope=training_scope,
        )
        # Activate any newly accepted backtest models.
        for candidate in list_backtest_accepted_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols):
            if len(list_paper_active_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols)) >= int(target_accepted_models):
                break
            activate_model_for_paper(candidate["model_id"], account_mode=account_mode)
            activated.append(candidate["model_id"])

    active_after = list_paper_active_models(timeframe=timeframe, training_scope=training_scope, symbols=symbols)
    return PoolSummary(
        target=int(target_accepted_models),
        training_scope=training_scope,
        paper_active_before=[m["model_id"] for m in active_before],
        activated=activated,
        training_summary=training_summary,
        paper_active_after=[m["model_id"] for m in active_after],
        target_met=len(active_after) >= int(target_accepted_models),
    ).__dict__


if __name__ == "__main__":
    print(json.dumps(maintain_paper_model_pool(), ensure_ascii=True, indent=2))
