# Skill task packs

Each subfolder holds a `SKILL.md`: a repeatable operational task with **when to use it,
preconditions, exact commands, verification, and red-lines**. They encode the project's actual
workflows so they can be run consistently. They defer to the single source of truth,
[`docs/ROADMAP.md`](../docs/ROADMAP.md), and to the role packs in [`../.agents/`](../.agents/).

| Skill | Purpose |
|---|---|
| [download-data](download-data/SKILL.md) | Backfill + incrementally update market data |
| [train-models](train-models/SKILL.md) | Train candidate models (per-symbol / multi-symbol) |
| [validate-backtest](validate-backtest/SKILL.md) | Temporal validation + OOS lifecycle backtest |
| [run-paper-loop](run-paper-loop/SKILL.md) | Run the paper trading loop (run-once / loop) |
| [evaluate-promote](evaluate-promote/SKILL.md) | Evaluate paper performance + lifecycle promotion |
| [reconcile](reconcile/SKILL.md) | Run account/DB reconciliation |
| [inspect-db](inspect-db/SKILL.md) | Inspect SQLite state + attribution chain |
| [deploy-server](deploy-server/SKILL.md) | Install + run the platform on a server |
| [security-audit](security-audit/SKILL.md) | Run security scans + adversarial safety tests |
| [run-tests](run-tests/SKILL.md) | Run the test suite with coverage |

## Universal red-lines
No real order unless all four live flags are set and `DRY_RUN=false`; never bypass risk/kill-
switch/reconciliation; never commit secrets/DB/models/logs/reports; never claim guaranteed profit.
