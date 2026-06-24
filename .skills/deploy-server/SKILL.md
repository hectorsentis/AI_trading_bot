# Skill: deploy-server

## When to use
Install and run the platform locally or on a server as supervised long-running processes.

## Preconditions
Python environment; `.env` configured (copy from `.env.example`); persistent storage for DB/logs/
models/reports outside git.

## Commands
```bash
# Local install (Windows tooling)
.tools\install.cmd          # creates dirs, preserves .env, inits/migrates SQLite, validates schema
.tools\run.cmd              # launches ingestor + bot(paper) + evaluator + maintenance + dashboard
.tools\status.cmd           # shows runtime status

# Server processes (systemd/tmux/supervisor) - run separately
python src/realtime_ingestor.py --symbols BTCUSDT ETHUSDT SOLUSDT --timeframe 1h --loop
python src/model_maintenance.py --target-accepted-models 5 --max-attempts 50
python src/trading_bot.py --mode paper --paper-mode per-model --loop
python src/paper_model_evaluator.py --evaluate-active
streamlit run src/dashboard.py --server.address 0.0.0.0
```

## Verification
Dashboard shows `Bot: RUNNING` with recent heartbeats; `platform_checks.py` reports
`real_orders_blocked_by_default: true`.

## Red-lines
Keep DB/logs/models/reports out of git and on persistent storage. Dashboard only behind HTTPS +
reverse proxy, never raw on the internet (see [docs/SECURITY.md](../../docs/SECURITY.md)). Linux
deployment hardening is roadmap Phase E.
