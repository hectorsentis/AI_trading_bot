# UI / control-panel specification (Phase G)

Goal: replace the current read-mostly Streamlit report with a **real, user-oriented operations
control panel** — the thing the operator actually runs the bot from. It must feel like an
internal trading-ops console, not a toy app.

## What the operator must answer at a glance

```
Is the bot running?
Is it safe (kill switch, mode, live risk)?
Is Binance connected and reconciled?
How much capital is available / at risk?
Which models are making or losing money?
Which trades are open, and where are their TP/SL?
What failed?
What requires my action right now?
```

## Information architecture (13 sections)

1. **Global status header** — mode (research/dry-run/paper/shadow/live), bot running/paused/
   error, kill-switch state, Binance connection, last market-data + account-sync UTC,
   reconciliation status, DB path, uptime, active warnings. Status badges: OK / WARNING /
   PAUSED / ERROR / LIVE RISK / RECONCILIATION REQUIRED.
2. **Capital & risk** — total equity, free/locked USDT, invested value, realized/unrealized/
   daily PnL, total drawdown, daily-loss-limit usage, total/symbol/model exposure usage, open
   trades/orders.
3. **Binance synchronization** — public/testnet/real connectivity, last snapshot, DB-vs-Binance
   balance differences, open-order and fill reconciliation, sync errors, stale-account warnings.
4. **Active models** — id, status, lifecycle stage, symbols, timeframe, family, trained-at,
   validation/OOS result, paper age, prediction/proposal/accept/reject/shadow counts, open/
   closed trades, realized/unrealized PnL, return, drawdown, win rate, profit factor,
   calibration, degradation status, last prediction/proposal/trade.
5. **Trade proposals & allocator decisions** — proposal id, model, symbol, direction,
   confidence, expected return / adverse move, MFE/MAE, requested vs approved notional, score,
   decision, rejection reason, shadow status, UTC. (Operators must see what models *wanted* to
   do and what the allocator allowed.)
6. **Open trades & exits** — trade id, model, symbol, side, status, notional/qty, entry/current
   price, TP/SL/emergency-SL, distance to TP/SL, unrealized PnL, holding time, horizon
   remaining, exit reason, linked orders, emergency-stop status.
7. **Orders & fills** — order/client-order id, trade, model, symbol, side, type, qty, price,
   status, venue (paper/testnet/live), times, fill qty, fees, slippage, error reason.
8. **Model performance analytics** — equity & drawdown by model, PnL by model/symbol, win rate,
   profit factor, expected-vs-realized return, calibration, accepted-vs-rejected performance,
   paper-vs-shadow.
9. **Shadow analytics** — rejected proposals that would have won, accepted trades that lost,
   allocator opportunity cost, model-vs-allocator quality.
10. **Data quality & ingestion** — latest candle per symbol/TF, coverage, missing/duplicate
    bars, gap status, ingestion latency, feature/label freshness.
11. **Safety events & reconciliation** — kill-switch events, risk rejections, max-loss/stale-
    data blocks, exchange errors, timeouts, reconciliation failures, forced closures,
    emergency-stop placements/executions.
12. **Controls** — see below.
13. **Server health** — process heartbeats (ingestor/bot/evaluator/dashboard), uptime, host.

## Controls (real, audited, guarded)

```
pause / resume / quarantine / archive model
disable new entries
enable shadow-only mode
manual close trade (strong confirmation)
activate kill switch
clear kill switch (explicit confirmation + re-auth)
request retrain
show recommended CLI commands
```

Every state-changing control is **audited** (append-only log: actor + UTC + action) and gated by
**re-authentication** for anything risk-affecting. Live trading is **never** a casual button; it
requires the four env flags AND explicit confirmation AND cannot be enabled if the dashboard
lacks server-level permission.

## Data contracts

The panel consumes the **persisted** equity/PnL/snapshots produced by Phase A
(`portfolio_snapshots`, `account_snapshots`, `balance_snapshots`, `trades`, `model_performance`)
and the attribution chain. Read-only DB access (`mode=ro`) for display; all writes go through the
audited control path, never direct SQL from the UI.

## Visual standards

```
compact cards, clear hierarchy, clean tables, meaningful charts
UTC timestamps, visible units, no raw JSON dumps, no toy layout
color semantics:
  green  = ok / profit / healthy
  red    = loss / error / live risk / kill switch
  yellow = warning / degraded / paused
  gray   = unknown / inactive / no data
  blue   = informational
```

## Stack decision

Default recommendation: **harden and restructure the existing Streamlit app first** (fastest,
already integrated, read-only `mode=ro` access in place). Migrate to a lightweight web stack
(e.g. FastAPI + a JS frontend) only if interactivity, role separation, or auth requirements
exceed what Streamlit can do cleanly. Record the choice in
[DECISIONS/0004-ui-stack-choice.md](DECISIONS/0004-ui-stack-choice.md).
