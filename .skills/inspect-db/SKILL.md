# Skill: inspect-db

## When to use
Verify state, debug, or confirm the model/trade-level PnL attribution chain.

## Preconditions
DB exists at `SQLITE_DB_PATH` / `config.DB_FILE`.

## Commands
```bash
python src/db_utils.py --check-schema

# Useful queries (sqlite3 or any client)
SELECT status, COUNT(*) FROM model_registry GROUP BY status;
SELECT model_id, account_mode, status, COUNT(*) FROM orders GROUP BY model_id, account_mode, status;
SELECT * FROM model_performance ORDER BY timestamp_utc DESC LIMIT 20;
SELECT * FROM reconciliation_events ORDER BY rowid DESC LIMIT 10;
```

## Verify the attribution chain
Every executed trade/order/fill should carry:
`model_id, prediction_id, proposal_id, allocation_id, trade_id, order_id, fill_id`. Cross-check
`model_predictions -> trade_proposals -> allocations -> trades -> orders -> fills`.

## Red-lines
Inspect read-only; never edit ledger/audit history by hand. Use the audited control path for any
state change.
