# Skill: reconcile

## When to use
After a disconnect/restart, or on a schedule, to compare local ledger state against account state
and decide whether trading may resume.

## Preconditions
DB initialized; for testnet/real modes, valid credentials for the relevant client.

## Commands
```bash
python src/reconciliation_engine.py --check --mode local_paper
# testnet:
python src/reconciliation_engine.py --check --mode testnet_paper
```

## Verification
An `account_snapshots`/`balance_snapshots` row is written; a `reconciliation_events` row records
status/severity. On mismatch or lost connection, the system marks `PAUSED_NO_CONNECTION` /
`PAUSED_RECONCILIATION_ERROR` and blocks new entries until it passes.

## Red-lines
Reconciliation failure must halt new entries. Current implementation is snapshot-only; deeper
Binance fill replay + order-id matching is required before live use (roadmap Phase F / later).
