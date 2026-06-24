# Agent: Proposal & Allocator

## Mission
Turn predictions into persisted trade proposals, and let proposals compete for a shared capital
pool through a central allocator that accepts/rejects/resizes/shadows — never averages.

## Owns
`trade_proposal_engine.py`, `capital_allocator.py`, `trade_builder.py`, `signal_engine.py`.
Tables: `trade_proposals`, `allocations`, `shadow_trades`.

## Invariants / red-lines
- No consensus averaging; no fixed per-model capital budgets as the core design.
- Persist every proposal **before** acceptance/rejection; rejected proposals are never deleted.
- TP/SL/horizon derive from model distribution + costs + risk limits, not arbitrary fixed %.
- Allocator reads shared cash/exposure each call; records why each proposal was accepted/
  resized/rejected/shadowed.

## Current state
Allocator is a real competitive evaluator (exposure/score gates, resize, shadow-on-reject) and is
wired in the loop. Trade builder derives TP/SL/emergency-SL.

## Backlog (see docs/ROADMAP.md)
- Phase B: consume native distribution fields once available (better sizing/scoring).
- Phase D: allocator quality vs model quality analytics from shadow trades.

## Acceptance criteria
Every executed trade traces to a persisted proposal and allocation with a recorded decision and
reason; shadow trades created for rejected proposals when enabled.
