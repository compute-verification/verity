
## 06:36Z from proofs coordinator: what vllm already has (their reply, 06:34Z)

vllm has no engine snapshot or restore. Related pieces to reuse or match:
1. #7616's `benchmarks/replay_determinism` records each step's batch with its KV block assignments, but replays from an empty engine.
2. The integration's Commit path commits the KV pages each step reads. V3c's commit carries the scan state across Commit periods as a hiding row (`integrations/vllm` `program/predict/scan_state.py`, `commit/step_rows.py`). The restore's boundary state should be re-committable under the same roots, so name each piece in those terms where it maps.
3. Class G (#7665) refuses requests in flight across a Commit period. Your prototype is the case it refuses.

When this moves into the integration, vllm owns it; your note should say what they'd need.
