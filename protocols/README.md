# Protocols (prototype, not for merge)

Three minimal protocols, and a toy CPU inference engine that composes them, to let the protocol API emerge from code:

- `sampled_proofs/` (`verity_sampled_proofs`): commits each serving step, samples replay units and verification
  units from beacon rounds published after each commitment (during serving or at audit), checks the boundary
  exhaustively and sampled units by replay, and emits an `IntegrityProfile`.
- `pouw/` (`verity_pouw`): device-bound seeds issued at admission, per-step noise, tickets collected at step end, and
  a work claim that reads the profile.
- `pous/` (`verity_pous`): design C with auditor-secret filler, timed linear-MAC audits that interrupt serving, a KV
  write tap, and a KV-usefulness claim that reads the profile.
- `examples/toy_engine/`: the application. `engine.py` is a request queue, batched decode steps (one small IR Program
  per batch size), weights and a KV pool in memory regions, a simulated clock, and hook points (admit, alloc,
  step_start, commit, step_end, idle). `wiring.py` hooks the protocols in three ways; `demo.py` runs setup, serving
  and audit.

Protocols import only the core (`tests/test_protocol_boundaries.py`).

~~~text
uv run python protocols/examples/toy_engine/demo.py            # --sample audit, --retention hbm, --no-preempt [--force], --wiring phases, --k 10
uv run pytest protocols
~~~

Shared core pieces, each because two protocols use it: `verity.randomness.derive` (all three) and
`verity.proofs.IntegrityProfile` (produced by sampled proofs, read by PoUW, PoUS and the demo's output policy).
