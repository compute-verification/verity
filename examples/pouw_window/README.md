# The PoUW window example

One served int8 window of PoUW (`verity/core/protocols/pouw`) taken through today's spot check,
`PouwSegCheckDither_v2{S, duo}` (E2E-120): the honest window and every attack in `attacks.py` played against the round,
each accepted or rejected, and the served design's ε and γ beside them. It exits 0 when the honest window is accepted and
each attack is rejected exactly when today's check is meant to catch it (MITM and skip-filler aren't, each with its reason).
It is compute-accounting's.

## Workload and placement

There is no workload yet (`example.toml`: `workload = "none"`): `window.Window.stand_in` seeds int8 W and X until vLLM's
int8 window exists, and `run.py`'s docstring names where the real values plug in. Every party runs in one process on one
CPU machine: the prover (`attacks.honest`, and each attack) and the round's check (`window.Round`), whose proof is
`window.ideal_proof`, the statement's public file computed from its rows, with no C-Flock. `example.toml` puts the example
at level 0 (moved) on the ladder (`examples/README.md`).

## Commands

~~~text
PYTHONPATH=benchmarks/pouw uv run verity example pouw-window [--m M --n N --k K --window S] [--seed SEED] [--json]
uv run verity example pouw-window --stage OUT    # one honest statement's circuit.txt, inst.bin and pub.bin
~~~

`PYTHONPATH=benchmarks/pouw` is for the headline's ε budget (`accounting.headline` imports PoUW's `eps_budget.py`), which
the window's check doesn't need. `--stage OUT` stages the statement through C-Flock's `UnitStager` for C-Flock `--zk` and
the Lean verifier of record; the pod job that proves it, which `run.py`'s docstring calls `job.sh`, isn't in the tree.

- `window.py`: the round (registered W, committed X, the coin, the committed receipts, the draw, the statement) and the
  honest prover, `Honest`. Nothing is retried or redrawn: a round refused at any step is rejected (`Refused`).
- `attacks.py`: the attack library, each a prover that departs from `Honest` at one step, and `ATTACKS`, whether today's
  check catches each.
- `accounting.py`: the served design's headline, H1 + fx1 on the RTX PRO 6000 (sm_120), from PoUW's machine model
  (`benchmarks/pouw/machine_model`).
