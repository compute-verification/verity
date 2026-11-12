The served `DsaPositionsMerge_v1{M=8,K=2048}` is checked at its `RANGES` spot (`circuit-check --ranges DsaPositionsMerge_v1`) and no longer in `--all`. Every cold catalog build, which means the check's records run and every pytest shard's circuit_check session, stops tracing a 16.6M-gate circuit, about 330–520 s of CPU in a 31 GiB process. A cold `--all` also stops tracing it a second time in the zygote that checks it.

What leaves `--all`: the served merge's comparison against its reference at M 8, K 2,048, its digest pin, its dead-gate count and its block fit.

What stays:
- `_v1` at (1, 4) and (3, 4), the same builder, with their lowering pins.
- `DsaPositionsMerge_v2` at (8, 2048), the same function with the same ANDs, written as Calls. Nothing holds it equal to `_v1` at that size.
- `--ranges` at M 1, 3 and 8, against the same pins. No gate runs it.
- The served port test under `VERITY_ML_DEFINITIONS_SLOW` (#9281).

The risk: a change that alters the served merge only at large K would pass `check`.

The change:
- One line out of `_model_roots` in `verity/definitions/registry/dsa_topk_reads.circuit-check.py`, with its docstring.
- The binding's weight out of `circuit_check/heavy.json`.
- In its own commit, `verity/settings.py`'s description of `VERITY_ML_DEFINITIONS_SLOW` names the served merge's port test, which #9281 puts under it. It reads true once #9281 lands.

Runs: [RUNS]

Issue: CIR-255. Unmarked: circuits' coordinator decides.
