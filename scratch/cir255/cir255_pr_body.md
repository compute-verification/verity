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
- In its own commit, `verity/settings.py`'s description of `VERITY_ML_DEFINITIONS_SLOW` names the served merge's port test, which #9281 put under it; #9281 has merged, so it is true now.

Runs, all on vy-nebius-cpu-1's research pool:
- `r20261112-142212-2090`: `circuit-check --no-cache --ranges DsaPositionsMerge_v1` on this branch holds at M 1, 3 and 8. At M 8 it runs what `--all` ran: six vectors evaluated, the digest pin, dead gates and the block. The definitions suite passes here (950 passed, 2 skipped).
- `r20261112-153542-f1f9` and `r20261112-163018-4cf7`: the circuit_check suite `--fresh`, with the cut reverted and then on this branch, each half with an empty `HOME` so that no records count and the session builds the catalog apart. Both pass (428 passed, 5 skipped).

| Workers | Build wait, base | Build wait, this branch | Suite, base | Suite, this branch |
|---|---|---|---|---|
| 12 | 1,004 s | 901 s | 1,346 s | 1,307 s |
| 5 (as ci's circuit-check shard) | 1,910 s | 1,615 s | 2,313 s | 2,037 s |

Issue: CIR-255. Unmarked: circuits' coordinator decides.
