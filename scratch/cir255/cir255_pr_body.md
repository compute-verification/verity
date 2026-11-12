The served `DsaPositionsMerge_v1{M=8,K=2048}` is checked at its `RANGES` spot (`circuit-check --ranges DsaPositionsMerge_v1`) and no longer in `--all`. After this, the merge at (8, 2048) is registered by its `RANGES` rule like the rule's other spots, and `--all` checks the family at (1, 4) and (3, 4).

Every cold catalog build, which means the check's records run and every pytest shard's circuit_check session, stops tracing a 16.6M-gate circuit, about 330–520 s of CPU in a 31 GiB process. A cold `--all` also stops tracing it a second time in the zygote that checks it.

What leaves `--all`:
- The served merge's comparison against its reference at M 8, K 2,048, its digest pin, its dead-gate count and its block fit.
- The by-calls roots that call this merge, and what each relies on for it after the change, from `r20261112-182439-0086`, checking both sidecars on this head merged into `main` `30d6fcc92ff7`, then on `main`:
  - `DsaTopKReadRows_v1{N=131072,TOPK=2048,BLOCK=16384,READ=EmbeddingRead_v1{V=131072,H=576}}` (this sidecar's `_by_calls_roots`) relies on `DsaPositionsMerge_v1{M=1,K=2}`, differing in K and M; on `main` it relies on `DsaPositionsMerge_v1{M=8,K=2048}`. The root passes in both.
  - `DsaTopKReadRowsCut_v1{N=131072,TOPK=2048,BLOCK=16384,READ=EmbeddingRead_v1{V=131072,H=576}}` (vllm's `long_row.circuit-check.py`, `_by_calls_roots`) relies on the same `{M=1,K=2}`, differing in K and M; on `main` it relies on `{M=8,K=2048}`. The root passes in both.
  - No other callee row of any by-calls root differs from `main`'s (81 rows each). `--update-pins` on that tree moves no pin file, vllm's included.

What stays:
- `_v1` at (1, 4) and (3, 4), the same builder, with their lowering pins.
- `DsaPositionsMerge_v2` at (8, 2048), the same function with the same ANDs, written as Calls. Nothing holds it equal to `_v1` at that size.
- `--ranges` at M 1, 3 and 8, against the same pins. No gate runs it.
- The served port test under `VERITY_ML_DEFINITIONS_SLOW` (#9281).

The risk: a change that alters the served merge only at large K would pass `check`. vllm's check template names this merge as a unit today. A change to `_merge` runs `--ranges DsaPositionsMerge_v1` and puts its report in the PR, as the Circuits rule in AGENTS.md asks of a change to a circuit; no gate runs it for that change.

The change:
- One line out of `_model_roots` in `verity/definitions/registry/dsa_topk_reads.circuit-check.py`, with its docstring.
- The binding's weight out of `circuit_check/heavy.json`.
- In its own commit, `verity/settings.py`'s description of `VERITY_ML_DEFINITIONS_SLOW` names the served merge's port test, which #9281 put under it; #9281 has merged, so it is true now.

Runs:
- `r20261112-142212-2090` (vy-nebius-cpu-1): `circuit-check --no-cache --ranges DsaPositionsMerge_v1` on this branch holds at M 1, 3 and 8. At M 8 it runs what `--all` ran: six vectors evaluated, the digest pin, dead gates and the block.
- `r20261112-153542-f1f9` and `r20261112-163018-4cf7` (vy-nebius-cpu-1): the circuit_check suite `--fresh`, with the cut reverted and then on this branch, each half with an empty `HOME` so that no records count and the session builds the catalog apart. Both pass (428 passed, 5 skipped).
- `r20261112-182439-0086` (vy-nebius-cpu-14), its source this head: `circuit-check` of `dsa_topk_reads.circuit-check.py` and vllm's `long_row.circuit-check.py` on this head merged into `main` `30d6fcc92ff7` (86 targets, 0 failures), again with `--update-pins` (86 targets, 0 failures, no file moved), then on `main` (87 targets, 0 failures). At this head `verity-config` (13 passed) and `verity-cli` (108 passed); the run reached its time limit inside the repository suite.
- `r20261112-211708-8db5` (vy-nebius-cpu-11), its source this head: `repository` alone, then circuit_check, definitions, `verity-config` and `verity-cli`, then vllm's `-k settings`, all `--fresh`.

| Workers | Build wait, base | Build wait, this branch | Suite, base | Suite, this branch |
|---|---|---|---|---|
| 12 | 1,004 s | 901 s | 1,346 s | 1,307 s |
| 5 (as ci's circuit-check shard) | 1,910 s | 1,615 s | 2,313 s | 2,037 s |

Each half of the two comparisons ran once, on a shared node.

`adapters/vllm/tools/ratchet_gate.py origin/main HEAD --json` at this head: `"failures": []`.

Lane circuits
Refs CIR-255
Goal infra/full-check-45-min
