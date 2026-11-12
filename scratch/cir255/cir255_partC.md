**The served `DsaPositionsMerge_v1`'s 401 s is its trace: 340 s of the 350 s build untraced, 387 s traced. The descriptor adds 10 s, and the walk of what it reaches is under 0.1 s. Its check in `--all` (107.5 s) is the comparison against the reference on its vectors (57 s) and the SHA-512 digest of its 16.6M gates (39 s). `_v2` at the same statics builds in 0.06 s and checks in 5.3 s, because its description is its rows of Calls and their small pieces, not every gate. The cheapest cut that changes no Definition is to check the served `_v1` at its existing `RANGES` spot (`--ranges`) instead of in `--all`. That is three lines, in PR [PRNUM] (unmarked). It saves the catalog's heaviest binding, 31 GiB and 327–522 s of CPU, in every cold catalog build: [AB]. It stops every candidate from comparing the served merge's digest pin.**

### Where the seconds go

| | `DsaPositionsMerge_v1{M=8,K=2048}` | `DsaPositionsMerge_v2{M=8,K=2048}` | Read in |
|---|---|---|---|
| Gates | 16,623,434 (6,046,233 ANDs) | 16,623,441, counted through its Calls | `r20261112-135225-3bf5` |
| Build: the trace (the body) | 340.3 s CPU, 31.2 GiB peak | the binding (with its pieces): 0.06 s | `3bf5`; `r20261112-125849-7459` |
| Build: what it reaches | 0.0 s | 0.0 s | same |
| Build: the descriptor (canonical JSON and SHA-512 of each Definition) | 9.7 s | 0.1 s | same |
| The binding built apart, untraced | 326.9 s CPU (cpu-7), 30.95 GiB | 0.06 s | `7459` |
| The binding built apart, traced as the records run does it | 387.1 s CPU (cpu-9), 30.94 GiB | – | `3bf5` |
| The same build in the pytest shards' suite sessions (17 shards read) | 399–522 s CPU | – | their `resources.jsonl` |
| Check in `--all`: total | 107.5 s, 11.3 GiB peak | 5.3 s | `r20261112-104146-92db`'s progress log |
| comparison against the reference (`vectors` 4, `edge_vectors` 2; no lowering realization: 6.0M ANDs is over the 65,536 whole-Definition budget) | 57.3 s | 4.1 s | same, and `3bf5` |
| digest (SHA-512 of every gate, against its pin) | 39.2 s | with dead gates and the rest: 0.04 s | same |
| lowering | 8.9 s | 1.1 s | same |
| dead gates | 2.1 s | – | same |
| partition | not checked (16.6M gates is over the 5M budget) | – | `3bf5` |

The same check in one process on the built body took 96.2 s (`3bf5`). It made no second trace and found no failures.

**A cold `--all` traces it twice [inferred].** In `92db`, e52d's circuit-check shard, the records run's child for this binding ran 10:48:31–10:54:34 (363 s of CPU, 31.5 GiB). After the prologue, one check zygote grew from 2 to 29 GiB on about 420 s of CPU (11:07–11:15) and dropped to 5.8 GiB. 5.8 GiB is the `rss_start` of the `_v1` check. That is `cli._made` making a target that missed in the zygote that forks its check. I read it from `resources.jsonl`, which names no target, so the target is inferred from the shape. So in a cold check this one target costs about 363 + 420 + 108 ≈ 890 s of CPU **[computed]**.

**Why `_v1` costs twenty times `_v2` in the check.** `_v1` is one traced circuit, so its description is every gate. The comparison evaluates all 16.6M gates on each vector, and the digest hashes all of them. `_v2` is the same function with the same signature and the same ANDs, written as Calls (`DsaPositionsKeep_v1`, `DsaPositionsShift_v1`, `DsaPositionsShiftIn_v1`, the two offsets). Its description is those rows, one more per bit of K, and the pieces pinned beside it are 32–57 ANDs each, checked once at their own size. The check's cost follows the description, not the unrolled circuit.

### The cut, and what each option stops checking

1. **Check the served `_v1` at its `RANGES` spot, outside `--all`.** This is the one I wrote: PR [PRNUM], branch `cursor/circuits-cir255-served-merge-build-6f04`, unmarked.
   - The change: one line out of `_model_roots` in `verity/definitions/registry/dsa_topk_reads.circuit-check.py`, its docstring, and the binding's weight in `circuit_check/heavy.json`. It doesn't touch `targets.py` or `ranges.py`.
   - What leaves `--all`: the served merge's comparison against the reference at M 8, K 2,048, its digest pin, its dead-gate count and its block fit.
   - What stays:
     - `_v1` at (1, 4) and (3, 4), the same builder, with their lowering pins.
     - `_v2` at (8, 2048) in `--all`. It is the same function, but nothing holds it equal to `_v1` at that size.
     - `circuit-check --ranges DsaPositionsMerge_v1`, which checks M 1, 3 and 8 at K 2,048 against the same pins. No gate runs it.
     - The served port test under `VERITY_ML_DEFINITIONS_SLOW`, nightly once #9281 lands.
   - The risk: a change that alters the served merge only at large statics, for instance in the 12-stage shifter or the clamp at K, would pass the gate. It would show only under `--ranges` or in a nightly.
   - Saving per cold catalog build (the check's records run and every pytest shard's session): 327–522 s of CPU and a 31 GiB process **[read]**. On the suite's wall time: [AB]. A cold `--all` also drops the zygote's second trace and the 107.5 s check, about 530 s of CPU **[inferred]**.
2. **Fewer or smaller vectors.** The comparison is 57 s of the 107.5 s check (`vectors` 4, `edge_vectors` 2). One or two vectors would save at most about 40 s of one check in a cold `--all` **[estimate]**, and nothing in any catalog build, where the 31 GiB and the 340 s sit. It would stop the edge vectors or the random ones at served size. Not worth a PR.
3. **Reuse the trace between the build and the check.** The trace cache (`verity/definitions/ops/boolean/trace_cache.py`) could carry the records run's trace to the zygote. That would save the second trace, about 420 s of CPU and 27 GiB in a cold `--all` **[inferred]**, and stop checking nothing. It would not help the suite's session, which only builds. It belongs in circuit-check's process setup and records, which aren't mine, and it is more than a few lines, so I describe it and stop.
4. **Not a cut under this rule, but the clean one: serve `_v2`.** The long row's `index` would bind `DsaPositionsMergeV2` (same function, signature and ANDs). Then `_v1`'s served binding could go with no loss, since `_v2`'s 5.3 s check and digest pin would cover the served circuit. It changes the long row's circuit and its pins. That's a Definition change for you to decide.

### The runs

- `r20261112-125849-7459` (cpu-7, slot r3): the binding built apart untraced, and `_v2`'s binding and pieces. Its cProfile'd in-process build was too slow to finish inside the run's limit, so I cancelled it.
- `r20261112-135225-3bf5` (cpu-9, slot r5): the in-process build's phases, the check on the built body, and the binding built apart traced.
- `r20261112-141120-09c0` (cpu-7): on the branch, `circuit-check --no-cache --ranges DsaPositionsMerge_v1` and the circuit_check and definitions suites, `--fresh`; then the circuit_check suite again with the cut reverted, on the same node and cores. [RESULTS]

`verity/settings.py`'s description of `VERITY_ML_DEFINITIONS_SLOW` now names the served merge's port test. That commit is in the same PR, and it reads true once #9281 lands.
