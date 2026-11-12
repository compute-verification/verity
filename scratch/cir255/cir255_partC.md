**The served `DsaPositionsMerge_v1{M=8,K=2048}` is slow because it is traced. Its build is about 350 s of CPU and 31 GiB, and 340 s of that is the trace: the descriptor adds 10 s, and the walk of what it reaches takes under 0.1 s. Its 107.5 s check in `--all` is the comparison against the reference (57 s) and the SHA-512 digest of its 16.6M gates (39 s). `_v2` at the same statics builds in 0.06 s and checks in 5.3 s, because its description is rows of Calls, not every gate. The cheapest cut that changes no Definition is to check the served `_v1` at its existing `RANGES` spot (`circuit-check --ranges`) instead of in `--all`. That PR is #9390, unmarked: one sidecar line with its docstring and one `heavy.json` line. It removes the catalog's heaviest binding from every cold catalog build. With no records, the circuit_check suite's catalog build then took 1,615 s instead of 1,910 s at 5 workers, as ci now runs it, and the whole suite took 2,037 s instead of 2,313 s. At 12 workers the build took 901 s instead of 1,004 s. The gate stops checking the served merge itself; `--ranges`, which no gate runs, still checks it and passes.**

### Where the seconds go

| | `DsaPositionsMerge_v1{M=8,K=2048}` | `DsaPositionsMerge_v2{M=8,K=2048}` | Read in |
|---|---|---|---|
| Gates | 16,623,434 (6,046,233 ANDs) | 16,623,441, counted through its Calls | `r20261112-135225-3bf5` |
| Build: the trace (the body) | 340.3 s CPU, 31.2 GiB peak | the binding (with its pieces): 0.06 s | `3bf5`; `r20261112-125849-7459` |
| Build: what it reaches | 0.0 s | 0.0 s | same |
| Build: the descriptor (canonical JSON and SHA-512 of each Definition) | 9.7 s | 0.1 s | same |
| The binding built apart, untraced | 326.9 s CPU (cpu-7), 30.95 GiB | 0.06 s | `7459` |
| The binding built apart, traced as the records run does it | 387.1 s CPU (cpu-9), 30.94 GiB | – | `3bf5` |
| The same build in the suite's sessions (17 pytest shards, and the A/B below) | 396–522 s CPU, 31.4 GiB | – | their `resources.jsonl` |
| Check in `--all`: total | 107.5 s, 11.3 GiB peak | 5.3 s | `r20261112-104146-92db`'s progress log |
| comparison against the reference (`vectors` 4, `edge_vectors` 2, evaluated; no lowered realization, because 6.0M ANDs is over the 65,536 whole-Definition budget) | 57.3 s | 4.1 s | same, and `3bf5` |
| digest (SHA-512 of every gate, against its pin) | 39.2 s | with dead gates and the rest: 0.04 s | same |
| lowering | 8.9 s | 1.1 s | same |
| dead gates | 2.1 s | – | same |
| partition | not checked (16.6M gates is over the 5M budget) | – | `3bf5` |

The same check in one process on the built body took 96.2 s (`3bf5`). It made no second trace and found no failures.

**A cold `--all` traces it twice [inferred].** In `92db`, e52d's circuit-check shard, the records run's child for this binding ran 10:48:31–10:54:34 (363 s of CPU, 31.5 GiB). After the prologue, one check zygote grew from 2 to 29 GiB on about 420 s of CPU (11:07–11:15) and dropped to 5.8 GiB, which is the `_v1` check's `rss_start`. That is `cli._made` making a target that missed, in the zygote that forks its check. `resources.jsonl` names no target, so I inferred which one from the shape. In a cold check this one target then costs about 363 + 420 + 108 ≈ 890 s of CPU **[computed]**.

**Why `_v1` costs twenty times `_v2` in the check.** `_v1` is one traced circuit, so its description is every gate: the comparison evaluates all 16.6M gates on each vector, and the digest hashes all of them. `_v2` is the same function, with the same signature and the same ANDs, written as Calls (`DsaPositionsKeep_v1`, `DsaPositionsShift_v1`, `DsaPositionsShiftIn_v1` and the two offsets). Its description is those rows, one more per bit of K. The pieces pinned beside it are 32–57 ANDs each, checked once at their own size. The check's cost follows the description, not the unrolled circuit.

### The cut, and what each option stops checking

1. **Check the served `_v1` at its `RANGES` spot, outside `--all`.** I wrote this one: PR #9390 (head `df3b53faac`), branch `cursor/circuits-cir255-served-merge-build-6f04`, unmarked.
   - The change: one line out of `_model_roots` in `verity/definitions/registry/dsa_topk_reads.circuit-check.py`, its docstring, and the binding's weight in `circuit_check/heavy.json`. It doesn't touch `targets.py` or `ranges.py`.
   - What leaves `check`: the served merge's comparison against the reference at M 8, K 2,048, its digest pin, its dead-gate count and its block fit.
   - What stays:
     - `_v1` at (1, 4) and (3, 4), the same builder, with their lowering pins.
     - `_v2` at (8, 2048) in `--all`. It is the same function, but nothing holds it equal to `_v1` at that size.
     - `circuit-check --ranges DsaPositionsMerge_v1`. It checks M 1, 3 and 8 at K 2,048 against the same pins, and at M 8 it runs exactly what `--all` ran: the six vectors evaluated, the digest pin (722b747a…, ok), dead gates (0) and the block (k 23). On the branch it holds: 3 spots in 431 s wall at `-j 20`, 31.5 GiB peak. No gate runs it.
     - The served port test under `VERITY_ML_DEFINITIONS_SLOW`, which #9281 put there (merged 17:17Z).
   - The risk: a change that alters the served merge only at large statics, for instance in the 12-stage shifter or the clamp at K, would pass `check`. It would show only under `--ranges` or the slow tests.
   - Saving, measured on cpu-1 with no records, so both sessions built the catalog apart:
     - At 12 workers (`r20261112-153542-f1f9`), the workers waited 1,004 s for the build on the base and 901 s on the branch **[read]**. The whole suite took 1,346 s and 1,307 s. The base build's 31.4 GiB, 396 s child is gone, and its largest child is now 21.5 GiB.
     - The builds' sampled CPU was equal, 8,485 s and 8,505 s. The other bindings took about as much more on the branch as the merge took on the base, so on this shared node (`nice 19`, other lanes' runs beside it) the CPU sums can't resolve one 400 s binding.
     - At 5 workers (`r20261112-163018-4cf7`, `--jobs 5 --mem-gb 60`, the worker count ci's circuit-check shard gives the suite), the workers waited 1,910 s for the build on the base and 1,615 s on the branch, 295 s or 15% less **[read]**. The whole suite took 2,313 s and 2,037 s. I assumed ci's shard has the same 40 GB worker share; I didn't read its slot.
     - Why more at 5 workers than at 12: the build's budget at 5 workers is about 37 GiB (5 × 8 GB, less what the waiting workers hold), and the merge holds 31 of it for 7 minutes. On the base, the merge ran 16:31:02–16:38:01, and the next heavy binding (12.85 GiB) started at 16:38:02. On the branch, the 12.8 GiB and 9.0 GiB bindings started together at 17:09:32. At 12 workers the budget is larger, so the merge mostly adds its 400 s of CPU spread over 12 jobs **[inferred from those start times]**.
     - The sampled build CPU at 5 workers was 7,273 s on the base and 6,275 s on the branch. That is 1,000 s less, of which the merge is 420 s, so here too the node's noise is the size of one binding.
     - A cold `--all` also drops the zygote's second trace and the 107.5 s check, about 530 s of CPU **[inferred]**.
2. **Fewer or smaller vectors.** The comparison is 57 s of the 107.5 s check. One or two vectors would save at most about 40 s of one check in a cold `--all` **[estimate]**, and nothing in a catalog build, where the 31 GiB and the 340 s sit. It would stop the edge vectors or the random ones at served size. Not worth a PR.
3. **Reuse the trace between the build and the check.** The trace cache (`verity/definitions/ops/boolean/trace_cache.py`) could carry the records run's trace to the zygote. That would save the second trace in a cold `--all`, about 420 s of CPU and 27 GiB **[inferred]**, and stop checking nothing. It would not help the suite's session, which only builds. It belongs in circuit-check's process setup and records, which aren't mine, and it is more than a few lines, so I describe it and stop there.
4. **Not a cut under this rule, but the clean one: serve `_v2`.** The long row's `index` would bind `DsaPositionsMergeV2`, with the same function, signature and ANDs. Then `_v1`'s served binding could go with no loss, since `_v2`'s 5.3 s check and digest pin would cover the served circuit. It changes the long row's circuit and its pins, so it is a Definition change for you to decide.

### The runs

| Run | Where | Waited for a slot | Held it | What I read |
|---|---|---|---|---|
| `r20261112-125849-7459` | cpu-7, r3 | 5 s | 53 min, then cancelled | the binding built apart untraced; `_v2`'s binding and pieces. The cProfile'd in-process build was too slow to finish. |
| `r20261112-135225-3bf5` | cpu-9, r5 | 8 s | 14 min 44 s | the in-process build's phases, the check on the built body, the binding built apart traced |
| `r20261112-141120-09c0` | cpu-7 | 10 min, no slot | – | cancelled: it queued behind ci's check |
| `r20261112-142212-2090` | cpu-1 research pool, r12 (20 cores) | 13 min 11 s | 36 min 40 s | `--ranges` on the branch (holds, as above); the definitions suite on the branch passed (950 passed, 2 skipped). Its circuit_check halves don't compare: the base session took its catalog from records cpu-1's cache held for main (329 s, no build apart), while the branch's changed sidecar missed them (1,432 s, built apart). |
| `r20261112-151952-8384` | cpu-1 research pool, r13 | 26 s | under a minute | failed at once: the script path wasn't expanded |
| `r20261112-153542-f1f9` | cpu-1 research pool, r4 (12 cores) | 4 min 20 s | 44 min 21 s | the A/B at 12 workers, each half with an empty `HOME`, so no records: both pass (428 passed, 5 skipped) |
| `r20261112-163018-4cf7` | cpu-1 research pool, r4 (12 cores) | 8 s | 72 min 39 s | the A/B at 5 workers, no records: both pass (428 passed, 5 skipped) |

The PR also corrects `verity/settings.py`'s description of `VERITY_ML_DEFINITIONS_SLOW` to name the served merge's port test, in a commit of its own. #9281, which put that test under the setting, merged at 17:17Z, so the description is now true.
