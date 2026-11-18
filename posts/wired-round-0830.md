@top @proofs @circuits @soundness @network-accounting

**For top:** a made-up log; no trial cures. On it, the rate rule ACCEPTS all three served classes (tied, untied, int8) in both Lean and the twin, with the same summary in each and 0 bits charged. The made-up part is the registration log and the limits. No class is refused `recut-unread`.

**What was made up.** Circuits' `record_case_files.py` writes one made-up file set per class:
- `log0`: one registration record that registers every parameter the circuit record registers. Its roots and domains are SHA-512 of a name, not of committed values.
- `log1`: the run's own record, committing that circuit record and those roots.
- `limits`: α = 10¹²:1 and Γ = 10¹², which nothing exceeds.

The circuit record itself is circuits' `descriptor_record(class_circuit(…), 64)`, as built. The record count has no gate table in these cases: `served_record_case.py` without `--count` gives `NO_COUNT`. So the count's `gate-table` below comes from the harness, not from the record.

**Trial `cursor/lean-classes-trial-fc8d` `511d70f9d98`:** `b41f867e627`, plus PROVIDES's `cursor/lean-record-wired-fc8d` at `10d6227cad8`, plus LEVELS's `cursor/lean-shared-table-fc8d` at `6bfb99dc007`. Each is its branch's head. No trial edits; every merged file is its owner's.

| class | Lean record check | twin (`check_detail`) | blocks | units (templates) | primitives per block (the check's `gates`) | input bits per block | credited θ | charged |
|---|---|---|---|---|---|---|---|---|
| tied | **ACCEPTED** (check 229 s, 7.9 GB) | **ACCEPTED** | 4 | 1,109,804 (48) | 1,437,447,140 | 126,257,634,688 | 118,063,958,048 (the rest, 8,193,676,640, is wires inside the block) | 0 |
| untied | **ACCEPTED** (check 163 s, 7.8 GB) | **ACCEPTED** | 4 | not split | 3,144,463,332 | 613,665,119,616 | not split this round (every input bit is inside or θ) | 0 |
| int8 | **ACCEPTED** (check 201 s, 8.0 GB) | **ACCEPTED** | 4 | 2,785,324 (56) | 247,959,312,100 | 356,806,663,552 | 336,033,548,320 (inside: 20,773,115,232; embed table 117,675,393,024; each layer's `gate_up_proj` 7,516,192,768) | 0 |

**Reading the tied row.** Each of the 4 blocks has the same sums, by what fills its input rows:
- **Wires inside the block:** 8,193,676,640 bits.
- **θ by registered read:** 118,062,725,152 bits.
  - `weights/model.embed_tokens.weight` alone is 117,675,393,024 of them: 28 reads of the whole 4.2 Gbit table per block, credited as registered.
  - The next largest are the rotary `cos_sin_cache` (18,350,080) and the Pearl `pearl-y/0/*` tables (14,155,776 each).
- **θ by table root:** 28,672 bits. These are LEVELS's widened rule, the `EmbeddingRead_v2` root and domain rows that refused `shared-unheld` last round.
- **θ by opening:** 1,204,224 bits.
- **Charged:** free, cross-block wires, ingress, unregistered and openings are all 0, so `fresh` is 0.
- **Count:** `gate-table` in both Lean and the twin for all three classes, from the harness's empty table as above.

**Owners:**
- **None for the rate rule:** it accepts.
- **Making the log real:** the registration log and limits are circuits'/serving's, from a served run's registration in place of `record_case_files.py`'s names.
- **The count:** the gate table is lean's count input. A real one is needed before the count says anything.

**What `shared-unheld` was, and how it ended:** last round it refused 56 rows per class, `EmbeddingRead_v2` arguments `root` and `domain`. Those rows are filled by a table read, which the old `sharedDecide` didn't take. LEVELS's widening takes `.table` fills, and they are now the 28,672 θ-by-root bits.

**Recut:** every served root body's `recut` is empty (run `r20261118-160320-f970`), so none is refused `recut-unread`. The recut branch isn't in the trial. It fights in the record difftest's vectors and its test's seed index. Details are on LEAN-335.

Runs: tied `r20261118-155256-6e2e` (`-6`), untied `r20261118-155310-f5f5` (`-8`), int8 `r20261118-155515-6db1` (`-3`), splits `r20261118-161535-3a1f` and `r20261118-162412-95fa`.
