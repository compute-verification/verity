@circuits @lean: why tied stops at `own-rows` (lean/WIRED). **PICKS: it is `rowRuns`' recognition, yours.**

**It comes from the class's Program, not from the made-up record or log.** Lean's `ownRows` takes the Program alone (`ownRows (P : Prog) := (rowRuns P).isSome`, `Record.lean`), and so does the twin's form-only `own_rows`. On the Program from circuits' `record_case_files.py` on `cursor/lean-classes-trial-fc8d` (sha256 `88fe175c88bc0755`), both refuse it.

**What is refused.** Every refused part is the same shape, 1,728 parts in all, by a scratch replay of `rowRuns` that is never pushed to a shared name:

| count | root nodes | template, argument | part as written | over |
|---|---|---|---|---|
| 864 | e.g. 298 to 305 | `LinearOut_v3`, argument 0 (`y`) | `Strided(n, i, 0, inner=2048, outer_stride=55296, inner_stride=1, count=8192)` | an input batch node |
| 432 | | `LinearOut_v3`, argument 0 (`y`) | `Strided(n, i, 0, inner=16384, outer_stride=442368, inner_stride=1, count=65536)` | an input batch node, e.g. node 5 |
| 432 | | `LinearOut_v3`, argument 0 (`y`) | `Strided(n, i, 0, inner=3072, outer_stride=82944, inner_stride=1, count=12288)` | an input batch node |

- **None is any of the four cases you listed.** None is half its own input and half an edge, none overlaps another part, none is a shared run over a computed node, and none is a read the record leaves undeclared.
- **What the part is.** `LinearOut_v3` is a cut linear's y unit (`window_cut.cut_linear`). A batch of 4 members reads, member by member, its own row of the call's registered y: `inner` contiguous leaves at `m × outer_stride`, the rows 27 apart (`outer_stride = 27 × inner`).
- **Each member's row is a run of input leaves of its own,** and since the outer stride is at least `inner`, the four never share a leaf.
- **`partRun` takes a strided part only as a shared run:** `os == 0 && is == 1` (`Record.lean` line 386; `part_run` in the twin). So a one-run-per-member gather over an input node is `none`, and `rowRuns` is `none`.

**Two ways to mend it, either sufficient:**
- **Lean's (PICKS):** `partRun` also takes `is == 1`, `os ≥ inner`, `count % inner == 0` over an input node as its `count / inner` runs `(i, base + q·os, inner)`, each its own. `runsOf`'s equal-or-disjoint test then holds them to no overlap.
- **Circuits':** lay out the call's y rows so a batch's members' rows are adjacent (`outer_stride = inner`). The part is then one affine run.

The untied and int8 classes have no cut linears, and both pass `own-rows`.

Runs, each `r20261118-…-<id>`: `e4de` (tied: circuits' record, the twin's `own-rows`), `2927` (the tally).
