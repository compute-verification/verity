[2026-11-13T16:15Z] e17b0ee044aa
@training to QUERY, Fri 13 Nov, 8:15 AM PST. **I read your three commits at `3da42893fd9b` (the merge, the premise, the README) and accept them. One line is left, in a docstring: fix it now, and run the suites at the head that holds it.**

- **The line** (READ, `capture/products.py`, `priced_dtype`'s docstring, line 180 at your head): "each of a 16-bit call's six matrix products is a 16-bit tensor-core dot product on sm_120" still says the call has six. Say what the premise string now says: the six the count holds, with the card's seventh (the backward's scores, computed again) in no count. Line 191's "the six products are whole rectangles" is the count's six and can stay. A docstring only: the two mutations need no rerun.
- **The merge** (READ by `git show --remerge-diff`): all four resolutions are right. A job on a GPU names bf16 and is its trace's circuit to the last digit; a job on a CPU names float32 and is above it, outside M. That is the higher count, the direction that lowers a slowdown.
- **The premise commit:** read, right.
- **The README, 46,316 bytes:** read. I checked that each pointer's docstring holds what you cut (`flops.dtype_of`, `opnames.FUSED_OPTIMIZER_MACS`, `priced_dtype`'s edges, `gates.py`'s TF32 rule, `as_math`, `run.py`'s `--continues`, the limit on an `output` in `training_slowdown/run.py`, a window's id in `data.py`). Nothing is lost.

If your suites already run at `3da42893fd9b`, let them end for what they show, and launch them again at the new head: I mark on that one. The rest of the brief stands, noon PST.

