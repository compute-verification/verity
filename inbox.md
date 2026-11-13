[2026-11-13T10:32Z] 548d05294da9
WALKER, Fri 13 Nov, 2:38 AM PST.

**Your question: yes, lf's step takes `softmax_rows`' column case. I propose it stays word-level in this PR, because it keeps nothing past its call. That is the one place the rule falls back by design, and the trace should show its time but no growth.**

- RAN: r20261113-103030-439d is a counted copy of #9676's reader on lf tiny(2,16,64), looped twice. `softmax_rows` was called on 222 operands. The rows case was reached 16 times (K=16, n=16), all True. The column case was reached 16 times (K=16, n=16), all True. Every other call ended earlier, on `unselected` or `softmax`. INFERRED: the column case is dV = Pᵀ·dO in the backward.
- READ (`work.py` at 3a94714cfb): the column case builds `col` (K constants) for each of n columns. Those lists, together with `got`, `cells` and `slices` (which the rows case builds as well), are O(n·K) and freed when the call returns. They go into no memo, so they cost time (ctx² a call, once per product node) and never accumulate. The growth was `value_of`, which the rule removes for both cases: `info`'s constants are runs now, and `col` reads them with `_nth`.
- What the trace should show after: no site growing with nodes visited. Peak is the largest single call's transient lists plus `_leaves` (fixes 1 and 2). At s-t20 I'd expect about 100 calls of about 1M words each in `softmax_rows`, which is time, not memory (INFERRED).
- If its time shows up in the before/after, the next step is exact without listing words: sweep the rows' run ends across the columns, keeping each column's finite and non-finite counts and a sum of per-row hashes. A claim "pairwise different" is made only when every hash differs (unequal hashes prove unequal columns), and equal hashes fall back to the words. I'd do that as a follow-up, not in this PR, unless you want it here.

Status:
- The rows (`row_by_runs`) and the select (`fold`/`condition`) are written on `cursor/trn145-mask-structure-eb6e`, cut from 3a94714cfb. They follow your four conditions:
  - every verdict goes through `_generic`, `_shape` or `_finite`;
  - runs are canonical via `_runs`;
  - the counts are in `Reader.read_by`;
  - `Reader(structure=)` and `read(structure=)` are constructor arguments.
- Writing the tests next: the five hand-built bodies, plus tiny with the rule on and off and zero fallbacks.
- 54ec finished its reading in 12,460.4 s at 124.9 GiB and is in the pass now. Pass heartbeat at 14,313 s: RSS 127.4 GiB.

