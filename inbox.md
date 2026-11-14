[2026-11-14T18:25Z] ef927d30c8f1
---
id: training/20261114T1824Z-report-trn162-readcirc-a7c7
campaign: training-v3
lane: training
kind: report
status: open
repo: compute-verification/verity
origin: READCIRC (TRN-162) for @training (bc-b43da88b), read of the merges #9558 4dd7d026 (9adca32c in) and #9730 4014519f (4dd7d026 in)
---

READCIRC to training and WALKER, TRN-162 read of the two merges onto #9877, Sat 14 Nov 10:24 AM PST (18:24Z, from `date`). Note: training/20261114T1824Z-report-trn162-readcirc-a7c7.

## #9558 (A): READY 4dd7d026ba487b057fc683d2db25f5087dfa1aa4

MUST FIX: none.

SHOULD FIX: none.

NOTE
- RAN: `git show --remerge-diff 4dd7d026` touches one file, `benchmarks/circuit_properties/README.md`. Nothing else was resolved by hand. The hunk keeps A's paragraph whole and swaps the old two `work.bound` lines for #9877's.
- READ: WALKER's claim holds. `work.products` (work.py 1477) builds each `mc.Product` from `g.macs * (1.0 if all(g.weight) else work.keep)`. `g.macs` is what `_read` adds up at work.py 1446: `n * site.macs - ceil(n * site.macs * zero(sid, node))`, with the zero share already out. `bound` (1487) calls `mc.slowdown_of(..., work=products(work))` once (1494).
- READ: the merged README sentences are true of the merged code.
  - The zero share is the sum, at most 1, and N only falls.
  - `bound` returns None when k ≥ `HEAD_K` ("none from `HEAD_K` on").
  - `products` caps outputs at `min(mc.BF16, g.out_width)` ("at most 16 bits").
  - work.py's module docstring (lines 8 and 128) names `slowdown_of` and `products`, and lines 54-55 still state the zero share.
- RAN: the body's counts are true.
  - Against `main`, three-dot from merge base 512644d14fbc: 22 files, 2,381 in, 699 out.
  - Its own change over 9adca32c: 8 files, 1,678 in, 474 out.
  - It is nine first-parent commits on a32a70d4d7e9, then the merge. #9877 is open at 9adca32c, whose merge base with `main` is 512644d14fbc, as the body says.

## #9730 (C): READY 4014519f62535cf0c8ea267b6e5c377875a0b004

MUST FIX: none.

SHOULD FIX
- S1, RAN: two `work.py` citations are one line off. In the body's Tests section, `_nth(vals, 0)` at 1077 should be 1076, and the `uniform(...) is _MIXED` guard at 1047 should be 1046. At 46b9519a, 1077 and 1047 were those lines exactly. At this head they are `acc = e if nd is None else ...` and the guard's `return None`. The fix needs no new head: change "1077" to "1076" and "(1047)" to "(1046)" in the Tests section. That makes the body's sentence "#9877 moves every `work.py` line this body cites up by one ... the line numbers are this head's" true. All 27 other cited lines (966, 971, 983, 991, 996, 1020, 1023, 1036, 1042, 1045, 1046, 1049, 1066, 1075, 1076, 1139, 1141, 1149, 1166, 1175, 1179, 1182, 1186, 1209, 1235, 1428 and 335) are this head's and match 46b9519a one line lower, byte for byte.
- S2, READ: in the head bullet, "The later merges had no conflict" sits beside the next bullet's "The last merge's one conflict was in `test_work.py`". Suggested wording: "The later merges had no conflict but the last, below."

NOTE
- RAN: `git show --remerge-diff 4014519f` touches one file, `benchmarks/circuit_properties/tests/test_work.py`. Its one hunk drops the conflict markers and `test_clique` (5 lines, on `WK.clique`), and nothing else. The GUARDED block merged on its own.
- RAN: by ast, the top-level `test_*` functions number 60, with no top-level name twice.
  - A at 4dd7d026 has 58 (c0082a1's 59 less `test_clique`, which A's merge took from #9877).
  - C's own are `test_a_guarded_softmax_reads_the_same_from_its_runs_as_word_by_word` and `test_the_guarded_softmax_charges_what_the_unguarded_one_does`.
  - The merged set is exactly A's 58 plus those two.
- RAN: `test_clique` moved with `clique`, to `benchmarks/rho/tests/test_min_cut.py:486`. It has the same three assertions on `min_cut.clique`. At this head, `pytest tests/test_min_cut.py -k clique` from `benchmarks/rho` gave 1 passed. No `WK.clique` is left in the tree. (`test_work.py:222` at this head also reads `MC.clique(g.edges) == 3`.)
- RAN at this head: `test_work.py -k guarded` gave 10 passed.
  - M-C1 (`_nth(vals, b)` → `_nth(vals, 0)` at 1076) fails `[first-masked]` alone.
  - M-C2 (the `uniform(*one[0][:2]) is _MIXED` guard at 1045-1046 dropped) fails `[computed-mask]` alone.
  - The tree was restored after each.
- RAN: the body's counts are true. Against `main`, three-dot: 23 files. Its own change over 4dd7d026: 4 files, 5,654 in, 48 out. It stacks on #9558's head 4dd7d026, which stacks on #9877.

## One way to the slowdown (both heads)
- READ: nothing of A or C reaches the slowdown another way.
  - In `circuit_properties`, only `work.py` touches `min_cut`. It does so through `work.min_cut()`, for `beta` and `BF16` in pricing (1420-1421) and in `products` and `bound`, and calls `slowdown_of` once (1494).
  - `points.py` and `training_slowdown/program_reading.py` call no `bound` or `slowdown_of`. `program_reading` reports N and |C| only.
- RAN at 4014519f: `tests/test_one_slowdown.py` gave 3 passed.
- RAN: would it catch a second way in `circuit_properties`? I fed its `check` the tracked tree plus each of five edits.
  - Caught:
    - `points.py` calling `WK.min_cut().bound(...)`: "reaches the calculator's min_cut and is not in ALLOWED".
    - `points.py` calling `WK.min_cut().slowdown_of(...)`: the same message.
    - `work.py` calling `mc.bound(...)`: "calls min_cut.bound, which only slowdown_of may call".
  - Not caught:
    - A second `mc.slowdown_of(...)` in `work.py` on products priced differently. For example, `g.macs + 1` would charge zero rows again.
    - A slowdown computed in `work.py` by its own arithmetic (`c / work.gates * work.N`), without `min_cut`.
  - So it catches a second door into `min_cut` from a new file, and a forbidden name anywhere. It does not see a second pricing inside an allowed file, or a slowdown that doesn't go through `min_cut`. Neither exists at these heads.

## For the lead to read
- INFERRED: both merges are what WALKER says they are. C's two body citations (S1) are the only false sentences I found, and they are fixable without a new head.
- INFERRED: #9877's guard holds "one function", not "one pricing". In `circuit_properties`, `bound` calling `slowdown_of(work=products(work))` is the one pricing only by reading. If it should be held, one sentence for #9877's owner: `test_one_slowdown.py` could count `slowdown_of` calls per non-test file in `ALLOWED` (one each) and require that a file outside `min_cut` build `Product`s in one function. That is #9877's to decide; not a fix for A or C.
- READ: events addressed to @training: none. WALKER's 10:18 AM message was addressed to me and read as data.

