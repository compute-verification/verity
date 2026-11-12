[2026-11-12T12:43Z] 938c6cc336e9
PRICER to training, TRN-135, round 6 done. Head 1068f3c5c2 on #9170 (draft, unmarked; body 793 words, with Refs and Goal lines and the two premises under Limits).

Runs:
- r20261112-123302-6a41 (tree 1068f3c5c2, final head), RAN. Last lines: "passed circuit-properties 42 passed in 34 s", "passed benchmarks/training_slowdown 569 passed in 404 s", "2 suites: 2 passed (0 cached), 0 failed". Its lf readings at each k: never_above held, and every figure is unchanged (6,160,384 charged; blocks 0.0700404, 0.0471321, 0.0143295, 4.61358e-4, 3.27393e-6, 0, 0).
- r20261112-121246-d2bd (tree 70ed063ad4, before the min_cut path change): the same figures and the same suite counts.
- Ratchet gate at 1068f3c5c2: "1 change(s), 0 failed". The branch merges clean with origin/main, so main isn't merged in.
- The lf Program reading's every product group (key, MACs, wires) at k = 65,536 and 8e6 is byte-identical before and after the rule (local script), so there was no STOP.

Lines: 158 added, 80 deleted against d7c1069a56.

What went:
- `shared_rows` whole: the function, its call at a product site, and its call at a batch's context.
- The scan clause (two lines), with its docstring sentence and the module docstring's bullet.
- They are replaced by `may_repeat`, used at the gathered reason and in `distinct_rows`.

Tests changed (test_work.py, 17 → 23):
- New:
  - the four forms (batch, calls, scan, inputs);
  - carried rows charged as weights, with the same-iteration control;
  - W + z·0, against W unchanged;
  - `carried_loop`, which holds N, O and bound.
- loop, the ternary code, the rank bottleneck and every-operand keep their rate and β assertions on `Work.products`.
- Two activations: its product is now gathered (neither operand specific).
- test_value_reading.py: docstring only.
- With the predicate reverted, 9 tests fail.

Left with no example by hand: the "activations" reason, and bound's N at keep on a charged token-indexed product. Both need a softmax over a mask by hand; the lf test holds both.

Closing the carried-words premise would take a Status bit for "rows of the current inputs" that survives `carried`: set where a carried word is may_repeat, cleared only by a reduction over the batch's rows (a weight gradient's contraction, a sum over positions), with lf's gradients and moments shown clear. The hard part is a reduction over rows with no stated token axis.

On the stated-0 case: yes, INJECTIVE's constant test would close it in about five lines of the primitive branch (a multiply by a stated all-zero argument gives a stated 0), for finite z. I didn't do it.

Your addition is done: `min_cut()` writes the directory as "verity/protocols/inference_only/rho" from the root, and its docstring says it puts that directory on sys.path and imports `rho.min_cut`.

Not checked: the forms and premises by evaluated values (read from N); whether any frontend writes a batch as calls; a bf16 step; the tiny lf step's values.

Note: note:training-20261112T1245Z-report-pricer-trn135-round6. TRN-135 commented.

