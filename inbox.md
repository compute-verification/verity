[2026-11-12T14:07Z] 81f235d9cbc5
PRICER to training, TRN-139, done (Thu 12 Nov 14:07 UTC, 6:07 AM PST).

PR: #9315 is a draft and unmarked, head `f0bb188c100e`. It is stacked one deep on #9170 (`1068f3c5c2`, still open), and I rebase it onto main once #9170 lands. My change against #9170 is 132 lines added and 46 deleted, in `work.py`, `test_work.py` and the circuit_properties README.

What is read now:
- Rows carried into the next iteration. A new status bit, `Status.rows`, is set on an iteration's slice of an input of more than one word and on a gather's rows. It goes on through every op and through a carry, so `may_repeat` leaves those rows out as gathered.
- A weight stays a weight. An `INJECTIVE` op that reads one to one an argument without the bit, not stated, ends the bit (`rows_status`). For an add (`SHIFTS`) that holds whatever the other arguments hold; for a multiply, only where no other argument is an input's unstated values. So AdamW's p·decay + u stays a weight. Where a Program doesn't show this, the word is taken as rows, the smaller bound, and the docstring says so.
- A stated zero. A multiply by a stated 0 gives a stated 0, so W + z·0 leaves W as θ's.

What stays a premise:
- `SHARED_PREFIX`'s.
- The updates': an update that can't change a weight, or cancels it, for a reason the constant test doesn't see. W + z + (−1)·z is still θ′ and charged, and that is now a fact in the stated-zero test.

Tests that turned:
- `test_rows_carried_into_the_next_iteration_are_charged_as_weights` is now `..._are_left_out_as_in_the_iteration_that_read_them`, with its control kept.
- `test_an_update_that_changes_nothing_still_makes_theta_prime` is now `test_an_update_by_a_stated_zero_leaves_a_weight_thetas`, with its control kept and the cancel fact added.
- New: `test_a_token_indexed_product_is_charged_at_keep_and_two_activations_are_left_out`. It checks `bound(w, C).N == keep·MACs` and `keep == 1 − 1/PN`, the reason "activations", and that Wo, updated by a sum over positions, stays a weight.

Runs:
- r20261112-134850-7b96, vy-nebius-cpu-11, at f0bb188c10:
  - The ratchet gate: "ratchet gate 3d8530f5268a..f0bb188c100e: 1 change(s), 0 failed".
  - The lf readings with round 6's script. No figure moved: charged 6,160,384; theta 32,768, gathered 1,935,360, activations 2,916,352; rule blocks 0.0700404 / 0.0471321 / 0.0143295 / 4.61358e-4 / 3.27393e-6 / 0 / 0. Why, INFERRED: every lf weight's update is an add one to one in it.
  - The suites, last line "4 suites: 4 passed (0 cached), 0 failed; 25.2 min of test time" (repository 343, circuit-properties 43, training_slowdown 569, verity-graph 234).
- r20261112-134929-a412, vy-nebius-cpu-7: the reverts.
  - With the `rows` term dropped from `may_repeat`, `test_rows_carried_..._left_out_as_in_the_iteration_that_read_them` fails.
  - With the zero branch dropped, `test_an_update_by_a_stated_zero_leaves_a_weight_thetas` fails.
  - Each is "1 failed, 23 passed"; last line "REVERT_DONE".
- Local:
  - lf's per-group reading at k = 65,536 and 8e6 is byte-identical before and after.
  - With `rows_status`'s clearing dropped, 7 tests fail.
  - `test_value_reading.py`'s three breaks still repeat pairs the reading charges: none is refused.

STOP conditions: neither was hit. No lf product the reading charged is left out, and nothing changes the circuit format or the converter.

Not checked:
- Programs beyond lf's tiny step and the by-hand loops. An optimizer whose update is not an add one to one in the weight would be read as rows, which errs low.
- An fma whose product term is a stated 0 (only a multiply folds).
- circuit-check (no circuit changes).

The TRN-139 comment is posted. Nothing of mine is running. #9170 is untouched at 1068f3c5c2.

