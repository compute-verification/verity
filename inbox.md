[2026-11-12T16:31Z] 20ad169fcd3e
TRN-139 round 4, PRICER. Head 698436104d on cursor/trn139-carried-premises-914d (#9315, draft, unmarked; #9170 untouched).

1. The site clause is gone (sx, sw = arg_status(...) again); `multiplier` returns `not o.reach or o.distinct or (o.fixed and one word)`. Docstring and README: "a product reads them as one row" removed; the multiply clause gains "or whose own rows are shown distinct (a weight times a weight)".
2. Tests (48): your table as `test_rows_times_an_input_are_left_out_however_the_product_reads_them[batch|calls|composite]` (0 each, one-word multiplier charged each); the scan test's docstring says u (*) g is a weight times a weight; revert job: "the third case removed" in, "the one-row operand removed" out.
3. Premise sentence in docstring and README as you wrote it; `test_a_weight_cancelled_against_itself_still_counts_as_a_weight`: rows alone 0, rows + (M + -1 M) charged (T-1)PN K N as (True, True), and W (*) (v + -1 v) with v registered and v carried (new `_scaled("carried-cancel")`), each charged. Cheap check for the family: the random-point evaluation (my earlier message); not built. It also refuses the swap fact I sent.
4. Runs at 698436104d:
- r20261112-161444-a434: `ratchet gate 87fec6c6491b..698436104d73: 1 change(s), 0 failed`; lf `never_above ok`, `charged 6160384` at every k 65,536..8e8, left_out as round 3; `4 suites: 4 passed (0 cached), 0 failed`.
- r20261112-161510-c176: 16 of 16 reverts fail a test; the third case removed: 5 failed, 43 passed.
- RAN locally: lf per-group dump byte-identical to 9ff0d233a1's.

The third case failed no fact you ran; it fails the swap fact (sent before this). No workaround made; the tree does not state it yet.

Not checked: other Programs; F16/E4m3/E2m1 roundings in a test; f(b, b); an fma with a stated-0 product term; circuit-check (no circuit change). TRN-142 read, not started.

