[2026-11-12T23:49Z] 7c8f63e4452d
WALKER, TRN-145, about 23:57Z. Head 102ad1d702 is pushed. It holds 2^-3 (`points.CENTRE`) and 1(b)'s test.

4. The compare-by-difference revert (RAN, r20261112-233543-0517 at 9ce133fd32). It replaces the np.unique block of rows() with pairwise `np.all(f[i] - f[j] == 0)` in float64. Its line:
REVERT compare-by-difference: test_work: 2 failed, 59 passed | failed: ['test_a_specific_operand_is_tested_at_the_point_too', 'test_rows_of_one_value_that_are_inf_at_the_point_are_refused_as_equal'] | CAUGHT
Every other revert in that run is CAUGHT, and the run is still going through the last three.

3. 1(b) now has its own test, test_a_nan_made_from_an_inf_is_refused. W⊙W·2^127·2^127 is +inf, and adding −1 times itself gives inf − inf, a NaN in every word, which reaches the charged operand's rows. It is refused with ("InfNanRowsStep_v1", 'w', nan 3 of 3). RAN locally (pytest -k): it passes. With rows()'s NaN test dropped it fails: the rows pass as equal rows (0, 1), not as a NaN. The nan-rule revert now expects it. Decision 8 gains the line "a NaN that reaches no claim's row touches no claim".

At 2^-3, RAN r20261112-233445-ccd0 at 9ce133fd32:
- lf's two loops at 2^-2, 2^-3 and 2^-4 each have 0 origins, 8 nonfinite nodes (the mask's) and 0 NaN nodes.
- Each has 23 claims, 0 refused, and N of 6,160,384 and 12,320,768.
- The lf dump is identical to the base.
- The count: circuit_properties 80 passed (79 plus the inf test), training_slowdown 577 passed, grader 28 passed and 3 skipped.
- No pass or pinned figure moved.

Suites, ratchet and ruff: RAN r20261112-233515-8b17. All 3 suites passed, the ratchet from 4520abe708b0 has 0 failed, and ruff is clean.

The same jobs are rerunning at 102ad1d702 for the record: r20261112-233844-46e6, r20261112-233904-3665 and r20261112-233931-95e0.

1/2. The criterion and the mask are in points' docstring. I named the mask there as "a causal mask's", not by node: node numbers of one capture would go stale. The 8 nodes are named in the PR body.

Beside `CENTRE` the line reads: completeness only, SiLU's exp operand goes as the centre squared and must stay below 88.7. I left 464/116/29/7 out of the tree because the brief's rule is that the tree holds no measured result. They are in the PR body and in spec decision 5. Say if you want them in the comment.

5. s-t20 at 2^-3 is r20261112-233958-dacc, instrumented for origins and NaN nodes. Expect it to run long: r20261112-222055-84cf (the old centre) is still in its reading after more than 1.5 hours, past its 1315 s build. I report when it ends; it will not be in by 02:00Z.

6. The draft opens at 102ad1d702 once its suites job is in, by 02:00Z.

