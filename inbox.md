[2026-11-12T21:04Z] 230dbbb4d60c
WALKER (TRN-145), 2 PM report. Head 676444758c (pushed). A finding first, which I stopped on.

FINDING: the NaN rule makes the point's test vacuous on part of lf (RAN r20261112-204453-a0e8, at 676444758c). On lf's loops, the point's values drive many claimed weight rows to NaN (likely overflow from θ sampled at four binades about 1 through the step and AdamW's update; inferred, not traced). A NaN row equals no row (spec decision 8), so those rows are never compared:
- 2 iterations: 23 claims, 8 with NaN rows, 3 with a NaN in every row; 1,721 of 2,240 rows compared.
- 3 iterations: 23 claims, 14 with NaN rows, 8 with a NaN in every row; 1,006 of 2,240 rows compared.
- Smallest case: loop(step, 3), TrainStep's node 147, operand 'w' (K 64): all 64 rows are NaN in every word, so the claim passes having compared nothing. Nodes 146, 242, 253, 271, 308, 335 and 346 likewise have a NaN in every row.
- Rows compared as plain words find no equal rows among the non-NaN rows of any claim. The all-NaN rows are equal as words (63 at node 147), which is why the revert of the NaN rule errors the lf point test.
The spec is silent on this, and the README now says each such claim "is tested at a point". Deciding it is yours. Options I see, not chosen: sample θ nearer the weights' scale so the step stays finite; count a claim with NaN rows as unevaluated (refused, like the scan mask); or compare only rows without NaN and require enough of them. Nothing is changed in code; I'm going on with the rest.

ITEM 3, the count of claims past a scan the point doesn't run whole:
- lf, loops of 2 and 3: 0 of 23 each (RAN r20261112-204453-a0e8, which records each claim's evaluated state).
- Every Program both suites read: RAN count pending (r20261112-194443-4b42, which reruns the plugin count, still in the queue). READ bound: Walk.run raises RepeatedRows on any such claim. Only test_work.py, test_value_reading.py and test_program_reading.py read Programs, and only six refusal tests catch the error. All suites pass at 676444758c (RAN r20261112-194406-cf52). So the count is 1, the constructed Outer test, which is refused naming OuterStep_v1, 1 of 4.
- No library entry is refused: none goes through the reading (READ: library.py reads entries' traced steps by calculator.slowdown).

Other node checks at 676444758c:
- Gate, RAN r20261112-194406-cf52: ratchet gate 0 failed against main's merge-base and against b23d961b3164; ruff clean; circuit_properties 78 passed, grader 28 passed with 3 skipped, training_slowdown 577 passed.
- The first gate run (r20261112-185122-987b at 759d5750ae) failed the ratchet gate on the retired term "draw" (22 sites PRICER's points.py and tests added). Fixed in 676444758c: points.draw is now points.sample, and the prose says "sampled". The test is now test_the_point_is_sampled_once_a_program_and_read_at_every_k. Wording only.
- Reverts, RAN r20261112-194513-1c72: 8 of 10 removals fail the test meant for them (P, the NaN rule, the claim filter's reason part, the scan mask, the memo, the seed, dot, a call's output sampled fresh, which fails the cross-call test).
  - Repeated inputs: fails 4 tests (cross-call, difference, swap twice) but not the self-cancel test. RAN locally: under that revert, SelfCancel's first iteration samples two tokens that both index table row 0 (mod V = 6), so its rows are equal by coincidence.
  - OPEN: the claim filter's status part (not specific, gathered or data) is caught by no test. lf's claim count stays 23 without it. product_status can mark an output specific while keeping its base's distinct, so the filter can matter, but no test holds a charged operand that is both distinct and specific. Whether a specific claim with equal rows should be refused is design (the spec leaves specific claims out on purpose), so I've built no test.
- lf dump against b23d961b3164: in r20261112-194443-4b42 (queued). The first try failed on my job's environment, not the code.
- s-t20, r20261112-185304-7c4e (at 759d5750ae, the same code with old names): running. Building the step alone took 1,114 s. The step is 3.28e11 primitive words flattened, so 2 iterations are about 6.6e11 words.
- Correction to my earlier message: the first-round runs had finished by 19:25Z (the status I read lagged). The count there failed on my environment handling, so it was relaunched.

