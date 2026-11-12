[2026-11-12T17:47Z] 51c93e4384da
TRN-139 round 4b, PRICER. Head b23d961b3164d1ad6cc812a0af5925e230ccd73b, a merge of origin/main (4520abe708, which holds #9170) into 0d54ed51ff, with no conflict. #9315 is a draft and unmarked; its body is updated. A merge-tree with main's newer head 15662e7c40 is clean (RAN); I did not merge again.

1. One premise in work.py's docstring and the README, your sentence fitted: "a weight combined with its own words still counts as a weight with distinct rows, though the combination may make its rows equal whatever the weight: cancelled against itself (M plus -1 times M; W times v plus -1 times v ...), or times its own rows in another order (M times swap(M), whose two rows are equal for every M), so the looked-up rows plus such an array are charged as weights though they repeat with the tokens".
2. test_a_weight_times_its_own_rows_in_another_order_still_counts_as_a_weight[carried|own]: swap and same order are each charged (T-1) PN K N as (True, True); its docstring says the work is the first iteration's PN K N, as _rows_plus("weight")'s. The swap fact adds no revert condition (no rule holds it), but "the third case removed" now fails six tests, not five: the swap test's carried form is the sixth.
3. The body's "Not checked" names both families, each with its charge (96) and its work: the cancel's is the looked-up rows exactly, which alone are charged 0; the swap's is the first iteration's 24. One sentence says a random-point evaluation would refuse both and is not built.
4. The TRN-143 paragraph follows.

Runs at b23d961b31:
- RAN r20261112-172751-b37d: `ratchet gate 6d95fcc68dd3..b23d961b3164: 1 change(s), 0 failed`; lf `never_above ok` and `charged 6160384` at every k from 65,536 to 8e8, left_out unchanged; ruff `All checks passed!`; `4 suites: 4 passed (0 cached), 0 failed`.
- RAN r20261112-170515-462f: 16 of 16 reverts fail a test; the last, the third case removed, is `6 failed, 44 passed`.
- RAN locally at b23d961b31: lf's per-group dump, its sets sorted, is byte-identical to 9ff0d233a1's. Unsorted dumps differ only in frozenset order, which hash seeds change between processes. 50 tests pass. Your swap probe matches your output line for line.
- Superseded: r20261112-170449-654a ended rc 1. Its suites passed, but "the tests changed 1 files: work.py". Its revert job ran beside it on the same node with --cwd source and edited work.py under it. That was my slip: a job that writes into the tree takes --cwd clone, as the writing-runs skill says, and my revert job now does. Run B's own results stand, since run A writes nothing.

Not checked: other Programs; the F16, E4m3 and E2m1 roundings in a test; f(b, b); an fma whose product term is a stated 0; circuit-check (no circuit change); any figure marked estimated in the TRN-143 paragraph.

TRN-143 sizing (PRICER; no code). Refuse rows made equal by an identity in a weight's own words, by a random-point evaluation.

Evaluator and level:
- The evaluator is `verity.circuits.evaluation.evaluate_batch`, node by node: its numpy kernels where registered, the plain-integer reference otherwise. `program_walk` walks this way but needs Boolean types, and lf's step is a word Program (its parameters are Value<32>), so it is a new walker of about the same shape.
- The level is each specialized Definition's body once, as `Reader.summary` reads it. Its parameters get random finite generic words (θ's, the inputs', a carry's drawn fresh), so a scan body is checked for one iteration from random carried words, which covers by induction any identity over its parameters.
- The draw must keep the aliasing a call site gives: `f(b, b)`, or M and swap(M) passed as two arguments. The summary's context doesn't carry that today; words drawn per root source and passed through views would. The flattened Program has the aliasing for free but costs a whole step.

Cost:
- lf tiny step (RAN, a count over its Program): \(1.51 \times 10^{10}\) words flattened, \(2.25 \times 10^{6}\) counted once a body (26 Definitions, 3,165 body nodes).
- RAN once on my CPU VM: `evaluate_batch` of the whole flattened step at random words took 83 s and 1.7 GiB. I did not split it into kernel time and reference time. A pass a body is estimated at seconds even on the reference, from the 6,700 times fewer words.
- `s-t20` (35.7M parameters, context 1,024), estimated and not run:
  - flattened: about 24,000 times lf's products a sequence, so days to weeks with the reference;
  - a pass a body: about \(10^{8}\) to \(10^{9}\) words (lf's 20 a parameter, more for the context), minutes and a few GiB, but only if the products and attention run on numpy kernels; the reference alone would be hours.

Refuses: any array the reading claims distinct whose rows come out equal at the point:
- M + (−1)·M, W ⊙ (v + (−1)·v) and M ⊙ swap(M), in every spelling;
- the same identities wherever the third case or the no-input case admits them.

Does not refuse:
- W + z + (−1)·z: W's rows stay distinct; what it leans on is θ′ for θ, another premise;
- a one-word multiplier that is 0, or inputs fitted to the weights (z = −g): the point draws inputs nonzero and generic, and the premise still takes these.

Could refuse wrongly:
- rows equal by chance at the point, likely only for narrow words: a 1- or 4-bit row, an index, a one-word row, or rows after a rounding to e2m1 or e4m3. The fix is at least 64 random bits a row (several points, or claims only on rows that wide), with each word drawn at its own width;
- a body evaluated outside its domain (a NaN or a division by 0 that the Program's real inputs never reach): draw finite floats, and count a NaN as unequal.

Where it runs: inside `WK.read`, on every read and memoized with `summary`. A separate command a submitter may skip would be a rule-following path that is opt-in. A refusal names the Definition, the node and the two equal rows.

Size (estimated): about 150 to 250 lines (the word walker, the draw with aliasing, the comparison). Tests:
- the cancel and swap tests flip from charged to refused;
- a narrow-word array is not refused;
- the lf per-group dump is unchanged;
- a count of words evaluated on lf is held below a stated number (no wall clock).

