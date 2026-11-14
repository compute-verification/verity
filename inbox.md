[2026-11-14T08:39Z] 3dff43f60aa3
WALKER (session c17bbff0, on A), Sat 14 Nov, 12:39 AM PST.

**An honest reading is refused at the word-by-word point, so I stop on A as you ruled: `test_value_reading.py`'s torch attention step is refused at `Derived[TrainStep]_v1` node 86, operand `x`, rows 0 and 4. Those two rows are all +0 words, the first position of each of the batch's two sequences. The refusal was already there at 5c8d1454, so the second point causes it, not the new member key. Neither PR can be marked until you rule.**

Three sessions of me are live on this VM since about 08:31Z: c17bbff0 (this one, A), 4d808059 (C) and 6d747131 (unknown to me; it has written nothing in /tmp/trn145).

The refusal:
- RAN, at A's head 1637d668 on the VM, `test_value_reading.py -k "distinct_pairs or breaks_the_premise"`: 3 failed, 6 passed.
  - `test_what_the_circuit_alone_charges_reads_distinct_pairs[sdpa]`: `Derived[TrainStep]_v1{SRC={6e2fa37a40d3}}` node 86, `x`, rows (0, 4).
  - `...[relu]`: `Derived[TrainStep]_v1{SRC={a5c37c433e8d}}` node 98, `x`, rows (0, 4).
  - `test_a_batch_that_breaks_the_premise_repeats_pairs_the_reading_charges[sdpa-prefixes in proportion]`: node 86, `x`, rows (0, 4).
- RAN, the sdpa case instrumented: node 86 is a batched f32 `LinearRef_v1{K=4,N=4}`. Its `x` has 8 rows of 4 words in one activation. At the word-by-word point, rows 0 and 4 are `0x0` in every word, and the other six rows are distinct and nonzero.
- RAN: the token input is found by use (node 33, 64 bits, 30 words), drawn as a permutation of the table's 8 rows. So M3 works here, and no member key is involved (A = 1).
- INFERRED: rows 0 and 4 are the first position of each sequence (2 rows × 4 positions). There a softmax's backward is 0 for every input: dS = P ⊙ (dP − Σ P·dP), and P = [1] at a row with one key. So the rows are equal by an identity of an honest circuit. `P_BATCH` itself excludes each row's first position, but the second point compares every row. The specific rule calls `x` specific, the second point refuses it, and the point can't tell a first position from any other.
- RAN: the quick tier at 5c8d1454, r20261114-080235-0f74 (the other session's), failed the same 3 tests (rc 1), so 093eb4b07b7's second point causes this. The honest small model passes at both heads.

Your ruling is needed: whether to exclude each sequence's first position from the second point's comparison (it would need the positions, which the point doesn't have now), to leave such an operand uncharged, or to do something else.

A:
- #9558's head is still 1637d66868708a6a4563ba95958af632e7c1e5a5, as sent at 12:30 AM. The reader's four probes are refused there, and the four mutations fail by name.
- Held, on the backup branch cursor/trn145-a-fix-wip-eb6e only: 4c96f3db79d6, one commit that fixes the quick tier's other failure, which is A's own. `program_reading.py:86` swallowed `(OSError, StopIteration)`, failing `tests/lint/test_repo_silent_excepts.py`. It now reads VmHWM with `next(..., None)`. RAN: that lint passes 3 of 3, and the command's small-case test passes. I push it to #9558 with the fix you rule, as one push.
- The quick tier at 1637d668, r20261114-083224-ce52, is waiting for a slot. It will fail on the two known causes. I leave it to show whether anything else fails.
- A's body (M4) is not written yet. It follows the ruling's head, with the command's small case at 1637d668 (RAN on the VM: refused null, N 6,029,312, claims 44, pass 1.6 s, pass peak 0.84 GiB), and with the memory section's harness figures taken out under ruling 4.

C (session 4d808059's report): its head is 7e94d534f322 (A's 1637d668 merged). Its quick tier is r20261114-083506-23be and its s-t20 reading r20261114-083504-83ab. 9b8b was cancelled at 08:34Z. C holds the same second point, so its quick tier fails the same three tests.

Events not addressed to WALKER: none this turn.

