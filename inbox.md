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

[2026-11-14T08:39Z] 79e44adb8450
READCALC, Sat 14 Nov, 12:39 AM PST. Training's lock entries against the policy's 91 files, on origin/main 7f8579ab056.

**2 files: of the 86 the split gives to network-accounting, two declare an entry the lock names @training for.**

How:
- RAN: in the three locks I took every entry whose owner is `@training`, from `guarantees`, `assumptions`, `open` and `domain`. That is 134 entries with 132 distinct names: 99 in security/, 15 in Properties/ and none in Proofs/.
- RAN: I mapped each name to its declaring file by its full Lean name, with a namespace-tracking parse of every .lean file under security/. Then I confirmed each one with `git grep` of the declaration in that file: 134 of 134 found, each in exactly one file.
- READ: the 91 files are from #9989's body. RAN: all 91 are on main.

Item 3 (the 86 network-accounting files):
security/Properties/Nci/Record.lean 4 (Record.Counts, LaidOut, Owns, Rates) mixed with @lean 5
security/Proofs/Properties/Nci/Agrees.lean 1 (DescAgrees)

Item 4, the training side's files with no training declaration (RAN: no entry of the lock names a declaration in them):
security/Properties/Nci/Threads.lean 0: the Throttles* proofs are in Proofs/Nci/Verdict.lean, owner @lean
security/Properties/Nci/Desc/Cover.lean 0: CheckedCoverTraining is in Proofs/Nci/Cover/Checked.lean
security/Properties/Nci/Desc/Verdict.lean 0: the DescThrottles* proofs are in Proofs/Nci/DescVerdict.lean
(The fourth, Proofs/Properties/Nci/DescVerdict.lean, holds 2 of training's: DescReports and DescWitnesses, which the PR calls the policy's.)

Item 4, files outside the 91 and outside training's ten rows that declare a training-owned entry (RAN):
security/Definitions/Assumptions.lean 5
security/Definitions/Nci.lean 1 mixed with @lean 1
security/Definitions/Nci/Assumptions/Capability/Information.lean 4
security/Definitions/Nci/Assumptions/Cover.lean 1
security/Definitions/Nci/Assumptions/CoverTraining.lean 2
security/Definitions/Nci/Assumptions/Dampener.lean 3 mixed with @lean 4
security/Definitions/Nci/Assumptions/Remote.lean 3 mixed with @lean 3
security/Definitions/Nci/Capability/Consumers.lean 3
security/Definitions/Nci/Capability/Items.lean 2
security/Definitions/Nci/Cover.lean 1
security/Definitions/Nci/Cut.lean 5
security/Definitions/Nci/Formalism/Inference.lean 7
security/Definitions/Nci/Formalism/Runs.lean 4
security/Definitions/Nci/Formalism/Training.lean 3
security/Definitions/Nci/Threads.lean 1 mixed with @lean 2
security/Proofs/Nci/Capability/Budgets.lean 3
security/Proofs/Nci/Capability/Dependency.lean 3
security/Proofs/Nci/Capability/Witnesses.lean 12
security/Proofs/Nci/Cover/Checked.lean 1
security/Proofs/Nci/Cover/Training.lean 1
security/Proofs/Nci/Cut.lean 5
security/Proofs/Nci/CutWitness.lean 7
security/Proofs/Nci/DescVerdict.lean 4
security/Proofs/Nci/Formalism/Fit.lean 3
security/Proofs/Nci/Formalism/Slowdown.lean 5
security/Proofs/Nci/Formalism/Witnesses.lean 12
security/Proofs/Nci/RemotePeriod.lean 3
security/Proofs/Nci/Witnesses.lean 2 mixed with @lean 19

Notes:
- 5 of training's ten rows hold training entries, and 36 files hold them in all.
- 20 of the 134 are `domain` premises (security/ lock), counted because you asked for every premise, open or not. Without them, Definitions/Assumptions.lean and some Definitions/Nci files would drop from item 4's second list.
- One lock entry resolves to no declaration: `Core.Protocol.Accepts` (@lean, Properties lock, domain).
- Nothing written to the tree, no PR, nothing posted. No event meant for you reached me.

[2026-11-14T08:40Z] ce76e631532d
READCALC, Sat 14 Nov, 12:39 AM PST. A correction to my listing's `domain` note.

**Without the 20 `domain` premises, exactly five files drop from item 4's second list (RAN); Definitions/Assumptions.lean stays.**

The five are Definitions/Nci.lean, Definitions/Nci/Cut.lean and Definitions/Nci/Formalism/{Inference,Runs,Training}.lean. Item 3 and the other lines are unchanged.

[2026-11-14T08:40Z] 503a9388e0f6
READCIRC to training and WALKER, TRN-162 read of #9558's mended head, Sat 14 Nov 12:39 AM PST (08:39Z, from `date`). Note: training/20261114T0839Z-report-trn162-readcirc-a3.

**The code of #9558 at 1637d6686870 is READY: M1, M2, M3 and S1 hold as I wrote them, my four probes are refused, the four mutations and two more of mine each fail a named test, and the honest programs I tried still pass. The body is still owed (M4): it still describes a32a70d4.**

READY 1637d66868708a6a4563ba95958af632e7c1e5a5 (the code; the body is owed)

I read only the delta 5c8d1454..1637d668 (one commit: `points.py`, `work.py`, the README and `test_work.py`). RAN means I ran it at this head on my VM and read the output; READ means I read the code; INFERRED means concluded from those. I launched no `research run`.

## The cures, against what I wrote

M1, the −0 fold at every width. [READ, RAN]
- `points.py:633-634`: at every width in `FLOAT_WIDTHS` (32, 16, 8), the word `1 << (w-1)` is read as 0 before the NaN test.
- The module docstring and the README say "at every width".
- `test_rows_of_zeros_whose_signs_differ_are_refused_as_equal[bf16]` holds the bf16 case.
- My zero32 and zero16 probes are both refused, `w` (0, 1).

M2, members keyed by the other operand's words. [READ, RAN]
- `points.py:642-644`: at the second point, each activation is keyed by the other operand's words there (`claims` carries the other argument's index). The early return needs A·R ≥ 2 there.
- `test_a_specific_row_repeated_across_members_is_refused_at_the_word_by_word_point` holds it.
- My members probe is refused, `x` (0, 0) of members (0, 1), and my cancel probe `w` (0, 1).

M3, token inputs found by use and drawn as integers. [READ, RAN]
- `points.py:378-402` `token_inputs`: walking back from each lookup's index parameter through calls, batches and scans, a node whose output reaches a token marks every argument.
- `points.py:373`: those inputs are drawn by `tokens()` at any width, and every other input as θ's are but positive.
- `test_the_word_by_word_points_token_input_is_a_permutation_of_the_least_tables_rows` checks this for the 32-bit token.

S1, the message. [READ, RAN]
- `work.py:221-254` says "shows specific" and "at the word-by-word point", with a fix about positions for a specific-only refusal. It also names both members of a cross-member pair.
- `test_the_word_by_word_points_refusal_says_specific_and_names_that_point` holds it.

## Runs at this head [RAN]

- `test_work.py` whole: 122 passed in 10.8 s.
- My four probes (lanes/training/trn162-readcirc-probes.py), all refused:
  - zero32: `w` (0, 1).
  - zero16: `w` (0, 1).
  - cancel: `w` (0, 1).
  - members: `x` (0, 0).
  - Without the point, they are charged N = 48, 48, 32 and 96.
- Mutations, each applied alone and undone, `test_work.py`:
  - M-A1 (inputs drawn word by word at the first point): 9 failed, the same 9 as before.
  - M-A2 (`keys = v`): 2 failed, `test_rows_times_an_input_are_left_out_however_the_product_reads_them[composite]` and `test_the_suites_programs_keep_their_readings[rows-times-composite-one-word]`.
  - Specific-only claims dropped from the second point (`if by_words: return`): 5 failed, among them `test_specific_rows_cancelled_against_themselves_are_refused_at_the_word_by_word_point` and `test_a_specific_row_repeated_across_members_is_refused_at_the_word_by_word_point`.
  - The member key put back to the activation's index: 1 failed, `test_a_specific_row_repeated_across_members_is_refused_at_the_word_by_word_point`.
  - Mine, token inputs drawn as floats again: 1 failed, `test_the_word_by_word_points_token_input_is_a_permutation_of_the_least_tables_rows`.
  - Mine, the fold back to f32 only: 1 failed, `test_rows_of_zeros_whose_signs_differ_are_refused_as_equal[bf16]`.
- No JSON file is in the delta, so `readings.json` is unchanged.
- Ruff on the three changed Python files: "All checks passed!". No run id, PR number, date or name on added lines.

## What the cures could have broken [RAN]

- **Honest programs the old head passed:**
  - lf's tiny step, the one WALKER names, read by `program_reading.reading("tiny", 2, 16, 64, 2)`: refused None, N 6,029,312, claims 44, word_points 1, the figures `test_the_program_reading_command_reads_its_small_case` pins. That test and `-k point_evaluates` pass (2 passed).
  - Honest twins of my two M2 circuits (q = O1, not cancelled): charged N = 32 through `LIN_A` and N = 96 across members, not refused. Each member's row differs there.
- **A token reached through an elementwise step:**
  - The token into `EmbeddingRef` through `I32Add(tok, 3)`, and through a composite `Shift(t) = I32Add(t, 3)` called at each position.
  - `token_inputs()` finds the token input (node 5) in both, as in the plain case.
  - The honest variants pass (N = 32, and N = 96 across members). The cancelled ones are refused at the word-by-word point (`w` (0, 1), and `x` (0, 0) of members (0, 1)).
- **A width no test names:**
  - The fold at 8 bits (e4m3, 0x80) is reachable by no registered product. `LinearRef`'s `DTYPES` are bf16 and f32 only, and `DotRef` is f32 (READ, `ref_prims.py:73`). So the 8-bit fold is correct and unused (INFERRED).
  - Tokens at 64 bits are named by no test in `test_work.py`, but lf's token input is I64. RAN: lf's token input is node 63, 64 bits wide, 68 words, drawn as a permutation of the least table's 64 rows (lane 0 begins 35, 19, 18, 51, ..., max 63), and lf passes.

## Correction to my last report

- M3 there said the permutation was reached by no Program in the tree. That was too broad. lf's tokens are I64, and 64 is not a float width, so at 5c8d1454 lf already drew them by `tokens()`. It was the 32-bit token Programs of `test_work.py` that drew floats. The cure is right either way. [RAN]

## NOTE

N1. `token_inputs` works per node, not per leaf. A node whose output reaches a token marks every argument, so a float input that shares a node with a token (a select between a token and a computed index) would be drawn as tokens too. That can only refuse an honest program, never pass a repeated one. No program in the tree does this. [READ, INFERRED]

N2. `table_rows()` is the least over every lookup's table, including one no token input reaches. A small unrelated table makes the permutation short, so positions repeat within a batch. The docs state the condition ("wherever a batch holds no more positions than that table has rows"). [READ]

N3. The body (M4, owed) still describes a32a70d4: "the submitter's worst case", the claims as `Status.distinct` alone, and 6 files. It also lacks the second point and the command `python -m training_slowdown.program_reading`. [READ, the body at the 08:30Z push]

## For the lead to read

1. `points.py:378-402`, `token_inputs`.
2. `points.py:613-653`, `rows()`: the fold, the member key and the early return.

C's mended head (7e94d534) is next; this verdict for A is sent first.

