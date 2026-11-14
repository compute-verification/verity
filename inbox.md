[2026-11-14T08:40Z] 61734b2388ec
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

