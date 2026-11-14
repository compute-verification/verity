[2026-11-14T08:04Z] 8281857ecfa8
READCIRC to training and WALKER, TRN-162 read of #9558's fix head, Sat 14 Nov 12:02 AM PST (08:02Z, from `date`). Note: training/20261114T0802Z-report-trn162-readcirc-a2.

**#9558 at 5c8d1454d380 is NOT READY, with 4 must-fixes. M1 holds only at f32, so bf16 rows of zeros with differing signs are still charged. M2's second point misses a specific operand's rows repeated across batch members, and it never draws the permuted tokens the docs and the ruling describe. The body still describes a32a70d4. M3, M4's runner and test, S2, S3, N1 and N2 hold, and each of the three mutations fails tests by name.**

NOT READY 4 5c8d1454d380a9d4f049a835443c934d8251a43b

I read only the delta a32a70d4..5c8d1454 (3 commits, 7 files, +445/-101), held to my report and to the lead's rulings. RAN means I ran it at the head on my VM and read the output; READ means I read the code or a file. I launched no `research run`.

## MUST FIX

M1. The −0 fold is f32-only, while ruling 1 says every width the point compares. [RAN]
- `benchmarks/circuit_properties/circuit_properties/points.py:601-602` folds only when `_operand_width(nd, i) == 32`, but `rows()` compares rows at any width.
- My probe at the head: W' = (W + (-1)·W) ⊙ S, with S a stated ±1 (row i negated at word i-1), rounded to bf16 (`F32ToBf16Rn`, which keeps rows distinct). Then u16·W'ᵀ through `LinearRef` with `DT="bf16"`.
  - It is charged, N = 48, with no refusal.
  - The same circuit at f32 is refused (`w`, rows 0 and 1).
- Fix: fold the sign-only word at each width before the NaN test, `v = np.where(v == np.uint64(1 << (w - 1)), np.uint64(0), v)` with w the operand's width (f32 0x80000000, bf16 and f16 0x8000, e4m3 0x80). Add the bf16 probe as a test, refused.
- The docstring's "an f32 row's two zeros one word" (`points.py:36-37`, README:73) then reads "at every width".

M2. A specific-only operand repeated across batch members is charged, and neither point compares it. [RAN]
- My circuit: q = O1 + (-1)·O1 from `_attention_loop`'s O1 = P·V1ᵀ, so q is specific only and 0 for every input. A composite `PerRow(q, w) = LIN(q, w)` is batched over the PN positions, with q mapped (axis 0) and a carried W shared (axis None).
- At the head it is charged, N = 96 = (T-1)·PN·N·K, with no refusal. Every member reads the pairs (0, W[n,k]), so the work is repeated PN times.
- The cause is `points.py:596-597` and `:608`:
  - Rows are compared within one activation only, keyed by the activation's index.
  - Here each activation holds one row of q, so `W // K < 2` returns before any comparison.
- At the first point, keying by member is needed, since one token makes every member equal. At the second point it is the hole.
- Fix (RAN as a scratch patch):
  - Where `by_words`, require `A·R ≥ 2` rather than `R ≥ 2`, and key each activation by the other operand's words at the second point's lanes instead of by its index. Pass `args[1 - i][0]` into `rows()` at `:474`. Then `member = np.unique(other_lanes_by_activation, axis=0, return_inverse=True)[1]`.
  - So two members' rows are compared exactly where the other operand is the same in both.
  - With the patch, my circuit is refused (`x`, rows (0, 0) of two members). `test_work.py` passes 118 of 118, and `training_slowdown`'s `-k "point_evaluates or command_reads_its_small_case"` passes 2 of 2.
  - Add my circuit as a test, refused.

M3. The second point never draws the permuted tokens that ruling 2 and the docs describe. [RAN, READ]
- Tokens are I32, and I32 is 32 bits wide. The type system has widths only (`P.I32 == P.F32` is True). So in `points.py:371-373` the token input takes the `w in FLOAT_WIDTHS` branch and is drawn as positive f32 words. `tokens()` (`:101-108`) is reached only at widths outside {32, 16, 8}.
- RAN, instrumented at the head on the author's `_specific_alone("cancel")`: `tokens()` is called 0 times, and the token input's second-point words are `0x3e18a847...`, which the gather reads modulo 4096.
- So "each lane a permutation of the least table's rows" and "meets `P_BATCH` by that construction" are false for every Program in the tree. That is in the module docstring (`points.py:25-31`), `inputs()`'s docstring, README:58-60 and 71-72, and `work.py:94-98`.
- Soundness is not hurt (INFERRED): a chance collision of two tokens only adds equal rows, so it can refuse an honest circuit but never pass a repeated one. The cost is completeness, and docs that claim what the code doesn't do.
- `test_the_word_by_word_points_tokens_are_distinct_rows_of_every_table` holds `tokens()` alone, so it does not catch this.
- Fix: find token inputs by use, the inputs that reach a lookup's token argument (`R.lookup(f).token`, which `table_rows()` already walks). Draw those with `tokens()` at any width, and draw every other input as a float. Then add a test that the second point's token input in `_specific_alone` is a permutation of the least table's rows.

M4. The body still describes a32a70d4. [READ, at the body as of the 07:46Z push]
- Sentences false at this head:
  - "now checks ... each charged operand whose rows its rules show distinct".
  - "Every input's words are one word repeated, the submitter's worst case."
  - "The claims are every charged operand with `Status.distinct` (`points.claimed`)."
  - "6 files, 1,075 lines in and 431 out" (now 7 files, +1,460/-472 against `main` 3d62f5507b01).
  - lf's "23 claims" (the point test now asserts 46: 23 at each point).
  - "`points.py` is new: about 590 lines" (now 668).
- What the body lacks:
  - The command `python -m training_slowdown.program_reading` for its s-t20, memory and smaller-shape figures (ruling 4).
  - The second point, and the one line saying one walk and why (ruling 2).

## SHOULD FIX

S1. The refusal message is wrong for a specific-only operand. [RAN]
- `work.py:244` says "whose rows the reading shows distinct" for every refusal. Refusing my M2 probe at the head prints "operand 'w', whose rows the reading shows distinct, holds rows 0 and 1 equal at the point; fix: compute it from a weight one to one ...".
- That operand is specific and not distinct, the point is the second one, and the fix text is about weights.
- Fix: carry `by_words` into `RepeatedRows` and say "whose rows the reading shows specific, holds rows 0 and 1 equal at the word-by-word point", with a fix that fits (compute it so that its positions differ, or let it read the inputs as gathered).

## NOTE

N1. Determinism holds: the second point's draws are a function of the digest and the label alone. [READ, RAN]
- READ: `word_seed` = SHA-512(`b"circuit_properties/points/words\0"` ‖ digest) (`points.py:340-341`). Inputs are drawn in body-node order, and `table_rows()` takes the least over a walk, so no order of a set or dict reaches a draw.
- RAN: the second point's draws for `_specific_alone("cancel")` hash to sha256 `c1740083ba721f39` in two processes at `PYTHONHASHSEED` 1 and 98765, with lanes 2 and P 1.

N2. `program_reading.py` reads `time.monotonic` for the seconds it prints. The test calls `PR.main` and asserts only that those keys exist, never their values. [READ]

N3. What the delta adds to suite time, at the head on the VM. [RAN]
- `test_work.py` whole: 118 passed in 11.6 s.
- The point test's call: 6.2 s.
- The new `slow` test of the runner's small case: 6.6 s.

## The rulings, line by line

1. M1. [RAN, READ]
   - My f32 probe is refused at the head (`w`, (0, 1)), and the author's `test_rows_of_zeros_whose_signs_differ_are_refused_as_equal[signs|ones]` holds it.
   - The docstring states the comparison of words with the two zeros one, but for f32 only. Every width is not done (MUST FIX M1).
2. M2. [RAN, READ]
   - One walk: the second point is P more lanes in the same pass (`Walk.lanes`, `points.py:349-350`). The why is not stated (M4).
   - Seeded from one SHA-512 of the digest under its own label (N1).
   - The docstring says how an input at a width no float has is drawn, but tokens never reach that branch (M3).
   - My M2 probe is refused (`w`, (0, 1)), and the author's test holds it.
   - The honest small model passes, and so does `training_slowdown -k point_evaluates` (RAN: 23 specific-only claims at the second point, 23 distinct at the first).
   - `readings.json` is unchanged: no JSON file is in the delta.
   - `test_work.py` passes whole, 118 of 118.
   - NaN and unrun-scan refusals at the second point: `test_the_word_by_word_point_refuses_a_nan_and_an_unrun_scan_as_the_one_word_point_does` passes.
   - The `P_BATCH` sentence is in README:58-60, `work.py:94-98` and `points.py:27-31`, but it is false as built (M3).
   - "every error of a reading the point passes ... is less work" (`work.py:62-64`) is false while M1 and M2 stand.
   - Mutations, each applied alone and undone, `test_work.py`:
     - M-A1 (inputs other than θ drawn word by word at the first point): 9 failed, the same 9 as at a32a70d4.
     - M-A2 (`keys = v`): 2 failed, `test_rows_times_an_input_are_left_out_however_the_product_reads_them[composite]` and `test_the_suites_programs_keep_their_readings[rows-times-composite-one-word]`.
     - New (specific-only claims dropped from the second point: `if by_words: return` after the claim count): 2 failed, `test_specific_rows_cancelled_against_themselves_are_refused_at_the_word_by_word_point` and `test_the_word_by_word_point_refuses_a_nan_and_an_unrun_scan_as_the_one_word_point_does`.
3. M3. `test_what_the_point_never_meets_stays_a_premise` adds W ⊙ (z0 + max(z0, -z0)) + z, charged, with `w.N == (T-1)·K·N` asserted. It passes. [RAN, READ]
4. M4. `training_slowdown/program_reading.py`, `python -m training_slowdown.program_reading` with its parameters, is in the README's table and commands. `test_the_program_reading_command_reads_its_small_case` (slow) holds the tiny case: refused None, N 6,029,312, claims 44. It passes. The body does not name the command yet (M4). [RAN, READ]
5. S2, S3, N1 and N2 all hold. [READ, RAN]
   - S2: "worst case" is gone from both packages, and the docs say one positive word repeated, as a submitter may choose.
   - S3: `monkeypatch.setattr(PT.Walk, "run", ...)`.
   - N1's caveat is a premise in `points.py:14-16`, README and `work.py`.
   - N2: `None if e.rows is None`.
6. Ruff on the five changed Python files: "All checks passed!". No run id, PR number, date or name on added lines. [RAN]

## Trying to break M2 [RAN]

I built three circuits. One broke it, and two did not.
- Repeated across members: charged with no refusal. This is M2.
- An identity for in-range tokens only, since the second point's tokens are out of range: no circuit possible.
  - A token reaches arithmetic only through a gather, which reads `tok mod rows`, the same row a served in-range token reads, or through `I32Le` and `I32Eq`, whose bits reach floats only through `SelectF32`.
  - `SelectF32` is not `INJECTIVE`, so its output is not specific, and the specific rule refuses an unstated argument (`work.py:1000-1009`).
- An identity for positive inputs only: the second point draws positive inputs, so it refuses such a circuit. An identity for negative inputs only is the ruled premise (M3's test).

## For the lead to read

1. `points.py:596-614`, `rows()`: the member key and the one-row early return (M2), and the per-width −0 fold (M1).
2. `points.py:354-375`, `inputs()`: the `FLOAT_WIDTHS` branch that 32-bit tokens take (M3).
3. `work.py:240-244`: the refusal message (S1).
4. `points.py:1-60`: the module docstring's second-point paragraph, against M3.

My scratch probes and patches are not in any branch: `points.py` was restored after each, and the worktree is clean but for two untracked probe files, which I delete after C's read.

