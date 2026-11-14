[2026-11-14T17:19Z] 1a3378bc1911

READCIRC to training and WALKER, TRN-162 read of the S1 and S2 cures, Sat 14 Nov 9:18 AM PST (17:18Z, from `date`). Note: training/20261114T1718Z-report-trn162-readcirc-a6c6.

## A (#9558): READY c0082a114ab5f6763e47b3b170337de284a5f2c0

**MUST FIX:** none. **SHOULD FIX:** none.

**NOTE**
- **Nothing else is in the range.** RAN: 91d8c6b7..c0082a11 is one commit, c0082a114ab. Its diff is `test_work.py`, 1 in and 1 out: the `anti` docstring line.
- **The docstring is true of its test.** READ: `_anti` zeroes member 0 of x and member 1 of W in a batch with `axes=(0, 0)`. x's zero row meets only member 0's W rows, which are not zero, and W's zero rows meet only member 1's x row, which is not zero. So no zero row of one meets a zero row of the other.
- RAN at the head: `test_work.py` 126 passed. Ruff is clean on `test_work.py`, and the edited line is 119 characters.
- **The body is true at the head.** READ, checked against the tree:
  - Line 53 reads "so a row, not zero, repeated a member against one shared operand is refused", which matches `points.py` 33 and `rows`.
  - Line 22 describes 1637d668's commit and is true of it.
  - The body names the head, lists c0082a114ab as "1 file, 1 in, 1 out; no code", and says `sdpa` and `relu` were refused at 1637d668.
  - RAN, the counts: 8 files, 1,678 in and 474 out against `main` 3d62f5507b01; nine commits on a32a70d4d7e9; six since 5c8d1454d38, 342 in and 126 out; `points.py` 731 lines. Each matches.
  - I didn't rerun the suites at this head; the body's 143/683 at c0082a1 is WALKER's (READ).

## C (#9730): READY 46b9519a24c249f1c448cf61824b61c9f3a80a28

**MUST FIX:** none. **SHOULD FIX:** none.

**NOTE**
- **The merge.** RAN:
  - 46b9519a merges c0082a11 into 203a62ad, and `git show --remerge-diff 46b9519a` prints 0 lines.
  - Its diff over 203a62ad is that one `anti` docstring line and nothing else; `work.py` is unchanged.
  - C's own diff over A's head (`git diff c0082a11 46b9519a`) is the same 4 files, 5,654 in and 48 out, every line identical to `git diff 91d8c6b7 203a62ad`.
- **The body is true at the head.** RAN or READ:
  - It names 46b9519a and the merge of c0082a114ab, and says the last merge brings one line with an empty remerge-diff (RAN).
  - "9 files" against `main` matches `git diff 3d62f5507b01 46b9519a` (RAN).
  - The cited `work.py` lines are unchanged since 9155c1dc, where I matched all 33 (READ).
  - I didn't rerun C's tests; the change is a docstring line identical to A's (INFERRED).

## For the lead to read

- Both are READY at their heads. The quick tiers (592d at A, 238b at C) and s-t20 (1aa5) are WALKER's to read; I read none.
- Before work I removed no timers or subscriptions, because I held none.
- Events I saw: WALKER's 8:44 AM message with the ranges, addressed to me, read as data. Launched: nothing. Posted: nothing. Worktree removed; the main checkout is clean on `main`.

