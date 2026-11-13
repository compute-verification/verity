[2026-11-13T09:53Z] 1ecca4d10b97
training, Fri 13 Nov, 1:53 AM PST. TRN-145, your 1:41 AM design. [read by me: #9676's `work.py` 230-277 and 820-982 at `3a94714cfb`]
**Go: start the rows (`masked_rows`) and the select (`unselected`) now. Four conditions, and one question on `softmax_rows`' column case to answer before you code that part.**

Conditions:
1. **No float fact is written twice.** Each run's verdict comes from what word-by-word calls: `_generic` on data and the run's constant (the add, the compare, the and), and `_shape` and `_finite` for a row. The rule's code names neither 0 nor -inf, and does not say what an add or a compare gives.
2. **Runs compare in one canonical form.** Two rows with the same constants, written as different parts, must compare equal. If they don't, the rule calls two equal rows different, which is the wrong answer in the direction that claims more. Test: a hand-built body in the form whose two rows hold equal constants under different part splits, equal on and off.
3. **Tests for the branches lf's step never reaches**, each a hand-built body in the form, equal on and off:
   - a row wholly masked, where the fold is a nonzero constant, `unselected` gives None and `_shape` gives None;
   - a mask constant that is finite and not 0.
   A causal mask reaches neither, so on/off on tiny and s would pass with either wrong.
4. **The reading says how it read.** It counts the softmax and select nodes the rule read and those that fell back. The suite's tiny test holds lf's step at zero fallbacks, so a builder change that leaves the form fails a test instead of reading for hours.

The question, before the column code:
- `softmax_rows`' second case builds `col = tuple(info[r][1][c] for r in seq)`: K words a column, for each of n columns. Does lf's backward take that case? If so, how does the rule decide "the columns differ pairwise, each with a finite word" without K words a column? If it falls back there, that node's time stays ctx², so say which and what you expect the trace to show.

PR body: the equivalence as one statement, then its conditions as a list, each naming the line that checks it. I read the code against that list. The flag is a constructor argument the tests use, not a CLI option.

Noted: d4d5 and 72e0 ended at their timeouts. 54ec at node 10,760 of 16,403. Send its end.

