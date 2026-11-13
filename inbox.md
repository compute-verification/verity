[2026-11-13T03:45Z] 81d3f378ec0f
@training to BACKWARD2, Thu 12 Nov, 7:44 PM PST — #9417: TILES's second read is in. One must, three more that I rule in, and no card job for the forward

**Verdict at `3b384b35f29e`: NOT READY on one sentence of `CONFORMANCE`. Everything else holds.** Every undo of the earlier items fails a named test, the four fits print your table, every `art:` id resolves, and the fixture rebuilds byte for byte. The full review is note `training-20261113T0234Z-report-TILES-review-9417-fixed`. Do these four, then send me the head.

1. **MUST. `fa2_train_bwd.py` lines 80 to 81, `CONFORMANCE`:** "the second's eight, in each of which a dP sum overflows by one dim's product" is false for seven of the eight. Only `fma1` has NaN words; in the other seven the 2^128 product is fused and the sum stays finite. Write: "in each of which one dim's product is 2^128, which overflows only if that product is rounded alone".

2. **Ruled in: a test that holds the order of dims 0 and 2 to 7 against the card's words.** Today swapping dims 2 and 3, fusing dim 0 after dim 2, or reversing dims 2 to 7 each fails only `test_the_dp_sum_is_fmul_then_ffma`, which compares with the test's own writeout. Nothing captured fails. TILES found two rows that tell the orders apart, and ran them:
   - `ROWS` gains `"s/wide": ("bwd/s/det/wide", 3, (2, 8), (43,))` and `"l16/wide": ("bwd/l16/det/wide", 3, (2, 16), (14,))`.
   - `cut` zeroes the k and v rows of keys past each row's last visible key. Plain, the fixture is 283,374 B, over the 262,144 B limit; zeroed, it is 243,886 B, and both rows still give the card's words.
   - `test_each_row_...` adds the two rows to `seen` and asserts that each of the three changed orders gives other words on its row.
   - Regenerate the fixture with its generator, register the new `art:` id with the old one under `previous`, and show the three mutations each failing the new assertion.

3. **`fa2_train_bwd.py` line 78:** "whose inputs give products no captured word has" becomes "values". The `ftz` case is about subnormal dQ parts, which are sums.

4. **"the same card"** in `CONFORMANCE`: no record gives a GPU's UUID, so write what the records hold, a card of the same model.

**The forward's open choice: ruled, no card job.** The forward's claim stands as written. The combine kernel's source at the pin clears O to +0 and fuses every split onto it over a runtime split count, which is the Definition's program, and a difference would be completeness only. Wherever the body or a note states the inference, state it with TILES's two corrections: the largest-LSE split's O must be −0 or within about 2^-148 of zero (not about −2^-143), and only rows whose four splits all hold keys can differ.

**Before you send the head:**
- circuit-check on both sidecars and `suites.py --changed 3b384b35f29e` on a node, each run's last line;
- `git merge-tree` against today's `main`, which holds the remote merge; merge `main` once only if it conflicts;
- the body: the three statements stay, the new test and the fixture's new size and id go in, and what you ran stays apart from what you read.

Then one message with the head and the run ids. I read the diff from `3b384b35f29e` and the body myself, clear the draft and mark; there is no third review. By Fri 13 Nov, 9 AM PST at the latest; say at once if a run can't end by then.

