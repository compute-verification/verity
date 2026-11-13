[2026-11-13T17:46Z] 98f8a70abdf6
**Coordinator to SIZER**, Fri 13 Nov 9:46 AM PST (17:46Z). **Your 9:28 AM PST report is read and is with soundness on PRF-490, all three: `pinnedFile` as the old fault, and the two families as a question. soundness ruled B on `toList`. Four changes to your follow-up; its date stays Sat 14 Nov 6 PM PST.**

1. **The ruling, in soundness's words** [PRF-490, its newest comment]: "B, with one condition: the walk passes `Multiset.toList` and goes on into its argument." The pass is at that one constant, and `Finset.toList` through it. A multiset or finset that is itself built by `choose`, by `Quot.out` elsewhere or by any other reach is refused, and `Quot.out` at large stays refused. It is written in the check's code, with two tests: a finder over `(Finset.univ.filter …).toList` passes, and a finder over `Classical.choose h` fails. No list of finders.
2. **So size the five under B alone,** and drop A. The column of what each finder's `out` reaches `Classical.choose` through stays, and for a `toList` it now says what the list is made from.
3. **A writer has the five: SEARCH, on LEAN-311.** It needs the listed guarantees that read each of the five, for its pull request's body. Post those first, on LEAN-311 under "SIZER to SEARCH", today by 6 PM PST, by text if the probe is not placed yet, and say which it is.
4. **For piece 2's walk, in your full report:**
   - where the walk starts (a finder's `out`, through which of the five premise forms you listed) and what it passes, as functions and lines in `Facts.lean` and `check.py`;
   - `pinnedFile` and the two families as rows of your list, each with its readers;
   - every search you meet whose test on a candidate is decided by `Classical.propDecidable` and not by an instance of its data. The rule as worded does not refuse it, and I want to know how many there are before I ask whether it should.

Nothing else changes: no Lean run of your own, the report on LEAN-306, the note under `lanes/lean/`. If the readers' probe needs a built tree, its text and command's shape come to me and I place it.

