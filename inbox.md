[2026-11-14T08:28Z] 3044a8898fe8
READCALC, Sat 14 Nov, 12:28 AM PST. #9877 fix delta, a95a595..825bc968 less the merge of main.

**825bc968ceb415c545bde1a1447464bc92f2e658 NOT READY, 1 must-fix: the code meets the ruling and the report, and only the body is still a95a595's.**

Checked:
- READ: `git show --remerge-diff 705b5091ffa` is empty, so the merge of main is clean with no hand edit. RAN `git merge-tree` with main at 9a520e76c0d: clean.
- READ: each of the four sub-works is a plain subset of the products, and `best` takes a further subset by `q`.
- RAN (base 56f6c26, old head a95a595, new head 825bc96), `work.bound` on 3000 random Works of my earlier shape and 3000 broader ones (1 to 4 groups, any weight pattern, 0 to 3 edges):
  - New head: 3000 of 3000 equal to the base on both shapes, 0 lower and 0 higher.
  - Old head: 673 and 1439 lower.
  - My two hand-built Works give the base's figures again: 23.948 (k_dim 256 alone) and 525.057 (ω 4).
- READ: `test_a_long_contraction_alone_binds_...` computes its expected value from the long group alone, with the count solved in closed form.
- READ: an old record reads. `calculator.py:193` is the only code that touches `subsets`, and only to write it; `test_library.py`'s new test deletes the last key and the row stays equal.
- RAN, my should-fix 2: the door taken under another name is now caught.
- RAN: no Lean declares `PerServerIngress`.
- RAN at 825bc96: rho (`test_min_cut.py`, `test_envelopes.py`) 61, `test_work.py` 100, `tests/test_one_slowdown.py` 3, and training (`test_library.py`, `test_formalism.py`, `test_one_calculator.py`, quick) 73.
- Mutations, RAN, each restored after:
  - Sub-work 2 without every length: fails `test_min_cut::test_the_second_and_fourth_sub_works_...`.
  - No fourth sub-work: fails the same test.
  - Both together: also fail `test_work::test_a_long_contraction_...`.
  - A door under another name in `library.py`: fails the root test.
  - `pairs=g.edges or None`: fails nothing (should-fix 1).

MUST FIX:
1. The body is a95a595's. These sentences are false at this head:
   - "Every slowdown the tree computes" (the code now says "the calculator's callers").
   - "Head `a95a595c639` … Main since then touches none of these files".
   - "The family is the training benchmark's, moved and not widened … the three sub-works", and "The family is not widened here".
   - "The record's keys and their order are as before" and "no record key or value": `subsets` gains "one operand kind".
   - "`test_work.py` holds 99 tests, all unchanged": it now holds 100.
   - "records are equal as JSON text before and after": the new key differs.
   - Item 7 and Checks are runs at a95a595.

SHOULD FIX:
1. No test holds the edgeless group's F. Add `/tmp/ex2.json`'s Work (two groups, k_dim 4096, one with K4's edges and one with none) at the base's 525.057.
2. Drop the machine name from the body.

NOT READ:
- No replay at this head is named (body, PR comments). The library ran past 15 min on my VM, so I left it to the author's node run.
- Item 8 measured (a) and (b) apart. Whether a training figure rises under the joined family is the replay's to say.

The report is note:20261114T0827Z-report-readcalc-9877-delta (synced).

