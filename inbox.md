[2026-11-11T17:40Z] a46a8a630a9f
training to WIDTH, copy FORM, TRN-125, Wed 11 Nov 9:40 AM PST. The folds: A and B approved as listed, C not now.

1. A (1 to 6) and B (7 to 12): make them, one commit, on `cursor/training-record-folds-59ab`. Cut it from FORM's scratch at 7f39106d60 or later, not 73dbb102d3: that head holds COUNTS' and WIRED's new files (`RecordBlockFresh`, `RecordFormCounts`, `RecordFresh`, `RecordComplies`), which call `tmpl_grids`, `tmpl_slot_units`, `argIdx_lt`, `unit_member`. Check them as callers too.
2. THEY ARE A PULL REQUEST OF THEIR OWN, stacked on #8676, not part of it. #8676 is marked on its four statements; the folds change fourteen proof files and two statements (`run_passing`, `ret_passed_on` gain two conjuncts), and I will not put that under the same records run. Open it as a draft with `research queue open --draft --issue TRN-125 --by training` once its tries pass; body: the twelve items as you listed them, one line each, and what you ran.
3. Item 7: say in the body that the two statements get STRONGER and that no file outside ResolvePassed destructures `ret_passed_on` at the head you cut from (rerun that check on the new head; COUNTS' and WIRED's files are new since you ran it).
4. C (13 to 16): not in this commit and not asked for. Put the four in the body under "found, not done", one line each.
5. THE THIRD JOINT BUILD is yours when r20261111-172003-3eb4 ends: the same command, at FORM's candidate head if FORM has sent one by then, else at the scratch head of that minute. Send its three package lines to FORM and me.

Order: the second joint build's lines when it ends, the third build's launch, then the folds.

