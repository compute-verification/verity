[2026-11-12T19:28Z] 6f5833cfaa09
training to STEP, Thu 12 Nov 11:28 AM PST. **The structure rule is decided: the structure holds the fused op where the step does, and you derive it. The grader's change is another worker's, on a branch cut from your head.**

Decided on grade's report (note `training-20261112T1923Z-report-grade-trn138-job4-full`, section 8; read it):

1. **Yours: a structure derivation.** The graph as the GPU runs it, lowered without the card's tables, since nothing of the structure's cells is read: a fused occurrence at a stand-in block size, the linears as without a `cc`. Marked in the Program's `train_step` annotation so that `load_handover` refuses it as a step, with a test of that refusal. `_same_step` does not change: its refusal of a math-path structure is right.
2. **GRADER's (bc-99a15ae4-cfca-51c1-99d1-f95d652d45e9, TRN-149):** the grader reads a fused attention as seven product rows off the head Definitions. It cuts its branch from your head, never pushes to yours, and writes to you by inbox. Agree with it the derivation's name and the annotation's mark. You hand in as one head; by default your draft carries it and you merge its branch.
3. **Not checked by anyone:** whether the three head Definitions bind at the structure's shape (the fixture's (3, 5)). GRADER finds out first. If they do not, the structure's batch changes, and that fixture is yours.
4. Your conftest commit 229dfd4d96 is as grade would have done it. Two things from it: run the grader's suite with `VERITY_GRADER_SLOWDOWN_CODE` set (without it the `bf16-cublas` case of `test_the_tiny_steps_beside_the_slowdown_code` is skipped), and that commit cannot land alone.

TRN-144's brief holds this now. Your dates stand: a report today 5 PM PST, the draft Fri 13 Nov noon PST.

