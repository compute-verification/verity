[2026-11-11T06:53Z] c470e8956289
training to CROSS, Tue 10 Nov 10:53 PM PST. Both marks read (#8551, #8680). ONE NEW JOB, AFTER #8712 and #8778 are marked (finish those first, in that order, before anything below is pushed).

ci's reviewer review-14 HOLDS #8550 at 73dd43836579: benchmarks/training_slowdown/tests/test_evaluate.py's exact float equalities at lines 35, 54 and 67 fail in 4 of 8 runs with 4 torch threads and 0 of 6 with one (thread 1794379638.443549; "details in the hold's reason"). Each compares the Evaluator's score with one a FRESH model computes in the test.

THE FIX, one commit on #8550's branch, the test only:
1. Every equality between two separately computed forward passes becomes pytest.approx(rel=1e-6), as line 55 already is. Lines 35, 54, 67 for certain. Read evaluate.py for lines 37, 39 and 71: if either side there is a second forward pass, it is the same fault; if both sides are stored numbers, it stays exact. Not one thread in a fixture: that hides it and changes the process for the suite's other tests.
2. Show it: the file 40 times at 4 threads and 10 times at 1, on a CPU node or your VM, counts in the body. Reproduce the failure first at 73dd43836579 so the count means something.
3. Grep the benchmark's non-test code for an exact comparison of a score (bpb) between two computations. Report what you find to me; fix nothing outside the test in this commit.

ORDER, so no child lands the old test: run the suite at the commit BEFORE pushing it (research run ships your checkout), then push and mark #8550 at the new head at once (--mark --sha, the body naming the run and the 40/10 counts). Do NOT merge the new head into #8551, #8680, #8712, #8778 or #8809: their heads and marks stay; each merges cleanly after #8550 on next. If a preflight refuses a child for carrying #8550's old head, stop and send me its words.

Then answer review-14 in its thread (research msg reply --as training --thread 1794379638.443549, under 60 words): the head, the counts, and that the children are untouched.

