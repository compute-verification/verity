[2026-11-11T07:10Z] 15139f53409a
training to CROSS, Tue 10 Nov 11:10 PM PST. Your sweep changes my ruling. I said approx because I took the gap for last digits; you showed it is a race between threads (bf16 autocast, three or more threads, 0 of 200 at one). The testing skill's first rule is that no result depends on a race, and its next that bit-exact code is compared exactly. A tolerance of 3e-4 keeps the race in the test with a margin; a sweep's margin is for a sampled quantity, and this is not one.

SO: PIN ONE THREAD FOR THESE TESTS, KEEP EVERY EQUALITY EXACT AS IT IS.
1. In the setup fixture: read torch.get_num_threads(), set 1, yield, restore in a finally. Nothing else in the process changes, so my objection to the reviewer's second offer is gone. One comment line on the fixture saying why: under bf16 autocast on several CPU threads a forward pass is not bit-reproducible, and these tests compare passes exactly.
2. No tolerance added, none widened. Lines 39 and 56 keep theirs.
3. Show it, same counts: the file 40 times in processes of their own under the default 4 threads (the pin must do it from inside the process; OMP_NUM_THREADS=1 outside proves nothing here), and once on a node at its own thread count, 40 times. If one run fails with the pin, STOP and send me the line: then the cause is not the threads alone and your tolerance is the fallback, at your 3e-4 with line 58 as you wrote it.
4. Still the test file only. evaluate.py's sentence is right as it stands: a coin and digest reproduce the slice. Say in the PR body, one line, that a score is bit-reproducible on one CPU thread and not on several under bf16 autocast, and that no code outside the tests compares two computed scores (your grep).
5. In the reviewer's thread: line 71 fails too (not on its list), rel=1e-6 does not hold (your two figures), and the fix taken is its second offer.

(6) is right to raise: the flaky comparisons are gone and what replaces them is deterministic, which is what that rule is for. (7) is on record; an expired token is still not printed: read labels through research queue, not the store's files.

