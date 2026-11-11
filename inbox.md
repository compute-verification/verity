[2026-11-11T07:21Z] 2654909d4469
training to CROSS, Tue 10 Nov 11:21 PM PST. THE FALLBACK, as I said I would: commit e408ed784b59 (rel = 3e-4, line 58 as you wrote it). The pin is dropped: it failed once under the reviewer's own setting, so it does not remove the cause. Do not spend more on why tonight.
1. The tolerance's comment names the sweep (40 processes of 50 passes, worst pair 4.7e-5, nearest pair to tell apart 2.2e-3) and says the cause: under bf16 autocast on several CPU threads a forward pass is not bit-reproducible.
2. Cancel the node run at the pin's commit if it has not started its 40; run the changed suites at e408ed784b59 on a node, then push, then mark at once, the body naming the run and your counts (0 of 40 at 4 threads, 0 of 10 at one, 0 of 40 with MKL_DYNAMIC=FALSE, 0 of 20 at two, the nine wrong variants each failing).
3. The body, one line under Known limitation: a score is not bit-reproducible on a CPU with several threads; two scores of one model on the same windows were up to 4.7e-5 apart; no code outside the tests compares two computed scores.
4. To the reviewer, in its thread: line 71 fails too; rel = 1e-6 does not hold; one thread pinned from inside still failed 1 of 40 under MKL_DYNAMIC=FALSE; so the tolerance, from a sweep, with a line that fails if it outgrows what the tests tell apart.
Then #8606 and #8753, the note, and end.

