[2026-11-11T19:33Z] 7dc04e33c58f
training to resume, TRN-132, Wed 11 Nov 11:50 AM PST. Your ask is with ci: sent 11:43 AM PST on review-20's thread (1794422617.679739; my message 1794425420.138169), for the `FAILED tests/...::...` line with its first assertion line, where it ran, and whether `ONEDNN_MAX_CPU_ISA` was set, by Thu 12 Nov 10 AM PST. I pass its answer on the minute it comes.

Your reading is accepted as it stands: not the order alone; the fake-tensor lines are every tiny job's stderr and name no test. Go on as you wrote (the slow tests of the reviewer's tree under load, with and without the limit; the package's files in reverse order on a node). If the slow tests fail without the limit and pass with it, that is fault 3 of #8963 and TRN-132 closes as "fixed by #8963", with the run ids, and no new branch. Do not open a PR for a failure you have not reproduced.

