[2026-11-14T06:57Z] ef707a4abbd1
TRN-166 ROUTES, Fri 13 Nov, 10:57 PM PST.

**Part 2 is done: draft #9964 (head 0d6dbc68351, base main) gives each of the 23 kinds on my list a case, every case passes, and the suite passed on a node.**

- RAN: run r20261114-062721-fad6, `suites.py benchmarks/training_slowdown --fresh` at 0d6dbc68351. Its last line is "1 suites: 1 passed (0 cached), 0 failed; 13.7 min of test time", with 787 tests passed. I read it from stdout.log after `research fetch --all`, and the tree_sha matches the head.
- Each case passes, with no difference found. That holds in the agreement test, at 3 steps as written, and in the either-kernel test, which is parametrized over JOBS. No case is xfail, and none needed a stated reason.
- Cost (RAN): six of the ten slowest tests the run lists are new cases, between 6.8 s and 17.6 s each; the slowest is strideshare, on a 4-block start. They add about 2 min on a 4-CPU VM. The docstring says what they catch. The new cases over 5 s are unmarked, as the test's existing cases are; marking them slow is your call.
- Change: `synthetic.KINDS` and a `blocks` option for the tiny start. `OPTIMIZER_PRODUCTS` is now derived from the dials, so every Muon, Shampoo and QAT job gets the products check. No new test file, so `untraced` is unchanged.
- Not on main before Monday's rename means one cross with restack.py, on your word. I have not marked the draft or touched the queue.
- Handoff note: training-20261114T065706Z-handoff-ROUTES. Nothing is in flight.

