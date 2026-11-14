[2026-11-14T06:57Z] ef707a4abbd1
TRN-166 ROUTES, Fri 13 Nov, 10:57 PM PST.

**Part 2 is done: draft #9964 (head 0d6dbc68351, base main) gives each of the 23 kinds on my list a case, every case passes, and the suite passed on a node.**

- RAN: run r20261114-062721-fad6, `suites.py benchmarks/training_slowdown --fresh` at 0d6dbc68351. Its last line is "1 suites: 1 passed (0 cached), 0 failed; 13.7 min of test time", with 787 tests passed. I read it from stdout.log after `research fetch --all`, and the tree_sha matches the head.
- Each case passes, with no difference found. That holds in the agreement test, at 3 steps as written, and in the either-kernel test, which is parametrized over JOBS. No case is xfail, and none needed a stated reason.
- Cost (RAN): six of the ten slowest tests the run lists are new cases, between 6.8 s and 17.6 s each; the slowest is strideshare, on a 4-block start. They add about 2 min on a 4-CPU VM. The docstring says what they catch. The new cases over 5 s are unmarked, as the test's existing cases are; marking them slow is your call.
- Change: `synthetic.KINDS` and a `blocks` option for the tiny start. `OPTIMIZER_PRODUCTS` is now derived from the dials, so every Muon, Shampoo and QAT job gets the products check. No new test file, so `untraced` is unchanged.
- Not on main before Monday's rename means one cross with restack.py, on your word. I have not marked the draft or touched the queue.
- Handoff note: training-20261114T065706Z-handoff-ROUTES. Nothing is in flight.

[2026-11-14T07:00Z] 553c21a7934d
WALKER, Fri 13 Nov, 11:00 PM PST.

**M2 needs your ruling: comparing a specific claim's rows within one sequence still refuses the honest small model, because at the point's one-token input the step's specific rows are equal within a sequence too. I propose testing the specific-only claims at a second point whose inputs are drawn word by word. I am doing M1, M3, M4, S2, S3, N2 and N1 meanwhile.**

First, the VM was replaced at 06:54Z. I rebuilt what I need from the notes, and no work was lost: every head was pushed.

What the summaries hold. READ in `work.py` and `points.py` at A: a claim is a product site's operand status (`distinct`, `specific`, `data`, `gathered`) and the product's dims. `specific` is one bit. It is set in `product_status` from a softmax's rows over the product's members, and it records no grouping of rows by sequence. The one sequence length a Program has is `least_sequence(sites)`, the one `keep` uses. So "one member's sequence" has to be inferred from the layout. I did that once, uncommitted, and it fails, as follows.

What I ran on the VM, at A, in scratch files only (all of them restored):
- RAN, READCIRC's probes rebuilt from the report: M1 with signs is charged at N 48; M1 with S all 1 is refused (`w`, rows 0 and 1); M2 is charged at N 32. Both reproduce.
- RAN, the prototype: claimed = distinct or specific, and a specific-only claim's rows are compared within blocks of `least_sequence` rows. M2 is refused, rows 0 and 1, and `test_work.py` passes 113 of 113.
- But training_slowdown's `-k point_evaluates` errors, `Derived[TrainStep]_v1 ... node 122: operand 'x' ... rows 0 and 1 equal`. RAN, instrumented: block 16, A 1, R 32, K 64, and the equal groups are [0, 1, 3] and [16, 17, 19]. The layout is one sequence after another, and three positions of one sequence are equal word for word.
- INFERRED, why: the point makes every token one token, so every V row is one row. Attention at each position is then an average of equal rows, which is that row up to the sum's rounding. Positions 0, 1 and 3 round alike; the rest differ only by rounding. The specific rule rests on `P_BATCH` ("past each row's first position, no two positions of the batch hold equal words"). The point's input breaks exactly that premise. So at this point a specific row passes by rounding noise, or is refused by it.

What I would do. A specific-only claim is tested at a second point whose inputs are drawn word by word, as θ's are (positive floats near 1, every other width a nonzero word), so that the inputs meet `P_BATCH`. All of a member's rows are compared, with no grouping by sequence. Distinct claims stay at the one-token point as now.
- Cost: at s-t20, a second walk over the Program, about the pass's 14,635 s again, unless a walk can stop once its last specific claim is read.
- INFERRED: the M2 probe is still refused there, since q's rows are 0 for any input. It is not yet RAN, and I have not checked that the honest models pass there.

The other way is your report's option (b): state it as a premise, with the M2 probe as a test showing it is charged, and correct `work.py` 62–64.

Say which, and I build it. Until you say, A has no M2 change, and no suites or s-t20 run is launched: they wait for A's whole fix.

