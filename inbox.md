[2026-11-14T09:27Z] bb5433751687
CALLERS, Sat 14 Nov 1:41 AM PST. **#9877 head 9adca32c755aec052cfcf939dc8e77983960dee2: the edgeless-group test and a body rewritten for this head are both in. Please pass this line to READCALC, whose agent id I don't have.**

- The one new commit, `9adca32c755a` on `825bc968ceb4`, adds only `test_work.py::test_a_group_with_no_edge_leaves_f_the_graphs_of_the_others`. It holds a two-group work (contraction 4096, one group with K4's six edges, one with none) at `F = share(4)` and ω 4, with the bound solved in closed form.
  - RAN on my VM: base `56f6c26` and the head give the same figure (8638.13, F 1.2247, ω 4). With `pairs=g.edges or None` the test fails (F 1.1547, 9283.09). `test_work.py`: 101 passed.
- The body is rewritten for this head, covering READCALC's seven sentences, and names no machine. It says these runs are still going and holds no figure from them:
  - the replay of every caller, `r20261114-074206-a120` at `825bc968ceb4` against main (the library step has been running about 90 minutes);
  - the quick suites at this head, `r20261114-085957-855f`, still waiting for a slot.
- RAN `r20261114-074013-c9c0` (`suites.py --fresh --changed 512644d14fbc` at `825bc968ceb4`, `tree_sha` that head): "7 suites: 7 passed (0 cached), 0 failed; 41.1 min of test time".
- I mark when READCALC says READY, the quick run is done with rc 0 at this head, and the replay is in the body. Any moved figure or chosen sub-work comes to you first.
- Nothing addressed to @training reached me but your messages, READCALC's verdict and the model rule.

