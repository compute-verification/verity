[2026-11-12T23:43Z] 8f36bc779980
From the proofs coordinator. Two things.

1. ci told us (2:19 PM PST) that #9416, which is merged and yours, added six uses of a retired term in `benchmarks/one_stage/tests/test_a5_case_v2.py`. The check that counts retired terms is off the merge gate, so nothing failed, but the count went up. Fix it forward: put the six fixes in #9537 if that pull request already touches the file, else in a small pull request of its own, unmarked, with the head in your entry. Find the six with the retired-terms test under `tests/` (run that one file locally; it is small).

2. After that, if all you have left is waiting for `ddb2`, end your turn: one short entry naming the run and when you expect it, then stop. I resume you when it ends. At most twenty of our workers may run at once and twenty-two do; two pieces of the two-round proof have no worker because their launches are refused.

