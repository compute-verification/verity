[2026-11-16T22:52Z] a9ce31a1dd59
RECORD to the training lead, 22:52Z. #10597 (head 849983c747be), not marked, as asked: it breaks two suites that main passes.

- r20261116-220050-1748 failed (rc 1, tree 849983c747be) [ran]:
  - benchmarks/training_slowdown passed, 803 tests, test_resume.py green on the node's workers. main's training_slowdown r20261116-220131-936f done, 803 passed. So on these two runs test_resume.py is green on both trees, and the one failure PAGE saw under -n 4 didn't recur [ran].
  - research, repository and spot/prover passed. The research suite passed whole here, so the treelock pair is node-dependent, as infra says on TOP-402 [ran].
  - benchmarks/spot_figures failed (44 failed, 24 errors) and verity-bench failed (32 failed, all test_exfiltration_*.py) [ran].
- Cause:
  - #10597 deletes the `[exfil.*]` tables from benchmarks/slowdown/data/reference.toml. benchmarks/spot_figures/figure1.py still reads that file (its `SLOWDOWN`) and its `REFERENCES` name `("exfil", "sparseloco-1000", "bits")` and `("exfil", "rl-rollouts", "bits")`, so it raises `KeyError: 'exfil'` at figure1.py:282 [read].
  - Locally, test_figure4.py fails with that KeyError at 849983c and passes 4 of 4 at main e80c1a37ebb [ran].
  - That verity-bench's exfiltration tests fail for the same reason is my inference from the same test run; I didn't trace each one [inferred].
- Fix for PAGE: keep reference.toml's `[exfil.*]` tables (spot_figures and the exfiltration benchmark read them), or move what they read with them. Then rerun spot_figures and verity-bench.
- Still running: #10553 at 7ef39626a6c5 (r20261116-224141-40b7), #10556's suites (r20261116-215836-155e), and #10619's suites (r20261116-222008-48f0).

