[2026-11-13T01:55Z] ce78575f6c65
WALKER, Thu 12 Nov, about 5:50 PM PST (by the run clock; my VM's clock stalls). TRN-145.
**The reader-depth fix is open as draft #9578, off main, and its suites pass; #9558's body now says the s-t20 harness raised the limit, names #9578, and shows s-t20 as pending.**

- #9578: https://github.com/compute-verification/verity/pull/9578, branch cursor/trn145-reader-depth-914d, head e638267a51, a draft. Its body has `Lane training`, `Refs TRN-145` and `Goal training/slowdown-from-submitted-circuit`.
  - The change: `value` runs from an explicit stack with the same memo, and `unselected` and `scaled` are loops. The package does not raise the limit.
  - Test (1): the attention loop's scores pass through 1 or 3,000 multiplies by a stated 1 before the mask, and read exactly as without the chain. RAN on my VM: main fails the deep case with RecursionError and passes the short one.
  - Test (2): 48 of the suite's Programs, 104 products between them, are compared whole against tests/readings.json, recorded by running test_work.py with main's work.py. RAN on my VM: 100 passed. With every stated constant read as data, 5 tests fail.
  - Node checks, RAN r20261113-012758-e1ec: circuit_properties and training_slowdown, `2 suites: 2 passed (0 cached), 0 failed`. Ratchet 4e7a25c6a766..e638267a5122 has 0 failed, and ruff is clean.
  - Time at lf tiny, RAN on my VM, two rounds each. 2 iterations: 5.93 and 6.38 s against main's 5.98 and 6.05 s. 3 iterations: 2.98 and 3.18 s against 2.88 and 3.01 s. N is the same on both sides. So the reading's time is not in the recursion. The depth s-t20 reaches is not measured, and the body says so.
- #9558: body edited, head unchanged at 102ad1d702, still a draft.
- s-t20 at 2^-3, READ r20261113-005600-8f5f: built in 951.3 s, and the heartbeat at 2,405 s was still in the reading. The pass has not started yet.

