[2026-11-14T06:50Z] 15a06da79dd7
CALLERS, Fri 13 Nov 10:50 PM PST. **#9889's must-fix and both should-fixes are in one push: head `bd87c4b01bfad9b89d445524cd22674fe8d537e7`, suites `r20261114-062253-eec8` "6 suites: 6 passed (0 cached), 0 failed; 32.3 min of test time", `tree_sha` = the head.**

- RAN `r20261114-062253-eec8` (`suites.py --fresh --changed 6c52a8dd9cae`, on a node). READ in its `job.json`: `tree_sha` `bd87c4b01bfad9b89d445524cd22674fe8d537e7`. It names 6 suites: rho 615, repository 362, circuit-properties 119, inference_only 21, training_slowdown 696, verity-graph 246, all passed. No merge of main.
- 1, must-fix, the reading I chose: `rho.lean`'s own sets. `rho/lean.py`'s cached `_locks()` became `read(locks=LOCKS)`, which `status` calls unchanged. The test's `listed(locks)` is `read(locks)`'s two sets plus the guarantees' keys (the `nci/arguments.py` reading). Why: a citation then passes exactly when `rho.lean.status` gives it anything but `absent` (or it is a guarantee key), with one reading that can't drift. No other code reads `_locks` (READ, `rg`). Under it the two locks list 7,049 names (RAN on my VM).
- 2, should-fix: `NAME`'s last part may start lower case or `_`. RAN on my VM: still 35 cited, 34 listed, `HECharged` the one missing, and no new citation from the wider pattern.
- 3, the toy: it now writes a toy lock (a guarantee key ending lower case with its signature, an assumption, a `reads` entry of an assumptions module and one of another module) and checks what is missing. RAN on my VM: the toy fails with the old `listed()` and with the old pattern. The reader's two mutations (`MUT = "Nci.Policy"`, `MUT = "Nci.Guarantees.noSuchBound"` appended to `rho/classes.py`) each fail `test_every_lean_name_the_calculator_cites_is_in_a_lock`. Reverted after.
- 4, the first line: "Every string of the calculator that is wholly a Lean name, as a literal or as a module's value, is one a lock lists." The docstring also says a name built inside a function, or inside prose, is not seen. The scan is not widened.
- For your decision, RAN on my VM, a scan of backticked names in the calculator's strings, docstrings included (outside tests, under the locks' roots, not namespaces): 5 names, 3 no lock lists.
  - `Accounting.Hardware`, `rho/hardware.py:1`.
  - `Accounting.linkCapacity`, `rho/hardware.py:1`.
  - Both are real definitions in `security/Definitions/Accounting/Physical.lean` (READ) that no lock lists.
  - `Properties.NetworkCertifier.PerServerIngress`, `rho/envelopes.py`'s docstring beside `draft:per-server-ingress`. READ, `rg` over `security/` and `verity/`: no Lean declares it, so it is a dangling name.
- The body is updated to the new head: what a lock lists, the toy, the counts, the checks. RAN on my VM: ratchet gate 0 failed, ruff clean on both files.
- Runs in flight: none.

