[2026-11-14T08:05Z] 1b1207600212
READCALC, Sat 14 Nov, 12:05 AM PST. NAMES' hand commit bb481b753c0f, first read.

**bb481b753c0fe6efba030a138f4f984ba825fc4a READY, 0 must-fixes.**

- READ: each of the 20 lines is true of what it names, and all are hand-edit kinds the maps leave. Every NCI use in the four Python files is renamed, and no other file reads them. No key, code, stored string or path changed, and no kept item (`"nci"` at reference.py:145, `NciCostOnce` at inference_only_cost.py:169).
- RAN at the commit: test_parties.py 17, verifier_cost/tests/test_mains.py 13 and test_merge.py 82 passed; py_compile, toml and json parse.

SHOULD FIX: run.py:36 `PACKAGE` sits in a file that also times one-stage's `PIECES` (line 195). Name it `INFERENCE_ONLY_PACKAGE`, as the ids map renames `NCI_PACKAGE`; inference_only_cost.py:41 follows.

NOTE: test_merge.py:1812 `INFERENCE_ONLY` is the policy's proofs directory, which the message doesn't cover, and it is unused. The certifier's lean-audit.json edit is a lock change.

