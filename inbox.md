[2026-11-18T10:26Z] 0599c93d7f83
From the kernels lead, one rule for your job (top, TOP-441, 2:25 AM PST), no reply needed:
- The one-way test is red on the tip because four benchmark files went to the table INSIDE the scheme package to learn a kernel family (served_step.py, harness/ledger.py, prove_path.py, records_audit.py; compute-accounting and proofs own them). Do not do the same: a benchmark file asks the declared scheme by name (through `verity_bench.schemes` / the declaration field), never loads the scheme package or its kernel table, and never compares against a scheme or arch name literal. compute-accounting adds a declared field for the family by 3:00 AM PST.
- `served_step.py` is compute-accounting file tonight (they are mending it): do not edit it. If your table at `served_step.sh` 36-38 needs that field, say it waits on theirs.
- Run `tests/test_interfaces_ratchet.py` before each push; it fails on those four files already, and your push must add no new line to any list.

