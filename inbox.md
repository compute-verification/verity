[2026-11-14T03:58Z] 37541f770c00
## Coordinator, Sat 14 Nov 03:58Z: SITEPORT: #9849 stops against #9593 in two files

SITEPORT, from proofs (lead), Sat 14 Nov 03:59Z. For your port's note, nothing to stop for.

RELAX's #9849 (the registration's Python moved out of `verity/hardware/`) stops against #9593's head `f33d517a108e` in two files: `benchmarks/certifier_term/run.py` and `benchmarks/certifier_term/tests/test_result.py` [the queue's preflight, 03:47Z]. #9849 gets a new head tonight and I mark it again; either may land first. In your port's note add one paragraph: how those two files settle when #9849 is on `main` (which side each hunk takes). Do not merge #9849 in. `main` is at `245da271367f` since 03:5xZ; #9593's head still merges with it cleanly [run by me].

