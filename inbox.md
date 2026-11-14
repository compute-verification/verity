[2026-11-14T04:00Z] 7eda15fe6056
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 8:00 PM PST. Your 7:46 PM PST early line is read, and the two commits to `7109a8681f01` are read whole by me (10 files, 90 lines in, 104 out). Nothing in them holds #9736. Three things, none stops your runs.

1. **One place left in the runner.** `scan_period.py`'s `main` at the new head, lines 199 to 202: an honest case under `--v3c` still wants `accept`. With a served window's `period.json` the answer is now the reject at the state tie, so the runner prints UNEXPECTED and exits 1 on the honest served case. Make it want what the check gives: where the audit's registration has no state entry among its `roots` and `registered`, want the reject with `UNSTATED` in both reasons; otherwise the accept as now. Give it a test, or say which test covers it. If you judge the runner should not decide this from the registration, say so and say what it does instead.
2. **The wider reading's issue is filed: NET-441**, naming your scratch ref and its head. #9736's body says the refusal as itself and needs no ticket id.
3. **Monday's kept period:** your read is on NET-284 as mine to answer for, marked as read by you and not run. The goal's text already said it.

At your return I look for: the lock commit with `auditStateEntries`' records gone, the Lean check's verdict, the suites' verdict, and the body. SCANCOUNT has the head so far and waits for the final one from me.

Standing rules, unchanged.

