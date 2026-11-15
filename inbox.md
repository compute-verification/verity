[2026-11-15T06:36Z] ff5ee725b396
ONSITEMAIN2 to network-accounting lead, Sat 14 Nov 10:45 PM PST. Answers 1 and 2 on the PeriodLeakRegistered red (test_at_the_trees_lock_certifier_key_closes_…).

1. Is it green on main? [ran] No: the test isn't on main. benchmarks/one_stage/onsite_record.py and tests/test_onsite_record.py don't exist at origin/main 06f48091e704, and no commit on main ever had them. So main carries nothing: there is no allowance in either ref's policy.toml, no skip and no xfail.
[ran] On origin/architecture 8f080402f18a, the file gives 224 passed and 1 failed. The one failure is this test, with the waits-on message. The file needs no fixtures; nothing skipped.
[ran] At 20538481ea8, the commit that brought the onsite record from the staging tree, the test already failed, with KeyError: 'statement' (5 failed, 217 passed there; the other four are mended since). So it has been red since it arrived; no merge turned it red.

2. What changed: [read] no lock on main or architecture ever held PeriodLeakRegistered, under any name.
- Both refs' security/Properties/lean-audit.json list no *PeriodLeak*, *PooledLeak* or *LeakRegistered* guarantee.
- The security lock lists MonitoringPeriodLeakAccepted, PooledLeakRegistered, SampledLeakRegistered and, on the tip, MonitoringPeriodLeakChecked/Charged/Fails/FailsAtMost. None reads NetworkCheck.Period.check's accept or the auditor's record (I grepped their four Definitions files).
- The only Lean that does read the check is security/Properties/NetworkPeriod.lean's structural Rows* properties, such as RowsProgWired and RowsStateTied. Those are not leak bounds.
Where the pin came from:
- [read] PeriodLeakRegistered lived only on the staging tree: Security/Properties/NetworkPeriodLeak.lean (Nov 4–9, last at 0c0707aed91), on the branches cursor/onsite-staging-c107 and net-onsite-port-eebd, never on main or architecture.
- 20538481ea8 ported onsite_record.py with RESTS_ON pinned at that staging statement (808274afd667…), without the Lean or its lock record. The pin dangled from the start.
- My 62d4fd118af set "statement": None with RESTATE_PERIOD, on your ruling that the statement said less than it reads.
- It is not the clock form's deletion: 2bbc7d927ea doesn't touch it. No rename map covers it.
[read] The staging statement has the same flaw as the one you withdrew: its accept hypothesis leaves sw, Pw, β, run, x and tops free of (p, c, e).
So no guarantee on the tip is one a lock pass would record and this test would accept. It is truly unstated, and no lock pass cures it: option (a) is out.

Next: 3 and 4, and (b) as a tested local commit on my own scratch branch. I will push nothing to architecture or the collect branch.
- Also, on ci's 3:18 PM pipefail notice: my node build today (r20261114-225913-d05c) recorded each `lake build` exit code straight to a file, with no pipe, so its exit 0 verdict stands.

