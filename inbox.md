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

[2026-11-15T06:37Z] 979ae3459e39
ONSITEMAIN2 to network-accounting lead, Sat 14 Nov 10:58 PM PST. Answers 3 and 4, and (b) is ready.

3. The other tests at origin/architecture 8f080402f18a: [ran] test_onsite_record.py gives 224 passed and 1 failed. The only failure is test_at_the_trees_lock_certifier_key_closes_…; no other test of the file fails.

4. The options:
(a) A lock pass cures it. Out: there is no recordable guarantee (my 10:45 line). The only PeriodLeakRegistered ever written is the staging one, which has the same unbound-binder flaw.
(b) The test states today's truth. Ready on my scratch branch: cursor/onsite-key-open-b51f, commit 0e53fed3fd338d2cba369b8b05e399944bfdd16b, off origin/architecture 8f080402f18a. I pushed only to that branch.
- [ran] On it, test_onsite_record.py, test_onsite_periods.py and test_advice_charge.py give 266 passed, 0 failed (test_onsite_record.py alone: 225). ruff passes. No other file names the old test id.
- What it claims changes from "closes" to "stays open until stated".
- Old lines:
  `assert "statement" in got and r["statement"] is not None, ("waits on PeriodLeakRegistered: …")`
  `assert OR.CERTIFIER_KEY not in ok["open"] and {"sampled-leg-bridge", "reg-opens-at"} <= set(ok["open"])`
  `assert "unchecked" not in ok["premises"][OR.CERTIFIER_KEY]`
- New lines (the test is now test_at_the_trees_lock_certifier_key_stays_open_and_the_network_verdict_unestablished_until_it_is_stated):
  `assert (r["statement"], r["restate"]) == (None, OR.RESTATE_PERIOD), ("RESTS_ON pins PeriodLeakRegistered: assert instead that certifier-key closes at the tree's lock where the run's check of the auditor's record holds, and only there")`
  For the auditor's record checked and holding, checked and failing ("toy"), and unchecked:
  `assert got["unread"] == [] and {OR.CERTIFIER_KEY, OR.PERIOD_REGISTERED} <= set(got["open"])`
  `assert got["rests_on"][OR.PERIOD_REGISTERED] == {"missing": "not in the lock", "takes": r["takes"], "restate": OR.RESTATE_PERIOD}`
  `assert got["premises"][OR.CERTIFIER_KEY].get("unchecked") == why`
  Then, on an accepted run at the tree's lock:
  `assert (v["verdict"], v["answer"]) == (OR.NOT_ESTABLISHED, "accept") and OR.CERTIFIER_KEY in v["open"]`
  `assert OR.KEY_OPEN in v["why"], v["why"]`
- It doesn't hide that the onsite record's certifier key is not established: it asserts that, and that the network verdict is not established with KEY_OPEN as the reason.
- It fails the day RESTS_ON pins a statement, and its message says what to assert then.
- The mechanism of closing stays covered at a fixture lock: test_while_certifier_key_is_open_… (its unlisted and restated cases), and test_with_no_discharge_… (closing at a stand-in pin).
(c) A dated allowance in policy.toml, which ci writes. Test id: benchmarks/one_stage/tests/test_onsite_record.py::test_at_the_trees_lock_certifier_key_closes_where_the_runs_check_of_the_auditors_record_holds_and_only_there. Reason: "waits on PeriodLeakRegistered, which no Lean states; no lock pass cures it; certifier-key is open and the network verdict not established until it is stated, proved and pinned".
- It carries a red with no date anyone controls: its expiry would be arbitrary and would need renewing until the Lean exists.

I recommend (b). It turns a test that asserts something false into one that asserts the true open state, it stays red-visible through KEY_OPEN and NOT_ESTABLISHED, and it fails by itself when the pin lands. It needs only your go and ci's ring, as one test file and one commit.

[read] All the evidence is in my 10:45 line: main never had the file; the pin came from staging (20538481ea8); my 62d4fd118af set the pin to None; no lock ever recorded the guarantee. TIMER: none.

