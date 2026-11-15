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

[2026-11-15T06:41Z] e013b62558a9
ROWSMERGE to the lead, 10:41 PM PST: the onsite test fix is carried; I'm stopping. architecture and cursor/na-architecture-b51f are both at 27aa7f73b791, pushed at 10:41:29 PM PST. ONSITEMAIN2's 0e53fed3fd33 touches only benchmarks/one_stage/tests/test_onsite_record.py. Collect had nothing architecture lacked, so it moved forward to architecture's tip ef12696a6113. I cherry-picked the commit on top, with no conflict: on the branch it is 27aa7f73b791, the tip itself. Counts on it: test_onsite_record.py 225 passed (whole); tools/verity/cli/tests/test_onsite_served.py 50 passed, 1 failed; test_onsite_periods.py 30 passed; benchmarks/demo_minutes test_onsite_minutes.py 8 passed; tests/test_repository.py 17 passed with 4 bookkeeping warnings; py_compile OK. The served failure, test_the_served_window_runs_once_with_every_piece_s_hooks (an env dict differs, H1_PEER among 13 identical items), also fails on architecture's tip without the commit, so it is not ours. Check: the domain tag files are unchanged, all 35 kept strings and 16 kept patterns are intact, and the allowlist is architecture's (parses, no path twice).

[2026-11-15T06:47Z] 70b5001f0caf
TERMGB2 to network-accounting (lead): ORPHANS done.

Branch `cursor/na-orphan-roots-b51f`, one commit `f930518dcd8d`, cut from `origin/architecture`'s tip `a03cb37a2957`. Pushed only there; nothing went to `architecture` or `cursor/na-architecture-b51f`.

1. Confirmed [ran]. I reproduced check.py's rule (`judge`: a package's own file, by `lean_modules`, that is not among the modules the lock's `roots` reach by imports and not under `exempt`) from the sources and the locks. Exactly four are ours:
- `security/Proofs/Properties/Certifier/Registered.lean`, which is `Proofs.Properties.Certifier.Registered`;
- `security/Properties/Certifier/Registered.lean`, which is `Properties.Certifier.Registered`;
- `security/Properties/InferenceOnly/PinnedRead.lean`, which is `Properties.InferenceOnly.PinnedRead`;
- `security/Properties/InferenceOnly/RecordForm.lean`, which is `Properties.InferenceOnly.RecordForm`.

There is no fifth one [ran]. I checked every `.lean` under `security/**/Certifier/**` and `security/**/InferenceOnly/**`, and NetworkPeriod, NetworkScan, Ties and `Accounting/Actual` under `security/Proofs/Properties/`. The `security` (Definitions) package has 0 orphans. The same computation finds proofs' ten (nine Proofs files and `Properties.Verity.LedgerPinned`), which checks the method. The other orphans are PoUB's and PoUS's.

All four were already lakefile roots [read]: `security/Properties/lakefile.toml` lines 111 and 128, and `security/Proofs/lakefile.toml` line 99. Only the locks' `roots` lacked them. So the cure is lock lines only, as proofs did for `Properties.Verity.LedgerPinned`. The lakefiles are unchanged.

2. All four are roots; none is dead [read].
- `Properties.Certifier.Registered` states `RegisteredLinkRated`, with its witness pair `RegisteredLinkRatedHonest` and `RegisteredLinkRatedNontrivial`.
  - `security/Definitions/Accounting/Guarantees/RegisteredSite.lean` and `LinkEnds.lean` cite it in docs.
  - Its only importer is the Proofs file.
- `Proofs.Properties.Certifier.Registered` proves `theorem registeredLinkRated : RegisteredLinkRated`. Nothing imports it.
  - No lock lists `RegisteredLinkRated` yet. It is something a lock should name, but naming it is a `guarantees` field and outside this task.
- `Properties.InferenceOnly.PinnedRead`. After `4c59dd4b82df` it still holds:
  - the definitions `PinnedReadAt`, `PinnedRead` and `ReturnsRow`;
  - `PinnedReadBinds`, which is still here and proved as `theorem pinnedReadBinds` in `security/Proofs/Properties/InferenceOnly/Lemmas/Opening.lean`, a file Proofs' roots reach;
  - the witness statements `PinnedReadNontrivial` and `PinnedReadHonest`;
  - `OpenedRowReturned`.

  No lock lists any name from this file. It already builds through Proofs, via `Opening.lean`.
- `Properties.InferenceOnly.RecordForm` holds `def form : Properties.Record.Form`. Eight Proofs InferenceOnly files and `PinnedRead` import it. No lock names it.

3. Lean's AREAS branch: `cursor/lean-arch-lock-layers-fc8d` does not exist on origin [ran]. The nearest ref is `cursor/lean-areas-map-fc8d` at `37aa9c71059`, which does not contain the tip. Neither of its locks lists or roots any of the four [ran], so I left none out.

4. The exact added lines. Neither list was sorted, so I kept each list's lakefile order, and the old entries' order is unchanged [ran].

`security/Properties/lean-audit.json` (`roots`):

```
   "Properties.NetworkPeriod",
+  "Properties.Certifier.Registered",
   "Properties.Core.Period",
...
   "Properties.InferenceOnly.Opening",
+  "Properties.InferenceOnly.PinnedRead",
+  "Properties.InferenceOnly.RecordForm",
   "Properties.Certifier.JointBridge.SpaceOfPeriodHeld",
```

`security/Proofs/lean-audit.json` (`roots`):

```
   "Proofs.Properties.Certifier.Records",
+  "Proofs.Properties.Certifier.Registered",
   "Proofs.Properties.Core.Period",
```

There is no `exempt` entry, no record, no lakefile change and no other key [ran: the key-by-key diff of both locks shows only `roots`].

Checks [ran]:
- Both locks and both lakefiles parse, with no duplicate roots.
- After the change, none of our files is an orphan.
- `tests/test_lean_packages.py` and `tests/test_repository.py`: 51 passed, 0 failed. Four tests in `test_repository.py` (owners, Markdown caps, report basenames, owned paths) warn on bookkeeping that names none of these lines.

There was no Lean build. That the four build is proofs' worker's word.

[2026-11-15T06:55Z] 51b76ecd7fae
DEVICEGONE to network-accounting (lead): lock pass, steps 1 to 3 (counts and state at the tip). Patch follows on cursor/na-lock-hand-lines-b51f.

Source [ran]: r20261115-054953-79ac stdout.log and lean-audit/requirer-security*/, 727 lines (466 security, 221 Properties, 40 Proofs), same totals as ci's file. Tip = origin/architecture = a03cb37a2957.

Ours: 108 lines. layers 85 (security 14, Properties 71), shape 16 (security 5, Properties 11), orphan 4 (Properties 3, Proofs 1), witnesses 1, open 1, guarantees 1.
Not ours, though the InferenceOnly filter caught them [read lock owners at tip]: the 8 `reads` lines (InferenceOnlyCheck...CheckedCoverTraining and DescThrottles*: @training; InferenceOnly.SecurityProofs.PartitionCovers: @lean). No reads line of ours, so class (b) is empty for us.

(a) Hand-kept, and I mend them in the patch. All still hold at the tip [ran]:
- security shape.frozen-stale x4: NetTiming.SecurityProofs.{EncardAccSeqsLeAdviceBits, EncardAccSeqsLeConstantRate, EncardAccSeqsLeConstantRateExecutionPhases, EncardDecodableLeConstantRate}. No decl at the tip. Remove.
- Also stale at the tip, not in the run's lines: 2 shape.frozen entries and 2 witnesses entries, NetTiming.SecurityProofs.Encard{AccSeqs,Decodable}LeConstantRatePerExecutionPhase. LOCKPASS's dress commit 5cf7ba575ea already dropped them, which is why the run didn't list them. Remove, the same as the dress.
- Properties shape.properties: rowsStartCharged has no entry. I add {function: NetworkCheck.Period.check, outcome: accept}, as its siblings rowsTiedCharged, rowsOkHeld and sampleBoundCharged have. I expect this also cures its `guarantees` line, whose "inline hypothesis" is `check p c e = accept`, its acceptance. That is unverified without a build.
- Properties shape.unstated x10: Properties.NetworkPeriod.{AnswersVerified, RatioReadsBack, RowsEntriesKeyNontrivial, RowsEntriesOrderNontrivial, RowsEntriesParsed, RowsExecutionPhaseAccepted, RowsSessionsComplete, RowsSessionsCompleteHonest, RowsSessionsCompleteNontrivial, SignsCertificateAt}. Each is defined, and no theorem at the tip proves it [ran: git grep]. I add a shape.frozen entry for each, with the `unstated:` reason check.py accepts (category `unstated`). Say no if you'd rather move them out of the spec.

(a) but yours to decide. I don't edit these:
- witnesses: Accounting.SecurityProofs.LeakPartsRegistered. The witnesses exist (Proofs/Accounting/Actual/RegisteredWitness.lean: LeakPartsRegistered_honest and _nontrivial). But _honest concludes Valid ∧ AtLinkEnds ∧ RecordsWithin ∧ Nonempty and not Verity.Site.RatedOn, which the rule now requires: 6e1ccffab4a added RatedOn to the statement and left the witness behind. Listing it would point at a lemma that doesn't conclude the rule's form. Your choice: a Lean fix (ours: add a RatedOn conjunct for the toy β), or witnesses_assumed with a reason.
- open.unlisted: Verity.Site.RatedOn (Definitions/Core/SiteSubtree.lean:88), a premise of EgressRegistered and LeakPartsRegistered, listed in no lock. An open entry needs {owner, digest, issue, due}. The digest is a record (--update writes it), and the issue and due are yours to name. Proposed: owner @network-accounting, the issue you choose.
- shape: Accounting.SecurityProofs.EgressRegistered takes A.cell_inj, a field of an input that is neither its acceptance nor a premise. It needs a Lean change (take it as a hypothesis or a named Prop), or a shape.frozen reason you approve.
- Leave to --moved (no edit): the Declares and TrustedHost names. certifier_area_moves.json and device_moves.json reach them.

(c) Not ours to edit, passed on. All still hold at the tip [ran]:
- layers 85, for AREAS: Properties.Certifier.JointBridge 68, Definitions.Certifier.JointBridge 9, Definitions.Certifier.Bridge 2, Definitions.Certifier.NetWitness 2, Properties.NetworkPeriod 2 (it imports Lms.Sign and Spot.OneStage), Definitions.Certifier.ResponseTie 1, Properties.InferenceOnly.Desc 1. The tip's layers sections for these keys are unchanged from the dress, and the imports are still there. Full lines on request (I hold them in a file).
- orphans 4, for TERMGB2: Properties/Certifier/Registered.lean (in both Properties and Proofs), Properties/InferenceOnly/PinnedRead.lean and Properties/InferenceOnly/RecordForm.lean. Only Proofs files import them at the tip, and the roots don't name them.

Least sure: that the rowsStartCharged shape.properties entry clears its `guarantees` line, and whether you want the 10 NetworkPeriod statements frozen as unstated or moved out of the spec. Not looked at: Proofs-lock lines other than the orphan, and the Registration entries (none of ours in the run's lines).
TIMER: none

