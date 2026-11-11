[2026-11-11T09:13Z] f87cb05306e9
LEANMERGE to the coordinator, Wed 11 Nov 1:13 AM PST. `378a`'s outcome, read from its files (`fetch --all`).

**In one line:** the package's own check passed and kept a record (`regenerated`, four true values, all my change's,
committed and pushed); the run failed on its requirer `Security`, for one lock value of my change and one failure that
is the base's, not mine. Not the node's.

## 1. Verdict lines, the package and step that ended it, `path`

- `path`: `slot` [run: `result.json` `workload_fingerprint.path`; log line 1 "lean-records: taking the slot path"].
- What ended it: the requirer `Security`, in `check.py`'s lock rules (`guarantees`, `trees`), exit 1 after 573 s. The
  build, the axioms and every declaration passed in all three packages.
- Verbatim [run: `r20261111-070615-378a` `stdout.log`, rc 1]:
  - `AUDIT …/Security/Properties: PASS  3290 declarations in 104 modules; axioms propext, Classical.choice, Quot.sound; 533 guarantees`
  - `lean-records: Security/Properties: check.py exit 0 in 5046 s … without the kernel replay or the tests: a lean-fast/v1 result`
  - `lean-records: Security/Properties: the regenerated record is …/lean-audit/record/Security/Properties/lean-audit.json; commit it over Security/Properties/lean-audit.json`
  - `AUDIT …/Security/Proofs: PASS  198848 declarations in 3896 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees`
  - `AUDIT …/Security: FAIL  27598 declarations in 727 modules; axioms propext, Classical.choice, Quot.sound; 1099 guarantees`
  - `lean-records: Security: check.py exit 1 in 573 s …`
  - `lean-records: Security requires Security/Properties, and its records fail on this build, not only on record differences, which take a person`
  - `OUTCOME`: `Security/Properties` regenerated, `Security/Proofs` pass, `Security` failed; outcome failed.
- Of the 5046 s, 4400 s were `proved-in-wait` and 338 s the build [run: the audit's `seconds`].

## 2. Which of the three

**`Security/Properties`: `regenerated`, true values, mine. Committed.**

- The kept record against the head's lock, as JSON: four values differ, all under `reads`; key order and layout are
  the same (the text diff is those four lines); `format` is 1 in both [run: my comparison on the VM].

| Entry | Was | Is |
|---|---|---|
| `reads["NetworkCheck.Period"].digest` | `58568376bc79…` | `22674ce8e3b4…` |
| `reads["NetworkCheck.Period"].definitions["NetworkCheck.Period.factsOf"]` | `ee751ac31ab2…` | `d6edc3686ad2…` |
| `reads["Properties.NetworkPeriod"].digest` | `90782066c03a…` | `3dbdf9f0d053…` |
| `reads["Properties.NetworkPeriod"].definitions["Properties.NetworkPeriod.RowsTiedChargedAt"]` | `e77972111642…` | `178350d0e4fd…` |

- Both are my change's own: `factsOf` has the three refusals, `RowsTiedChargedAt` the conjunct `f.drawn ≤ f.units`.
  The audit's `review` prints these two definitions as changed and no other (its one further line: "reads: 397
  definitions read in another module or under another name: the same under their old names").
- **`rowsTiedCharged`'s own record does not move**, against what we expected: its `type_hash` (`967b07df…`) and
  `unfolded_hash` (`01fba8c7…`) are as in the base's lock, so its `unwitnessed` entry stands. No guarantee's record
  moves in this lock (0 of 533). The statement's change is carried by `RowsTiedChargedAt` under `reads`.
- Who reads `factsOf` there, by the `review` line: the three `Properties.NetworkPeriod.rows…` and 23 `Verity.*`
  guarantees (`coins_couple`, `remote_soundness_of_subprotocols`, `soundness_at_of_componentBounds`,
  `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods`, `wipe_target`
  and their witnesses). Their records do not move; the definition they read did.
- **The commit:** `13419dcffc5a04702396bda69d670e874d144183` on `cursor/net325-drawn-le-units-22f3`, pushed as a
  fast-forward at 1:12 AM PST: the kept record over `Security/Properties/lean-audit.json`, byte for byte (sha256
  `ab85ba4ec320…`, the run's `record_sha256`). I did this under your rule of 1:06 AM PST; the brief's "no lock
  commit" said the opposite. If you want the base's lock back, it is one revert commit: say so.

**`Security`: a real failure, two messages. Not fixed.**

- `guarantees: definitions in NetworkCheck.Period that Flock.SecurityProofs.RecursiveAuditC3 read changed (changed: NetworkCheck.Period.factsOf)`
  - Mine: the same `factsOf`. `Security/lean-audit.json`'s `reads["NetworkCheck.Period"]` needs the same two values
    (digest `22674ce8e3b4…`, `factsOf` `d6edc3686ad2…`) [run: the requirer's report's `reads`]. It is the only
    module of that lock whose reads move, and `RecursiveAuditC3` (owner `@proofs`) its only reader there.
  - The run kept no record of `Security`, so I committed none and wrote none by hand.
- `trees: NetTiming.Witness.PooledLeakAccepted_honest read NetTiming.RowsFormat.recordFormat in Proofs.NetworkCertifier.RecordRows, a module of the proofs tree (Proofs, Proofs.Properties): a guarantee's or property's statement reads only the definitions tree …`
  - Not mine. The witness (`Security/Proofs/NetworkCertifier/PeriodWitnesses.lean:30`, owner `@network-accounting` in
    the lock), `recordFormat` (`Security/Proofs/NetworkCertifier/RecordRows.lean:199`) and `Security/lean-audit.json`
    are the base's byte for byte [run: `git diff --quiet f5b870781287 HEAD` on them]. So the base alone fails it
    [argued: I ran nothing at the base].
  - The base's last commits on those files: `0fb03a8fe6` (the witnesses) and `96c2c1a9dc` ("fxQueueSched moves to the
    definitions tree"), which moved one of the two definitions the witness reads and left `recordFormat`.
  - While it stands, no records run of `Security` ends green on this base, whatever my branch holds.

## 3. `d815`: the three suites, all passed [run: `r20261111-070538-d815`, rc 0]

- `passed  verity/core/protocols/network 89 passed in 522 s; 89 outcomes recorded`
- `passed  verity-core-composition      30 passed, 1 skipped in 4560 s; 30 outcomes recorded`
- `passed  verity-network-traces        545 passed in 383 s; 545 outcomes recorded`
- `3 suites: 3 passed (0 cached), 0 failed; 91.1 min of test time`
- The old base's nine reds in `verity-network-traces` are gone: 545 is the 536 and the nine.

## What I did not do

- No second run: nothing here is the node's.
- No run of `Security`'s records, and no hand-written value in its lock. Say if you want the run sent; it will end
  failed on the `trees` line until that is cured.
- I read `378a`'s `stderr.log` tail once on my VM; its slot lines carry other runs' commands, cut at 260 columns, no
  secret in them. I filter those lines now.

Next from me: job 1's report and job 2's draft, by inbox.

