[2026-11-12T18:19Z] 6dea9b9879ee
NETSEATS to the network-accounting coordinator, Thu 12 Nov 10:19 AM PST. **The scratch on lean's branch is built and pushed: `cursor/na-device-start-clock-onlean-4e9a` at `b3fcd10dd7a8`. Its two suites runs are going; their exits follow.** The pull request's branch is untouched at `08740c9dfa46`.

**#9369's head run is in** [ran]: `r20261112-165657-07cc` at `08740c9dfa46`, exit 0, class SUCCESS by its status, 48 suites passed of 48, no failed line, no guard line. I put it in the body next.

**What the scratch is** [ran]: `08740c9dfa46` merged with lean's `238ad6621c27` (the tip of `cursor/lean-remote-merge-onmain-fc8d` when I took it, and still its tip at this line; three commits on `main`'s `6d95fcc68dd3`), then two commits of mine.

**What stopped: four files, eight blocks, all closed by hand.**
- `benchmarks/network_traces/warden_serve.py`, three. The imports: mine, with lean's new name `BlockGrid` for `Record`. The body of `serve`: mine (the device's state, the start signed before the origin, the origin on the device's clock) with lean's sentence on the pacer. The log: both sides' fields, `clocks` and `paced`.
- `verity/protocols/certifier/verifier/pacer.py`, two. lean's drain before the wait stays; the wait and the arrival time read the pacer's clock, not the steered one.
- `warden.py`, two: the docstring and the signature, both sides kept.
- `coin_link.py`, one: the import.

**The line git merged by itself and got wrong** [ran]: lean's rule that an unpaced device on the host's clock is refused tested the steered clock by name (`clock is time.monotonic`), so on the device's raw clock it no longer fired.
- Mended: it refuses on every clock of the host's (`pacer.CLOCKS`: the raw one and the steered one).
- The test you ordered is in: with no `paced`, the default clock, the raw clock and the steered clock are each refused by name, and only an injected clock runs unpaced.
- It was not silent to the suite: with lean's line put back, lean's own test of that refusal fails too.
- My test of the raw clock's waits now runs paced, on a stand-in pacer that keeps the clock it was started on.
- One effect to know: `serve(paced_ms=0)` on the real clock is now refused by name. The command line already refuses `--paced-ms 0` (lean's check). Tests drive `serve` unpaced on an injected clock, as before.

**The table's line, built ahead as its own commit** (`8c5d1c14d0e3`): `CERTIFIER_START` in `domains.py` and in `Identity/Domains.lean`, the same string. The certificate keeps its literal, as its other tags do, and my test holds the two equal.
- `tests/test_domains.py`: 4 passed [ran, on the VM].
- The Lean file, with its siblings: `r20261112-180139-d0e6`, exit 0, "no theorems in it", no name declared twice [ran].
- No lock record should move [argued]: the identity lock lists no guarantee, and the locks that read the table keep digests for its two parts only, which I did not touch. I launched no records run. Say if you want one before the mark.

**The table's line widens the check.** Most suites read `verity/commitments`, so the change from lean's branch now touches 85 suites, not 59. Both runs are on tree `8c5d1c14d0e3`:
- `r20261112-180023-b7fe`: the 74 that run together, limit 200 minutes.
- `r20261112-180056-2250`: the 11 that run one at a time, limit 240 minutes.
- `r20261112-175824-a54c`: cancelled by me after one minute, still waiting for a slot. Its limit was set for 48 suites.
- The head is one commit past those runs' tree: `b3fcd10dd7a8` rewords one docstring in `warden_serve.py` and changes no code.

**On the VM, one file a command, on the merged tree** [ran]: the device 60 passed, the pacer 9, the coin link 16, the device's state 10, the runner 24, the start 6, the chain 5, the gate 100, the probes 12, the live challenger's coin link 6, the stall probe 5, the cost run 9, no wall clock 4. Ruff is clean on the files I resolved.

**The lint on the retired name** [ran]: 14 new sites against lean's branch, 12 at the pull request's head. The two more are the device constructed in the ordered test [argued from the count; the tool prints 12 of the 14].

**Not done, and when:**
- I take lean's branch once more at lean's ready line, not before. The suites run again at that head.
- The body's file and line cites move with the merge. I rewrite them at rule 3.

[2026-11-12T18:19Z] 68f56f3e5158
TERMGB to the network-accounting coordinator, Thu 12 Nov, 10:20 AM PT. **Both scratch branches are pushed with every stop closed, your A in. Three runs wait for a slot on the second one. The runs' exits follow when they end.**

**The branches** (lean's branch taken once, at `238ad6621c27`)

- `cursor/na-records-rates-onlean-b4e6` at `58306be0e2f2`: #9305's `2d5f7a169d85`, the merge `19fa5aa21660`, then the weights-tie lines alone in `58306be0e2f2`.
- `cursor/na-forged-no-proof-onlean-b4e6` at `c0f05db5ae4c`: #9329's `c77b01b8b79a`, the merge `13f1498ad330` of the first scratch (no stop), then `c0f05db5ae4c`, the tests' sample of the ZK server's refusal built from `RA.ZK_SERVER_REFUSED`.
- Nothing is pushed to either pull request's branch.

**The six closes** (16 hunks; in each, lean's additions stand at #9305's rates)

1. `served_verdicts.sh`: lean's text stands (the challenger's coins, the day, the deferral's settings, the tie's preflight worded for either W). From #9305: `V3C_LAW` a rate in every place, the two defaults, the preflight line on a rule that is no rate.
2. `test_served_verdicts.py`: lean's tests stand. From #9305: its rates in the expected values.
3. `prove_path.py`: both sides. #9305's `window_design` and the required `--law RATE`, beside lean's `pearl_tie`, `window_commit_of_builds` and `--pearl-passes`.
4. `records_audit.py` (a docstring): lean's paragraph stands, with "a stratified rate" where it said `stratified:K`.
5. `test_prove_path.py`: both sides' tests.
6. `test_records_audit.py`: lean's argument (`declared`) with #9305's record of the rule.

**What is in the merge beyond the body's resolutions**

- **Your A.** Under `PEARLPLUS_TIE` there is no default. The preflight exits 3 with what the launch must state and that the count is not known here; no number is suggested. `pearl-tie` refuses an empty rule too. Tests: none stated exits 3 from both entry points; a rate that is not stratified exits 3; a stated stratified rate goes on and reaches the audit.
- **`--law` is stated once.** `window_commit` hands it to every commit, so lean's own `--law "${V3C_LAW:-subset:1}"` on its three callers is dropped.
- **The rate preflight is skipped in a compose follow-on.** It reads no rule, and under A its rule can be empty.
- **Two of lean's tests take a rate:** its untied test (it expected `subset:1` and `stratified:1`), and its own-builds test, whose `window-commit` call named no `--law`.
- **`test_prove_path.py` is a stop the body did not list.** Both sides add a test at the same place; both stay.

**Checked on this VM** [ran]: `test_served_verdicts.py` 411 passed (two selections); `test_records_audit.py` 49 passed; `test_prove_path.py` 82 passed and the one that failed (lean's own-builds test, no `--law`) passes once fixed; `test_weights_tie.py` 11 passed; `test_run_rules.py`, `test_pearltie.py`, `test_remotetie.py`, `test_kept_scan.py` pass.

- One test fails on this VM alone: `tools/verity/cli/tests/test_onsite_served.py`, at a line neither side changed. This VM writes a redacted value where the test's runner lists a key's digest. The suites run decides it.

**The runs, on the second scratch at `c0f05db5ae4c`, each launched once, all waiting for a slot at 10:19 AM PT**

- `r20261112-181342-dd6c`: the suites both changes reach (`suites.py --changed 238ad6621c27 --quick`).
- `r20261112-181454-3f14`: the real two-member module, both tests, binaries built in the run.
- `r20261112-181836-593d`: the weights-tie arm of `run.sh` (`PP_WEIGHTS_TIE=1`). You did not ask for it. It has never run with a rate, and its two forged arms go through #9329's check, so I would rather know today.

**The close for #9362** (`260823dd87e6`), written down and not applied

- One stop, a docstring in `test_audit_two_members.py`. #9305's sentences stand (the rule is the rate the audit is given; `rule_of`, which #9362's text names, is gone), and #9362's last sentence on whose coins and the case is added after them.
- Its assertions merge without a stop, among them one line in the forged test: sampled proofs' answer on that audit is `reject`. I take it to hold, since the audit's record there is not accepted [argued from `served_integrity.answer`; not run]. The two-member test must run again once it is applied.

**The command that counts the Pearl+ window**, for its owners: `prove_path.py class-circuit --serve SERVE_PHI --class CLASS --pearl-tables pearl-c-h100-v1-h3 --out F`, CPU only, from φ alone. F's `templates` lists each template's proof blocks; the largest is the DEN at which `stratified-rate:1/DEN` samples one of each. A Pearl+-tied run already writes that record before it serves, as `pearlplus/circuit-record.json` in its serve dir [read in the script; not run].

**No branch or run of the other agent seen** [ran: `git ls-remote` for the two changes' names; I did not list runs].

TIMER: none

[2026-11-12T18:24Z] 645701d1d44e
NETSEATS to the network-accounting coordinator, Thu 12 Nov 10:24 AM PST, to your 10:20 AM PST. **Done as ruled. The scratch's head is now `4b67da63b3f0`, pushed; the pull request's branch is still `08740c9dfa46`.**

- **The rule on the tag** is in the scratch as its own commit, `7fb534f4840f`: in `certificate.py`'s docstring where the start's certificate is defined, and in PROTOCOL.md's "The device state and the session's start". It is not in the table's commit: that one (`8c5d1c14d0e3`) was pushed before your message, and I do not amend. On the pull request's branch the two go in as one commit.
- **The test file is renamed** `test_session_start.py` (`4b67da63b3f0`); nothing in the tree named the old file [ran: `git grep`]. 6 passed under the new name [ran].
- **The body is edited** [ran: read back]: the rule and what it rests on in place of "The start signs nothing about the clock"; the two reads of the raw clock, in both places; the sentence on new names says the one exception and its rename.
- **The kernel point:** I know nothing otherwise. My memory of the kernel is the same as yours, and I did not check it here either.
- **The runs** stand at tree `8c5d1c14d0e3`. The three commits since change two docstrings, PROTOCOL.md and one test file's name. The lint counts 14 new sites still [ran].

Next from me: each run's exit, one line.

[2026-11-12T18:37Z] 58e33ea32006
ONSITEMAIN to the network-accounting coordinator, Thu 12 Nov 10:38 AM PST, to your 9:56 AM. **Six workers is in both launch lines. The script and the reader changed by one line each, so both digests are new.**

**1. `LAUNCH.md`** [ran]: `V3C_NEGATIVE_WORKERS=6` in both lines, and under them the sentence: six, your ruling of 9:56 AM PST; seven's worst reading sits on the stop's line and a run the stop ends keeps no negative; six leaves room at every member's size and costs 580 s. The `sampled:` line's W reads the count from the environment: at six it prints 2,600 s [ran on the toy with `V3C_NEGATIVE_WORKERS=6`: "ends about 2600 s from here"]. Nothing fixed there.

**2. The two figures for the demo's run were not printed at the stage's end**, so one line each, no new sampler [ran]:
- The script, when the stage of the outcome and the negatives' workers is left (the 5 s read that was already there):
  `REHEARSAL +Ns the outcome and the negatives' workers ended: T s in the stage; MemAvailable's lowest in it L GB, X GB under the job's start and Y GB under the stage's own start`
- The reader, with the result, from a5's own record of the workers (it was already copied into `rehearsal.json`, never printed):
  `REHEARSAL negatives' workers: 6 at once, W s; each task's own seconds, longest first: ...; the most a worker held of its own at its end A GB, the highest peak of a worker's Lean process B GB`
  No task is named and no index or count of the window is in it: seconds and bytes only.
- The seconds of a block with five others beside it are that line's list. Which entry is which negative is in `rehearsal.json` under `negative_workers`, by the negative's label.

**3. New digests** [ran]:
- `/tmp/rehearsal/rehearse_audit.sh`: `0193a347231e605b545f5d7979fa0cfcffdaf0df9ac5cf18e5d4f5ec4c89510c`
- `/tmp/rehearsal/rehearsal_read.py`: `6fdded1e5e4dfe052b21431e73303b68a3c82e85c33185ffcd1fefcdcd6a202d`
The 9:55 AM files are kept beside them as `prev-0955/`.

**4. The toy, once** [ran, three runs]: six workers on the four-hour line, exit 0, both new lines printed; six workers with `JOB_SECONDS=16200` and a member of 8.2 GB, exit 0, "ends about 5780 s from here … inside the job's 16200 s"; the memory stop inside the workers' stage, exit 72 as before (the stage's line is not printed there: the stop's own line names the stage and the figure). The work parent was empty after each.

**Least sure of:** the stage's line reads MemAvailable every 5 s, so a peak shorter than that is missed; and "under the stage's own start" counts from the first read inside the stage, by which the workers may already hold memory. The figure under the job's start is the one to plan the demo's run on. Not looked at: whether a5's `own_s` of a task includes its wait for the pool's slot; I read it as the task's own seconds inside its worker [read, `NegativeWorkers.timing`], not measured.

The port's recount follows in the next message.

[2026-11-12T18:37Z] 62105ff8546c
# NET-327: the recount on lean's branch, and the cuts (ONSITEMAIN, Thu 12 Nov 10:40 AM PST)

Marks: [ran] a script of mine on the agent VM; [read] git's output at the named commit; [derived]. No run launched.

## The one thing to know first

**By file, almost nothing of staging is on lean's branch or in an open pull request of ours.** Of the 262 files
staging changes, lean's branch (`238ad6621c27`) has staging's change in 6. In 33 more staging changed wording only.
The other 223 hold code the branch lacks: 113 where only staging changed a declaration, 110 where both sides did.
Our open pull requests touch 29 of the 262 and hold staging's own declarations in three files only, all #9319's (the
pool). What reached `main` since Tuesday reached it rewritten, in other files or under other names (#9350's pooled
form, #9333's and #9378's rows), and a count by file does not see that. [ran]

## How it was crossed

- `restack.py` as a command cannot cross staging: its first step merges each move's base into the branch, and
  staging is 5,290 commits from `main` with over 150 merge bases (the remote side). [read] I ran its second step
  alone (`run_script`: the move's base's own script at the root of a scratch worktree) for each of the 21 moves
  lean's branch has and staging lacks, on staging `00d79b22e4d2` and on its base `700a981f5d90`, with no merge. [ran,
  826 s a tree] Results, local refs only: `refs/scratch/onsitemain/staging-crossed` `9f4ad72cccb3`,
  `refs/scratch/onsitemain/base-crossed` `573de5fc6f1b`.
- Three departures from the moves' own lines, each mine: `--no-lock` on every layout move; `rename.py`'s refusal of
  a map's rows that match no tracked file turned into a report (72 rows, files `main` has and the side lacks; none
  is one of the 262); and 46 rows added to `layout_one_map.toml` for the side's files under a retired path, each
  placed where lean's branch has the file. Two have no place on lean's branch:
  `verity/core/protocols/certifier/scan_service.py`, staging's own, which I put beside `tie_service.py` at
  `verity/protocols/certifier/verifier/scan_service.py`; and `pous/period_reports.py`, not staging's.
- The check on the crossing: for 89 of the 262 files the crossed base is byte for byte lean's branch. [ran]
- The count: `git diff` of the two crossed trees gives the 262 (Tuesday's 261: 255 found by rename, 6 deleted by
  staging and found by path, 1 new: research's `moves.json`). Each is compared with lean's branch by text
  (`git merge-file`) and, for `.lean` and `.py`, by declaration with Tuesday's script, comments, docstrings and
  string literals left out. The sort is by text, not by meaning.

## The count, by item

| item | files | branch has | wording only | only staging changed | both changed | touched by an open PR of ours |
|---|---|---|---|---|---|---|
| 1 the period check | 11 | 0 | 0 | 5 | 6 | 3 (job 6) |
| 2 a5, the v2 case, the record | 30 | 0 | 2 | 12 | 16 | 6 (#9319, #9305, #9329) |
| 3 the service | 4 | 0 | 0 | 2 | 2 | 0 |
| 4 the ingress tie | 19 | 0 | 0 | 15 | 4 | 2 (#9333, #9369, #9378, job 6: prose) |
| 5 the rows audit, its caller and test | 14 | 0 | 2 | 10 | 2 | 1 (job 6: a root's prose) |
| 5 the drivers, other | 64 | 0 | 12 | 34 | 18 | 6 (#9329, #9369, job 6, prose) |
| 6 the composites | 24 | 1 | 2 | 7 | 14 | 1 (#9333, #9378: `test_run.py`) |
| 7 Security | 70 | 4 | 6 | 18 | 42 | 6 (#9350, job 6) |
| the certifier's other files, under no item | 24 | 0 | 9 | 10 | 5 | 4 (#9333, #9369, #9378, job 6) |
| other (`moves.json`, a file both deleted) | 2 | 1 | 0 | 0 | 1 | 0 |
| all | 262 | 6 | 33 | 113 | 110 | 29 |

What the open pull requests hold of staging's declarations the branch lacks [ran: each head against staging's
crossed file, a rename counted as none]:
- #9319: `a5.py` 17 of 41, `tests/test_a5_lean_pool.py` 18 of 19, `a0.py` 2 of 4.
- job 6 (`cursor/network-accounting-okbit-constant-4e7a`): 1 of 40 in `Proofs/Properties/NetworkPeriod.lean`, 0 in
  `Period.lean` (of 69), `period_case.py` (of 75), the protocol test (of 21), `Properties/NetworkPeriod.lean` (of
  24). It changes the same four files item 1's cut changes. It sits on lean's old tip `f3837f57d113`, one of the
  heads that never land.
- #9350: none of staging's files; `PooledLeakRegistered` is in its own files. `PeriodLeakRegistered`,
  `RowsStateHeld`, `KSWired`, `RemoteNetwork`, `kind_scan` and `rows_audit_case` are named on staging and in no
  `.lean` or `.py` of lean's branch, `main`, #9350 or job 6. [read: `git grep -lw`]
- #9305, #9329, #9333, #9369, #9378: 0 of staging's declarations in every file they share with it.
- NET-379's branch is #9378 (`cursor/na-held-rows-5b2e`), cut on #9333's head.

## The cuts

Order as I would build them. "Shown by" is one `research run` each unless it says otherwise.

| cut | files | rests on | whose yes | shown by |
|---|---|---|---|---|
| **A. The rows audit's Lean** | 6 and two locks: the certifier program's `NetworkCertifier/Rows.lean` (new) and its root; the verifier's `NetworkCheck/RowsAudit.lean`, `RowsMain.lean` (new), its root, `lakefile.toml` (the `network-rows` executable); the two packages' `lean-audit.json` | item 5: the rows audit comes on top | ours | the two packages built with `network-rows` linked, and their Lean check |
| **B. Its caller and test** | 6: `warden_chain.py` (`RECORD`, `deployment_of`, `own_record`, `rows_audit_case`) and its test; `certifier_piece.py` (`ROWS`, `rows_check`, `lean_exe` by name, `--rows`, step 1's call) and the three rows tests of its test; `scan_period.py` (`lean_check`'s two lines); the suite's `pyproject.toml` (the built executable as an input) | item 5 | ours | the suite's rows tests on the built executable: the honest run accepted; cells cut, clocked or ordered otherwise refused |
| **C. Item 7, the tap and the warden** | 4: `Definitions/NetworkCertifier/{Warden,Tap}.lean`, `Proofs/NetworkCertifier/{Warden,Tap}.lean`. Only staging changed a declaration in them and they read no file both sides changed | item 7: beside `main`'s | ours | `verity lean try` on the four with `--siblings` |
| **D. Item 7, the rows' format** | 2: `Definitions/NetworkCertifier/RecordRows.lean` (10 new declarations), `Proofs/NetworkCertifier/RecordRows.lean` (22). `fxQueueSched` keeps `main`'s place and value; staging's value differs (`![5, 1, 2]` against `![6, 2, 2]`) | item 7; **one point to rule: staging's proofs at `main`'s fixture value, or a second fixture beside it** | ours, and yours on the fixture | `verity lean try` on the two |
| **E. Item 7, the wired form's statements** | 9: `Definitions/Claims/Sampling.lean` gains six declarations (`BlockCheck`, `correctOk`, `IncorrectBy`, `incorrectBy`, `RBy`, `KSBy`); ours: `SampledAudit.lean` (the boundary gates and wires), `SampledCharge.lean` (`freeInputs`), `Period.lean` (`WindowGoodReg`, `RegOpensAt`), `PeriodConditions.lean` (five conditions), and `Guarantees/{Period,SampledAudit,SampledCharge}.lean`, where each onsite guarantee becomes a new one beside `main`'s; `Flock/Guarantees/SamplingClaims.lean` | item 7: `main`'s statements stand, the onsite ones beside | lean for `Claims/Sampling.lean` and `SamplingClaims.lean`; **yours for the four new guarantees' names** (staging reuses `PeriodLeakAccepted`, `SampledAuditAccepted`, `SampledLeakAccepted`, `SampledLeakRegistered`) | the definitions package built and its Lean check |
| **F. Item 7, the proofs that are ours** | 20: nine new (`Wires`, `KSWired`, `ReadFree`, `WiredNet`, `WiredRead`, `WiredToys`, `NoWires`, `KsToy`, `PeriodReg`) and eleven both changed (`Period`, `PeriodWitnesses`, `SampledAuditAccepted`, `SampledAuditInstance`, `SampledCharge`, `SampledLeak`, `SampledLeakReg`, `SampledLeakRegPlain`, `TapRows`, `Witnesses`, `WitnessToys`); the new instances go to `Definitions/NetworkCertifier/Witness.lean`. Staging's ten declarations in proofs' `KS.lean` go to a file of ours, since `KS.lean` gains no line | item 7; after E | ours; proofs reads what leaves `KS.lean` | the proofs package's NetworkCertifier area built, replayed, and its lock by `--update` |
| **G. Item 7, the rest** | 30, not ours: lean 26 (`Properties/NetworkPeriod*.lean`, `Proofs/Properties/*`, `Verity/{Game,Idle,Soundness,Subprotocols}`, the roots, `lakefile.toml`, the locks), proofs 2, memory-accounting 1, and the pin. The third instance of the soundness theorem is not ported | item 7, item 6 | lean, proofs, memory-accounting | the Lean check, whole |
| **H. Item 1, the period check's own change** | 11: `Period.lean` (69 declarations, 9 both changed), `Circuit.lean` (new), `RecordsMain.lean`, `period_case.py`, the protocol test, the package's README and `pyproject.toml`, two manifests, the lock; with it `scan_period.rows_case` and Security's two `NetworkPeriod` files | item 1: one change of its own, with NET-325 and NET-328 | ours; lean for Security's two; **order with job 6, which changes four of the same files** | the period suite and the package's Lean check |
| **I. Item 5, the kind scan** | `kind_scan.py` and its test (new); with it `attacks.py`, `scan_tie.py`, `network_sim/onsite.py`, each new with its test | item 5: the kind scan on top | ours | each suite's new tests |
| **J. Item 5, the run class and the rest of the drivers** | about 40: `chain_record.py` and `.sh`, `scan_zk.py`, `negatives.py`, `n2_zk.py`, `compile_slice_zk.py`, `leak_bounds.py`, the rest of `certifier_piece.py` (the front, the auditor's record's gate, the relay), their tests, the READMEs. `main`'s count of unchecked inputs stands | item 5 | ours; top for `tests/test_retired_terms.py` | `benchmarks/network_traces`' suite |
| **K. Item 5, the serving driver** | `served_session.py` and its test | item 5; **waits for vllm's answer** | vllm, ours | not built |
| **L. Item 6, two development executables** | `CoreRemoteNetwork.lean`, `CoreRemoteNetworkNoPouw.lean`, their two mains, the lakefile and the lock; and the removal: neither is in `composites` (`config.py`, `Site.lean`, `site/__init__.py`, the site difftest, `composite.py`, `test_run.py`, `test_site.py`). They read `Period.lean`, so after H or on `main`'s `Period` | item 6 | soundness, proofs | the package built, both executables linked |
| **M. Item 3, the scan's service** | `scan_service.py` and its test (new), calling `main`'s `Stream`; nothing of `spot/verifier/service.py` | item 3 | ours; **the file's place is my row, not the layout's** | its test |
| **N. The certifier's other files** | 15 with code: `warden.py`, `ingress_scan.py`, `service.py`, `standin.py`, `domains.py`, `Identity/Domains.lean`, tests; after #9333, #9369 and #9378, which change `warden.py`, `active.py` and the two prose files | none of the seven: Tuesday's clean files | ours | the certifier's suite |
| not cut | item 4's 19 files (staging deletes the tie's service and driver; `tie_service.py` stays; `scan_tie.py` is in I); item 2's 30 (out; the record and the site's registration read its case); 33 files of wording only | items 4 and 2 | | |

Every body carries: a record written from `main` is not of record for sampled proofs on a window of several
members, and every window of the onsite demo has several; `main`'s verifier refuses that form until proofs proves it.

## What I build next, without waiting

A, then B, then C, in scratch branches on lean's branch at `238ad6621c27`, taken once more at lean's ready line.
Nothing pushed to an open pull request's branch; no pull request opened.

## Least sure of

- **The sort is by text.** A declaration lean's merge rewrote by hand counts as lacking where staging's differs by
  more than a rename. The count of "both changed" is an upper bound on what needs a decision.
- **Cut B's size.** Staging's `certifier_piece.py` changes 35 declarations and the rows audit is a few of them; step
  1 of the piece reads the auditor's record on staging. I carry `own_record` and `deployment_of` with it and leave
  the record's gate (`held`, `--certifier-key`) to J. If that split does not hold when built, I say so.
- **E's shape.** "Beside" for a guarantee both sides state under one name means a new name. I have not tried whether
  staging's proofs go through with `main`'s `WindowsPlaced`.
- Where I did not look: whether any cut builds; #9375's own diff against `main`; proofs', vllm's and soundness' files
  beyond their counts.

[2026-11-12T18:40Z] 5b85f6708444
ENTRIESONCE to the network-accounting coordinator, NET-284 Part 1, before I commit the view rows' charge. One thing you asked to hear first: **`scanOf`'s result gains a field, and no check of `scanOf` changes.**

- [read] `Scan` gets a sixth field, `view`, and `scanOf`'s last line sets it: `view := T * paramBits de.params`, the `T` egress instances' `state` parameters. These are already input gates `scanOf` counts in its `inputs ==` check (`T * paramBits de.params` is a summand there), so the bits come from the registered Program. No refusal is added, removed or reordered; `ingressReturn`, `argsApart`, `inputsRead` are not added.
- [read] So the view charge needs no field of the registration: that stop condition does not fire.
- [argued] The charge, for you to overrule: `viewTieHeld := false` (no registration states the tie, nothing reads it). A scan with every unit proved is charged its view rows alone: no tied audit's free inputs, no faulty entries, no slot entries (Monday: 384 + 384 + 25,600 + 32). **A scan sampled in part is charged as today plus its view rows**, since they are untied input gates whatever the draw; NET-284's first item says "otherwise it charges as today", so this is my addition. Say so if you want part-sampled scans left as today.
- [read] Group A's rule (entries once, no minimum) stays in `Scan.terms` behind `viewTie := true`, which no accepted period's term is at; guards and the mirror test it there. `RowsKBoundsCount` is restated over both values of `viewTie`.
- [read] The public's rows form gains `scans[].view`, `scans[].whole` and `scan_bits: {entries, view}`, so the record can state 1,696 as K − view + entries from the check's own output without the check printing it.
- [read] The diff touches none of the five functions NET-301 moved, nor the lines between them.

Nothing is launched yet. Next: commit, push `cursor/network-accounting-view-rows-675b`, then the Lean tries.

[2026-11-12T18:40Z] f6043fdbd2e8
TERMGB to the network-accounting coordinator, Thu 12 Nov 10:41 AM PST. **One more meeting of #9329 with lean's branch, not a file stop: lean's weights-tie arm asks `--forged` of a window that holds no forged unit, and #9329's rule refuses to pass it [argued, not run: the run of that arm still waits for a slot]. I stop on it and ask before I choose. Your 10:36 AM order (NET-394) is read; one line on it at the end.**

**What meets**
- lean's branch brings `PP_WEIGHTS_TIE=1 bash benchmarks/prove_path/run.sh`: one honest tie and two refused ones, both audited with `records_audit.py audit --forged` behind the ZK server [read: `run.sh`, the `tie` function].
- lean's own rule for `--forged` was: every check before the sample passed, the record rejects, and some Lean verdict is false [read: its `records_audit.py`].
- #9329's rule behind the ZK server is: no proof of the false member leaves the prover, and the prover's record gives the ZK server's refusal as why [read: `forged_outcome`].
- **`weights-tie-forged`** (the last units' digests forged to X's): those units are not their Definition's, the ZK server stops them, and #9329's rule passes it [argued, the same path as the stand-in's forged arm].
- **`weights-tie-other`** (checkpoint Y's words served as X's): every unit is its Definition's. What fails is a read: each last unit's digest opens no root the verifier supplied [read: the arm's test, `test_weights_tie.py`]. The ZK server is set up on the statement, the partition and the program, not on the verifier's registered reads [read: `a0.firewall_env`], so a true proof leaves it and Lean refuses the session on the read. Under #9329's rule that arm then fails: `no_proof_released` is false [argued].
- No check runs this arm, so nothing red would show on `main`; the arm itself would exit 1.

**The choice, yours**
- **A (what I would do):** the audit says which refusal it expects. `--forged` stays #9329's, word for word. One more mode, `--refused-read`: the window holds no forged unit, a read of it opens no root the verifier holds, and it passes when every check before the sample passed, the record rejects, and Lean refused a proof it read (#9329's `lean_refused`, not for want of the session's files). `run.sh` gives it to `weights-tie-other` alone. About ten lines in `records_audit.py`, one in `run.sh`, one test; a commit of its own on #9329's scratch, after the merge.
- **B:** leave it. The arm exits 1 on that tie after #9329 until its owner (lean's worker for the weights tie) changes it. I name it in #9329's body.
- **C:** `--forged` passes on either check behind the ZK server. I advise against: a false unit whose proof left the ZK server and was refused by Lean would pass, which is what #9329 closes.
- Until you answer I change nothing there. The run `r20261112-181836-593d` shows which it is when it gets a slot; I send its verdict with the 1:30 PM report.

**Runs at 10:38 AM PST** [read: each status]: `r20261112-181454-3f14` (the two-member module) running; `r20261112-181342-dd6c` (suites) and `r20261112-181836-593d` (the weights-tie arm) waiting for a slot.

**On NET-394, one thing I read that does not agree with the order** [read: NET-390's comments]. soundness's comment of 10:35 AM PST says the figure of record "stays the two-schedule one", with the simulated card's rows printed beside it. Your order and NET-394 make D_j = 1 the term of record and print the two-schedule figure beside it as "without the card". I build what the order says unless you tell me otherwise before I start, which is after the 1:30 PM report.

Least sure of: whether the ZK server releases the proof in `weights-tie-other`. I read its setup, not a run of it.

TIMER: none

