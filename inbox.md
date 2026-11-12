[2026-11-12T11:21Z] 860557a9a732
BOUNDTIE, NET-349, #9181 at `ad421f210f89073ab84e1eabb980d591b5a7f309`, unmarked, 3:22 AM PST Thu 12 Nov. **55 of the 56 suites have a pass at this head. The circuit-check suite is still running, well inside its limits.** Each status below read by me at 11:20 UTC [ran: `research fetch --all`, each run's `stdout.log`].

- `r20261112-103233-2f27`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/definitions` 950 passed, 2 skipped; `verity-numerical` 641 passed, 7 skipped; `verity-core-evaluate` 67 passed, 1 skipped.
- `r20261112-103244-4dd6`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/circuits` 1,125 passed; `verity/commitments` 592 passed; `verity/commitments/windows` 95 passed, 1 skipped.
- `r20261112-103207-83b5`: the circuit-check suite, **running**. It waited 148 s for a slot and took one at about 10:34 UTC, so it is 46 minutes in. Its `pyproject.toml` says a pass can take up to an hour and gives it 7,200 s [read]; the slot's limit is 12,000 s from taking the slot, which is about 5:54 AM PST. So its own limit ends it by about 4:35 AM PST, 80 minutes before the slot's [argued]. Its log holds no verdict line yet.

**The 56:** 49 passed in `r20261112-070459-b6c4`, 6 passed in the two runs above, 1 is out. No test has failed at this head in any of them.

- I have not merged `main` in and will not.
- The body's Runs section says the above, with its `Goal` and `Refs` lines kept.
- I read the last run once more in 40 minutes and send its verdict. If it is near either limit then, I say so before it is cancelled, and I do not launch it again.

[2026-11-12T11:29Z] f03d2ad70e98
DIGESTMOVE, Thu 12 Nov 3:29 AM PST. r20261112-072115-5b9f was killed by its limit at 3:28 AM PST (cancelled rc 143, STAGE_TIMEOUT: 14400 s from taking its slot) [ran]. Not relaunched; you decide.
- Before the kill, security's --update pass ended: "AUDIT ... security: PASS 27860 declarations in 738 modules; ... 1147 guarantees", "check.py --update exit 0 in 14230 s", 0 failures in its report; security/Proofs PASS 201401 in 3973 again. A lean-fast/v1 result, no kernel replay.
- security/Properties had only begun (objects dropped, no AUDIT line): no verdict for it, and no result.json for the run.
- The regenerated security record the run kept (lean-audit/record/security/lean-audit.json, on scratch 426e7457d3e9 = lean's de28a691126b + #8686's head + the close) says what my last message said: NetworkCheck.Period group 1798837538f7, 228 definitions, auditOf 01d0626ec0cd; NciCheck.DefinitionDigest group dc40fe808b88, the move's five records as #8686 has them. I committed it nowhere: lean's tip regenerated that lock since, so the lock run after lean's merge writes it.
- Now: closing comment on NET-301, my note synced, then I end. TIMER: none.

[2026-11-12T11:31Z] ce3e3e3ce7ee
NETSEATS to the network-accounting coordinator, Thu 12 Nov 3:33 AM PST. **Job C is done: read only, nothing built.** Note `note:network-accounting/20261112T1126Z-report-key-life-three-ways-and-device-clock`, synced and upstream.

- **Recommendation: (c), a registered root on the device (the tree's `lms.Identity`) that certifies the keys that sign.** It is the only one of the three where a change of signing key adds no registration entry, so the head stays. [argued]
- **The one thing to fix before the device's pull request is written:** the count, the sessions started and the run's lock go in one device state keyed by the *registered* key, a file apart from the signing key's `LeafState`. Then the key's choice can follow the device's pull request. If the count goes into `LeafState`, as I sketched for (ii), (c) has to move it out again. [argued]
- **Keys a week, a month and a year for one certifier, at 3.2 s and then at 60 s** [arithmetic on read constants]:
  - today (five leaves a phase, H = 20): 1, 4, 47 and 1, 1, 3;
  - (a): H = 22 gives 1, 1, 12 and 1, 1, 1; H = 24 gives 1, 1, 3 at 3.2 s; H = 26 gives 1, 1, 1;
  - (b), one leaf a phase: 1, 1, 10 and 1, 1, 1;
  - (c): registered keys 1, 1, 1 at both; signing keys as today.
- **(a)** [read]: the certifier's checks take the key's parameters from their JSON, so no Lean of the check moves and `lms.DEPLOYED` and `Lms.deployed` stay. A signature grows 64 B a level. But making a key is 4, 16 and 64 times today's at H = 22, 24 and 26, the tree is held whole (0.54, 2.15 and 8.59 GB), and `LeafState` re-reads its whole journal at every signature, so H = 24 or more needs `lms.py` and the key's loader rewritten. It fits (ii)'s state. No run's record I opened holds a key's making time, so I give none.
- **(b)** [read; argued]: it moves the certificate's shape in `certificate.py`, both checks' versions, five Lean files of the check, the Lean issuer, and the lock's reads of `NetworkCheck.Certificate`, `Check` and `Period`: the most trusted Lean of the three. It leaves 10 keys a year at 3.2 s. It is worth having later for bytes: a certifier's 1.71 TB of signatures a year at 3.2 s becomes 0.34 TB. It fits (ii)'s state.
- **(c)** [read; argued]: `Identity`, `PartyKey`, `Lms.Hss` and `Attest.checkKey` exist, and only tests use them. The Site registers the root's 80 bytes under a new scheme id that names both levels. With the signing key's certificate carried once in the evidence, a certificate carries and costs nothing more; an answer costs 34,836 B and one LMS check more for each signing key in it. It changes (ii)'s state as the second line says. About 300 lines of Python and 130 of Lean. Open: what `deployment` its party-key message binds, which must not move with the head.
- **The device's pull request as the one change** [argued sizes]: about 260 lines of Python, 250 of tests and 110 of Lean. The Lean is the Lean device's mirror, which the gate does not run, so it can follow. The files, and what the tests inject, are a table in the note.
- **The clock is not one argument** [read]. The Warden's clock is injected, but `precise_sleep` waits on `loop.time()`, the pacer's process reads `time.monotonic()` six times against a `due` on the Warden's clock, and `serve` sets the origin from `time.monotonic()` and passes the Warden no clock. `warden.py:391` is the client's shim, not the device; the device's is `:500`.
- **On the raw clock a tick can run early** [argued]: no sleep runs on the raw clock, and `run()` ticks after one wait whatever the clock then reads. The pull request needs the tick to wait again, with a test; nothing tests an early tick today.
- **No host model for the certifier and no clock tolerance is in the tree** [read by search]: six cards on `main` now, none the certifier's. The tree names no path for the developer to set the device's rate; what time source the host follows is outside it.
- **A rational step works** [ran]: try `r20261112-111326-0840`, exit 0, on a scratch commit never pushed. A step of 9999/10,000,000 s gives 59,994 ticks for 60,000 steps and 3,199 for 3,200, rounded down.
- **Not established:** nothing is measured; the line counts are estimates; I did not read the proofs that name `signs`.
- **Slip:** one search printed unmasked showed a node's name in my own run; it was sent nowhere.
- `main` moved to `33866d93e66e` while I read. Of the files I read only the hardware cards changed, and I read that diff. #9197 is still open at `368902c12484`.

[2026-11-12T11:33Z] fddf306dafda
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 3:34 AM PST. Job 5 part 3, ahead of Fri 13 Nov noon PST.

**Part 3 is built and passes: branch `cursor/network-accounting-okbit-constant-4e7a`, head `60b8a8560f28`, ten commits on lean's landing tip `f3837f57d113`. Not opened as a pull request, no lock `--update`. No STOP: the constant is computed from the schedule's T and fixed tags alone. The equal-roots case ran against the built Lean and passes.**

What it does: a public `o0` is committed under public salts (`scan_ok.salt`); the period check computes each member's ok root once a `check` from T (`ScanOk.consts`: `first`, `ingress-0`, `last`, `egress-0`) and, last in its findings, refuses an execution phase whose scan's registration names a `<member>/o0` at another root, leaf count or domain, names none for a member, or names another port ending `/o0`.

Runs, each launched once:

1. `r20261112-103307-0ce8` at `a930b03c6919` [ran]: `lean_changed.py --try` on `ScanOk.lean`, `Period.lean`, `NetworkCheck.lean`, `security/Proofs/Properties/NetworkPeriod.lean`, `CoreOnsite.lean`: each `lean exit 0`; the proofs' 43 theorems, no axiom beyond the three. `ScanOk.lean`'s guards hold Lean's constants at T = 16 and T = 3 to the hex Python printed. (`r20261112-100746-28b4` at `5bfa1b25b4`: the same on three files.)
2. `r20261112-103307-baca` at `a930b03c6919` [ran]: `suites.py --changed f3837f57d113 --quick --fresh`: 83 suites, 83 passed, 0 cached, 0 failed. (83 because the domains table gained two entries.)
3. `r20261112-104617-8275` at `b2f2658fda25` [ran]: the certifier verifier's two suites whole, slow tests included, each outcome kept: network 97 passed, verifier 1360 passed and 1 opt-in skip.
4. `r20261112-111640-75ed` at the head `60b8a8560f28` [ran]: `Period.lean`, `NetworkCheck.lean` and the proofs exit 0 again; the network suite whole, 97 passed. The head differs from run 3's by one bound variable's name.
5. `r20261112-103307-25cb` at `a930b03c6919` [ran]: `scan_zk.sh` at the test class, one execution phase a registration (T = 16): rc 0, honest round drawn 32 of 32 and accepted; the honest registration as the verifier received it names `first/o0`, `ingress-0/o0`, `last/o0`, `egress-0/o0` at exactly the computed constants; held; not held with one root moved; the unread round's (ok bit 0) not held.
6. Ratchet gate from the landing tip at the head [ran locally]: passed.

**The equal-roots case** (T = 3: `ingress-0` and `last` at one domain and root): `test_two_members_at_one_root_are_each_held_to_it` `PASSED` against the built `network-period` in runs 3 and 4: both accept a registration naming both ports at the one root with one public, and both refuse, by name, each way of breaking it, a registration naming one twin and not the other among them. `ScanOk.lean`'s guard runs the same in Lean alone.

**The cost ruling**: `test_network_period_ok_cost.py`, 4 `PASSED` in runs 3 and 4. The mirror is counted as it runs: the constants once a period, the same SHA-512 count at one, two and three execution phases, zero in every comparison, and a refused period costs the same. Lean is read by the test: `ScanOk.consts` is named once, outside the function mapped over the execution phases, and that function and `held` name nothing that hashes. The count: `ScanOk.calls 32 = 524`, 512 for the trees and 3 a member for its domain [argued from the code, pinned by guards, not instrumented].

Five things for you:

1. **It holds the registration's `roots`, not a member's file.** The period check reads no member's registered file. A member's statement's `o0` root is held to the registration's `<member>/o0` by the registration check, and that entry to the constant here. [read] in `scan_zk.audit`; I did not re-read the Lean of that tie. Say if soundness meant the member's file itself.
2. **Member names are matched as text** in `scan_ok.members` and `ScanOk.members`. They move with proofs' naming rule; a test compares the table with the staging's.
3. **Public salts.** PoUW's ok row is committed the same way and `ScanOk.root` uses the same known-rows code (`Flock.RegKnown`), but `Flock/Firewall/Salts.lean` is worded for salts under the ZK server's key. For proofs: is a known row outside that rule?
4. **Every scan registration staged before this is refused** by the period check, its `o0` being under the prover's salts [ran locally on a kept run's files]. The table is one execution phase a registration's; a group of two is refused.
5. **No property statement.** `RowsOkHeld`, in `RowsStateTied`'s shape, is written out in the note and not built: it needs a lock entry. Say if you want it stated and proved on this branch.

Also: `r20261112-091349-288a` ended, 7 suites, 7 passed, 0 failed at `4a07fca408fd`; it is in #9228's body. #9228 is open, unmarked, head `c4a83b814a6f`.

Note: `20261112T1105Z-report-net357-part3-okbit-constant`. Not looked at: the Lean check (`check.py`), `circuit-check`, a scan on V3c's trees, PRF-586's branch.

[2026-11-12T11:34Z] 441aeea0e773
CERTNEG to the network-accounting coordinator, job H (NET-366), Thu 12 Nov 3:55 AM PST. A plan line, no number in it; nothing is launched yet.

**Built and pushed, unmarked, no pull request yet:** branch `cursor/network-accounting-scan-cost-06b2`, head `51929f9d5f`, on `main` at `33faab2be8`. Two commits: `scan_zk.sh`'s new option, then `benchmarks/scan_cost/` (`run.sh`, `result.py`, `public.json`, README, a suite of 20 tests, the two rows in `benchmarks/README.md`, the two metric rows, the directory in our `owns`). [ran] the suite's file, the new `scan_zk.sh` test, and the root invariants that read these files, by name (182 passed in one run; 60 at the pushed head), all pass here, and no warning names my files. The pool run of the changed suites goes next.

**Five choices you did not rule on, each as built. Say if any should be otherwise before 5:30 AM PST; after that the baseline runs as below.**

1. **Cores: 21.** [ran] The network pool's slots give 21, 23, 24, 48, 180 or 256 cores, so 21 is the count every slot can give. `run.sh` pins the job to its first 21 cores and refuses a job with fewer. The count is in the identity (`cores`, read from the scan's own `host.txt`).
2. **"On one core" is made true, not assumed.** [read] The Lean verifier's code spawns tasks; my proposal's "one core" rested on telemetry (CPU seconds about equal to wall). So `scan_zk.sh` gains an opt-in `SC_LEAN_ONE=1`: each `flock-verify` process is pinned to the job's lowest core and leaves its command, core, exit, wall and CPU seconds in `$OUT/lean.jsonl`. It is a wrapper around the binary, like `SC_RSS`; `scan_zk.py`, its sessions and its grouping are untouched. A scan without it has no verifier number, and says why. The ZK server's Lean is not wrapped.
3. **The group is outside the identity.** Identity: class, sample (every unit), T, tick, source, cores, verifier cores. The group is a setting beside it, and both numbers divide by G × T × tick, so the G = 8 run compares with the baseline. [read] `_defs` takes the class only, so the pinned public part at group G is the pin with G times the blocks; [argued] no group above 1 has been run against it.
4. **The pin has one more key than the 9 Nov answers.** [read] Today's code answers `rests_on_period_check: ["prog"]` in the public part; `public.json` holds it. [ran] The test-class pin equals the public part of this morning's every-block run, and the w63k4 pin equals the 9 Nov w63k4 answer plus that key.
5. **The tests' fixture is in a kept run's form, with stand-in figures.** You ruled "`tests/` on a kept summary"; a kept summary is a measured result, which the tree may not hold. The test builds files with the summary's keys and the logs' lines and round numbers of its own. [ran] The same code on the two kept runs' real files (outside the tree): each session's five stages sum to its process to 0.1 s.

Smaller: the metric rows carry no `goal` key (optional; say if you want `network/period-bound-checked` there). The guard also fails a result whose scan exited 0 but whose answer differs from the pin; then no measurement is written and the run's validation fails.

**Next, in order:** the changed suites on the pool now (`--on auto`); the pull request by `research queue open`, unmarked; at 5:30 AM PST one rehearsal of `run.sh` at the test class in a research-pool slot (minutes, 16 GB), which proves the pin, the wrapper and the writer on a node; then the baseline, launched once, `--pool network`, from a clone, 128 GB, `--timeout 10800`. I report the launch, and the numbers when it ends.

[2026-11-12T11:37Z] 24dbc1fd0614
TERMGB to the network-accounting coordinator, Thu 12 Nov 3:37 AM PT. Job G, the first pull request: open, unmarked.

- **Pull request:** [#9262](https://github.com/compute-verification/verity/pull/9262), branch `cursor/na-prove-path-rates-b4e6`, head `7aec9b3cb828af69dc6e7e5e6eff6cf7a3627719`. `main` `33866d93e66e` is merged in. Two files: `benchmarks/prove_path/run.sh` (+16/−6) and the new `benchmarks/prove_path/tests/test_run_rules.py` (128 lines).
- **The stand-in's run at the head [ran]:** `r20261112-112219-d7fa`, `research run --on auto --cwd clone`, launched once at that head: exit 0, 416 s. Honest: `ok: true`, both ranks accepted, 4 of 8 sampled on the challenger's coins. Forged: `ok: true`, all 8 sampled, the four checks true. Both `PROVE-PATH` lines are in the body whole. `served_tp --law rate:1/2 --deferred` commits.
- **The guard [ran locally]:** 2 passed in 0.39 s with the change. On `main` `33866d93e66e` as it is: 1 failed, 1 passed in 0.31 s, naming `subset:4`, `subset:8`, `stratified:4`, `stratified:8`. It lists all 17 partition files.
- **How I searched [ran]:** every tracked `.json` on `main` (1,085) parsed; a partition file is a top-level object with `templates` and `law`: 17, your 17. Seven other JSON files hold a `law` key and are not partition files (five record fixtures, one schema, PoUW's `window.json` vector).
- **Suites [ran]:** `r20261112-112257-0a4b` at the head, exit 0: `benchmarks/prove_path` 152 passed, `repository` 346 passed, `verity-graph` 210 passed.
- **Merge-trees at the head [ran]:** against `main` clean. Against lean's landing tip `f3837f57d113`: 24 files, 23 of them that branch's own against `main` alone; the one that is this change's is `benchmarks/prove_path/run.sh`, one header hunk (its `PP_WEIGHTS_TIE` paragraph sits directly above the two stand-in lines I rewrote; keep both).
- **Heads and runs before this one:** the head moved three times after my first runs (the adapter import mended, the list grown to 17, `main` merged twice), so the stand-in ran three times in all, once a head: `r20261112-105419-2423` at `a9c422221a48`, `r20261112-111205-1f96` at `44d352275d5c`, and the one above. All exit 0. I cancelled `r20261112-110428-dcff` (my own, at `d67c977e51f4`) while it waited for a slot. One plain run beyond the order, `r20261112-105545-50be` (`PP_PLAIN=1`, at `a9c422221a48`): exit 0.
- **The ratchet gate** fails for 6 retired-term sites in the new test, all the code's identifiers or the script's variable; off the merge gate. No allowlist touched.

Least sure, and smells:
- **The forged `--zk` run's `lean_refused` rests on an absent proof:** the ZK server released none, and the Lean verifier refused for want of the session's file. Under `PP_PLAIN=1` the Lean verifier read the proof and refused it at its lincheck. Not this change's code.
- **Lean's landing branch brings a fifth rule text** (`PP_LAW`, default `work:1`, the `PP_WEIGHTS_TIE` arm), which the guard's first test does not read.
- **One slip:** a machine's instance name appeared once in my own tool output (a snippet masked one name pattern, not the other). It was not posted or written to any file; every later snippet applies both masks.

Next: the table with the scan's, the period draw's and `served_commit.py`'s rows by 7:30 AM PST; then the second pull request.

