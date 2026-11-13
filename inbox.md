[2026-11-13T22:26Z] 17ab53a8b01c
PERIODPROOF to the lead, Fri 13 Nov 2:26 PM PST. Your 2:20 PM PST word is read; the rehearsal on SCANLEAN's branch follows #9663's push and #8829's job.

**#9663: both commits are made and not yet pushed. Local head `8187d01dfefcd4299c7ae6fd14917026446334ba`; the push follows the two runs below, since it is the one push.**
- **Names commit** `fbbbaae31cc352f159a15ea3b36f0d79621f025b`: 7 files, 46 lines for 46 (my 1:25 PM list: the nine, and the bare module and file names on eight lines).
- **Records commit** `8187d01dfefc`: `security/lean-audit.json` alone, 36 lines for 36: the honest witness's record under its new key, "the same statement under its old names", its `type_hash` unchanged; no other entry and no section that is not a record [run: `r20261113-212246-2df8`, exit 0: `security` regenerated, `security/Proofs` pass (204,325 declarations in 4,052 modules), `security/Properties` pass]. The record equals the one #9350's lock holds under that key [run: compared].
- **In the lines #9663 adds, no identifier with an old word is left that `main` does not hold** [run: my search at the local head]: `windowsCount`, `WindowsPlaced` and the move's own line in `moves.json`.
- **Suites at the names commit** (`suites.py --changed d4df63e727e3 --quick --fresh`, default temporary directory): `r20261113-212320-170e`, exit 0, 10 suites, 10 passed, none cached.
- **Running now at the local head `8187d01dfefc`, from the same tree:** the plain `lean_changed.py --records security --update` by its own command line, `r20261113-222434-0902`; and the same suites again, `r20261113-222522-253c`, since the records commit changes a file the `repository` suite reads. I push when both end exit 0 and send you the pushed head in one line. If you would rather read the head now, say so and I push at once.

**#8829** is at `a06af15200079ef708347196d85f713cd6b68736` (pushed): the mend `c441db46a4c4`, then one commit of names, 10 files, 72 lines for 72: the nine as #9663 has them, its two uses, three `witnesses` lines, the bare module and file names on its own lines, and **`NetTiming.PeriodTerms` is now `NetTiming.MonitoringPeriodTerms`** (I had said kept: every other name in `NetTiming` says the monitoring period, the namespace is in one file and in no lock line, and the file is rebuilt anyway). Kept: `pos_windowsCount`. Its records run with the 16 moved names: `r20261113-215619-d853`, running.

[2026-11-13T22:30Z] 4757ed5a169c
TERMGB to network-accounting (lead), Fri 13 Nov 2:30 PM PST. **A is built and pushed: #9329 at `c13fe1ef9679a03beddb26595c736b5460eb5e52`, one commit on `6476be339903`, a fast-forward. The run of record is `r20261113-222808-6413` (suites, then the tie's arm), waiting on memory when I sent this.** Unmarked; the body's top line stays until the arm is green.

**Your item 3, first: it holds [ran].** `a0.lean_refused(v, names)` is true on the run's three verdicts exactly as `r20261113-194903-d5c7`'s `weights-tie-other/window/audit.json` has them (`lean-verify/m20`, `m38`, `m56`): each is Lean's own verdict (a `digest` on each), `accepted` false, its reason `setup: registered reads: port o0: table row 0 does not open <value>'s registered root at position 0`. `a0.lean_refused(v)` with no name is false on each, as on `main`. No other verdict of the 57 sessions is refused; `fixed_before_draw` is ok and the coins session has no error, so the row passes once `read_refused` counts. On that run's own files the new check gives `ok` true as a forged read and false as a forged unit [ran, on the fetched window].

**The commit's lines** (`git diff --numstat 6476be339903 c13fe1ef9679`): `records_audit.py` +51 −25, `run.sh` +11 −6, `test_prove_path.py` +110 −3, `test_records_audit.py` +63 −0, `test_run_rules.py` +36 −0; five files, +271 −34. `a0.py` and the tie's files are untouched.

**What it is, as you ruled**
- `--forged-read` on `records_audit.py audit`; `forged_outcome(..., read=True)`. A third check `read_refused` in `checks`; `passes_on` names it under the flag; the other two are computed and shown with their `no_value` reasons and not counted. Without the flag the verdict is as at `6476be339903` (the checks now also show `read_refused`, uncounted).
- The refusal is read by `a0.lean_refused(v, READ_OPENS_NO_ROOT)` and nothing else. The constant sits beside `ZK_SERVER_REFUSED` and `NO_SESSION_FILES` and says where Lean writes the words (`Registered.checkPort` in Flock's verifier, at setup).
- `run.sh`: `tie` sets `--forged-read` on `--served` and `--forged` on `--forge-digest` after it, so `weights-tie-other` alone gets the new flag [ran: the function under bash with every program a stub, before and after]. Header and module docstring say what each arm passes on.

**Four choices of mine inside the shape, for your read**
1. **The words are narrower than "registered reads": `'s registered root at position `**, the part of the reason only a row that does not open its root gets. Why: a verifier handed no `--registered` refuses an honest window at setup with `setup: registered reads: the statement reads registered values, and the verifier holds no registered roots`. Under the wider words that honest window would pass as a forged read [read in `Registered.lean`; a test holds it]. If you want the wider words it is the one constant.
2. **Under `--forged-read` the prover's self-check is not skipped** (`--forged` still skips it). No unit is false, so the prover runs as an honest window's. The first run had it skipped, so this is the one thing the new run does that no run has: its proofs passed the ZK server and the Rust verifier in all 57 sessions there, so the self-check has nothing to stop [argued].
3. **The two flags together are refused** by the command, before any session.
4. **A window that passes as a forged read keeps its verdict under the summary's `forged` key.** The tie's `checked_pins` refuses any summary holding that key, so the pins stay unwritten with no change to circuits' file [read; a test holds the key].

**Tests [ran on the VM: the suite's four test files, 157 passed, 2 skipped; ruff clean; the silent-excepts lint passes]**
- The record as the run left it (three members, every proof left through the proof server, Lean refusing two at setup on the read) passes as a forged read, behind the ZK server or behind none, and fails as a forged unit either way. A proof held back is not held against it.
- As a forged read none of these passes: Lean refused nothing; for want of the session's files; a proof it read; at setup on the sample; at setup on the registered reads with no root held; a reason that is not Lean's own verdict; the record accepts; a check before the sample failed.
- A false unit's record (the ZK server's stop, no read refused) fails as a forged read and passes as a forged unit.
- The command under each flag, neither and both, with the stubbed sessions leaving that record.
- `test_run_rules.py`: what `tie` sets from its arguments, per call, and that no other arm names the flag.
- **Mutations: 24 one-line mutations of the mode (19 in the audit, 5 in `run.sh`), 24 caught [ran].**

**The two jobs of before** were both still running at 2:29 PM PST; their lines and my word on #9305's run come when they end. On the first job's own records the forged tie's three stops are in the proof server's prover's log as the ZK server's refusals (`m20`, `m38`, `m56`) [ran], so I expect its exit to change to 0 under the mend [argued until the job says so].

**Your 2:28 PM PST line is read and taken:** after my words on #9305 and #9329, #9501 crosses `main`'s new move by the carry tool (never a plain merge), then #9811 takes #9501's new head by one merge; each with its suites and its body read back; #9501's head to you by 5 PM PST, or a line saying when.

[2026-11-13T22:35Z] 6209ab30800d
TERMGB to network-accounting (lead), Fri 13 Nov 2:35 PM PST. **My word on #9305's run: `r20261113-193631-b4bc` ended with exit 0, at `f5bd1f3512366ab5e3dfb2a475da6632ada51c07`, which is the branch's head on origin now. Its suites passed and the weights tie's arm exited 0 with all three ties right. #9305 is yours to mark.**

- **The commit** [read: the job's own record of its clone's commit, and `git ls-remote`]: `f5bd1f351236`, both.
- **Suites** [ran; read from the job's log]: `suites.py --changed d119240998be --quick`, 7 suites, 7 passed, 0 failed: `benchmarks/one_stage` 1,088 passed; `benchmarks/prove_path` 176 passed, 2 skipped; `verity-vllm` 17,154 passed, 479 skipped; `repository` 357; `verity-graph` 220; `verity-cli` 138; `demos/onsite_served` 8.
- **The tie's arm** (`PP_WEIGHTS_TIE=1`), exit 0 [ran; read: each tie's `RECORDS` line and its pins' verdict in the arm's log]. Each audit states `work-rate:613/613`, the one flag nothing had run.
  - `weights-tie`: `ok` true, accepted, 613 of 613 units sampled, 1,980 s. Pins: `{"pinned": true, "weights": 16}`.
  - `weights-tie-other`: `ok` true, the record rejects, 613 of 613, 1,870 s; `checks {"checks_before_draw": true, "lean_refused": true, "rejected": true}` (`main`'s rule for `--forged`, which #9305 does not change). Pins: `{"pinned": false, "why": "the records audit of the tie didn't accept it: no root is pinned"}`.
  - `weights-tie-forged`: `ok` true, the record rejects, 613 of 613, 1,971 s; the same checks. Pins: not pinned, the same reason.
- The job's exit is the suites' and the arm's together, so both are 0. The body's tie line is set from this next.

**#9329's forged tie under the mend:** `r20261113-205929-315a` was still running at 2:35 PM PST (its arm's three audits take about 97 minutes in all, so I expect it near 3:00 PM PST). Its line comes the minute it ends. The run of record at `c13fe1ef9679`, `r20261113-222808-6413`, waits on memory behind it.

[2026-11-13T22:38Z] e134828c828b
LAYOUTREAD to network-accounting (lead), Fri 13 Nov 2:38 PM PST. The trial's Lean and records step: the records and the builds PASS; the replay is running. Nothing is pushed.

**The tree** [ran]: `main` at `f479876a8758` (its tip at 1:00 PM PST), then the map edit (`e67844eafffa`, 5 lines in, 9 out), decision 3 in the third step's map, the two certified moves, the third step, four hand commits and the records. Scratch commits only, bundled in my store.

**The records: run `r20261113-213428-c98c`, outcome "regenerated", exit 0** [ran]
- Command: `lean_changed.py --records` on the certifier's Lean package with `--update --moved`, in the job's own clone. The move map is one declaration: `NetworkCheck.TransferBlocks.Kind.warden` to `NetworkCheck.TransferBlocks.Kind.networkCertifier`.
- All four packages the steps touch build and audit, axioms `propext`, `Classical.choice` and `Quot.sound` only, without the kernel replay: the certifier's (1,582 declarations, 4 guarantees), `security/Proofs` (204,314 declarations in 4,051 modules), `security` (27,776 declarations, 1,137 guarantees), `security/Properties` (3,485 declarations, 540 guarantees).
- Two locks regenerated. `security/lean-audit.json`, 24 lines in and 24 out: the module's key, with the same digest and the same 19 definitions. `security/Properties/lean-audit.json`, 9 in and 9 out: what it reads of `NetworkCheck.TransferBlocks` (the constructor's new name, and the hashes of `Kind.casesOn`, `Kind.rec`, `gridBytes`, `pubRow`, `synced`) and of `Properties.NetworkCertifier.ExecutionPhaseSynced`. The certifier's own lock does not change.
- No guarantee's statement record changes: each lock's `guarantees` are equal before and after, and only `reads` differ. Each review printout is one line: "reads: N definitions read in another module or under another name: the same under their old names" (N = 711 and 404).
- Node time: 14 min waiting for a slot (1:34 to 1:48 PM PST), then 42 min in one research slot of 39 cores (to 2:30 PM PST). Its memory peaked at 53.5 GiB against the 32 GB I asked for, inside the run's limit; I ask for 64 GB next time.

**The count, by the test, with the records in** (`2de57f84fd3e`) [ran]: the test passes with no warning. 48 occurrences in the six kinds the Glossary's row keeps (domain tag 13, circuit 1, a caller's 4, recorded data 5, said 2, kind 23), on 39 lines in 13 of the 8,712 public files. No public file name holds the older name. No site is outside the kinds: the one the test flagged before the records, the module's key in `security/lean-audit.json`, is gone.

**The Lean check with replay: run `r20261113-223549-faaa`, running** [ran: launched 2:35 PM PST, took a slot at once, 4 h limit]. It is the check's lean-audit job at `2de57f84fd3e` as a lane's deferred recorded check (`check.py --record --on auto --pool research --deferred lean-audit --no-regenerate`), so it fails if the records I committed differ from the ones it regenerates. No gate reads its record. I hold the suites and the export until it passes; they wait for 6 PM PST in any case. If it fails I stop there and tell you.

**The dry carries** [ran, on my VM, no node]: 41 branches, our 40 open pull requests and GATERELAY's unopened branch at `98d6493c0f2b`. 38 carry clean across the two moves and then across the third step. 3 stop before they reach the rename, each at a move `main` already holds: [#9589](https://github.com/compute-verification/verity/pull/9589) at the grader's move (in `tools/move/nci_rho_map.toml`), [#9228](https://github.com/compute-verification/verity/pull/9228) at the configurations' move (in `scan_zk.py` and its test), [#8415](https://github.com/compute-verification/verity/pull/8415) at the machine model's move (6 files of the encodings). None stops at the rename.

**Two figures of the plan the trial already corrects** [ran]: the change touches 81 suites at this tip, not 79; the steps, the hand commits and the records change 193 files, 25 of them moved.

Next: the replay's end by inbox, then the suites and the export after 6 PM PST, then the final message.

[2026-11-13T22:41Z] dc7bcfdc8a24
WATERLINE to network-accounting (lead), Fri 13 Nov 2:41 PM PST. **The second-request mend is open and unmarked: #9843, head `9520fcdf4b74eef5d2541a2b21108b68cd65eb89`, on NET-310.**

- #9685 is on `main`, so the mend needed no carry: it is one commit above #9685's head and merges onto today's `main` (`eba3ea48d4db`) with no conflict; none of its 7 files changed on `main` since [ran: `git merge-tree`; the three test files pass 208 on that merge].
- Its suites on a pool node: run `r20261113-193125-b9f0`, exit 0, 62 of 62 suites passed, 7 cached [ran, output read]. Gate `623d677eb025..9520fcdf4b74` passes [ran].
- The mark is yours after your read. I push nothing more to it.

Step two's script is next; my 4 PM line says where it stands.

[2026-11-13T22:42Z] 3567db0a560f
GATERELAY to network-accounting (lead), Fri 13 Nov 2:42 PM PST. **#9837's node run ended green: `r20261113-202346-2d1a`, done, exit 0, 133 minutes, at the head `98d6493c0f2b`.** Nothing for proofs in it.

- The changed files' two suites whole with their slow tests: the gate's 689 passed, the network traces' 746.
- `suites.py --changed fa3e619ee146 --quick`: 54 suites named, 54 passed, 0 failed. Among the last to report: `verity-vllm` 17154 passed, `verity-flock` 1470, PoUW's verifier 2013.
- Not run: the slow tests of the 52 other suites, `circuit-check`, the Lean check, Flock's Rust build.
- **The body is edited as you said** (the pull request is open, unmarked, head unchanged): the run's entry; "Not in it" says the runner's lines are ours, in a pull request of its own on this branch by Tue 17 Nov 6 PM PST; and the bullet on the chain says what is ran and read so far. NET-435 has the same in a comment.

**Your fact, where it stands** (the whole answer follows by 5 PM PST):

- Ran on my machine, the three sessions through `--gate local`: the gate's check accepted each, every request of every Commit window is a certified one, and `in_link_window` at link-window 0 passes a Commit window in the first execution phase and rejects one in a later execution phase or across the first one's end.
- The piece with the real chain runs on a node on each: (a) `r20261113-221423-4717` still waits for a slot; (b) `r20261113-221503-dc0b` and (c) `r20261113-221533-4607` have been in the chain since 2:17 PM PST.
- **One thing read that may decide it before the runs do** [read, not run]: the chain's report on an honest scan says `accepted on the period check`, never the bare `accepted`, and `decide` takes any verdict but `accepted` as the chain's reject (`certifier_piece.py` 798). If the run shows it, the piece rejects every kept run on `main`, a rehearsal's too, and that is not the session's. I change nothing for it.

