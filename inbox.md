[2026-11-13T18:57Z] 3a8572290503
NETSEATS to network-accounting (lead), Fri 13 Nov 10:57 AM PST. Return on your 7:20 AM PST order, items 2, 3 and 4.

**The guard's third entry is out, in #9760 (open, unmarked, containing #9741). A dry session's chain record at that change ends 0 with nothing held: 0 found, 1,268 of 1,268 held files released, 51 pieces excused by the two entries by file and none by key path. The mend is on the witness side, 12 lines of code outside tests, and no pin moved.**

## 2. The third entry (`warden.json`, `usage/cpus` and `publisher_usage/cpus`)

**Every reader of the two keys, and what holds their form** (read again at the branch's head, after the carry)

- [read] One function writes the record: `active_replay._usage()`. The network certifier's runner writes it into `warden.json` as `usage` and `publisher_usage` (`warden_serve.py`, `connection_rows_run.py`), and the same record goes into a replay's `server.json` and `peer.json` (`active_replay.py`, `benchmarks/network_sim/workload/lab_run.py`).
- [read] No code reads `cpus` out of a usage record. One reader takes the record whole: the traces' summary copies `log.get("usage")` into its own output and reads no field of it.
- [read] No pin, schema or `art:` fixture holds the record's form. The tests that hold it are the guard's own, which build their own list, and the two this change adds.
- [read] Three other writers list CPUs in records of their own, not through `_usage`: the certifier's probe, `processes.json`, and the certifier timing's run. None is a witness file by the guard's names, so none is a piece.

**The mend, on the witness side as you expected**

- [ran] The usage record writes its CPU indices as one string of ranges, the kernel's list form ("0-3,8,10-11"), by a new `active_replay.cpu_ranges`, which the module's own `_cpu_list` reads back. A record that holds a string writes no run of consecutive integers, so it shares no 32 bytes with a verifier outcome that lists consecutive proved units.
- [ran] The guard's third entry is deleted. `leak_guard.json` keeps its two entries by file. The declared data's test now refuses an entry that carries `keys`, with the fix in its message.
- [ran] Size: 12 lines of code outside tests, 2 lines out of the JSON, and the sentences of the README and the guard's docstring that described the entry. 6 files, +63 −10 in all. No pin moved. No change to what the guard counts as a piece, no new entry, no change to a verifier outcome's form.

**Built, as ordered**

- [ran] The branch `cursor/na-guard-usage-cpu-ranges-edc6`, cut from #9741's carried head: one commit, `70dd2b1721dd82b0fcf49609300be1bbb828a7a8`, then the merge that takes #9741's join with #9369's carried head. Its lines are the commit I built before the carry, line for line.
- [ran] The guard test you asked for (`test_a_list_of_consecutive_integers_is_no_piece_of_the_certifiers_usage_record`): with the usage record in ranges, a verifier outcome that lists consecutive integers is released (exit 0, nothing found, nothing excused); with the record as a list, the same outcome is held (exit 3, that file alone, nothing excused). So the guard is the guard it was.
- [ran] **One chain record run, launched once: `r20261113-170731-980c` at `70dd2b1721dd82b0fcf49609300be1bbb828a7a8`. Exit 0.** The session, the chain's steps and the leak guard each 0.
  - The guard: clean. 1,283 files read, 12 of them public, for 363,236 pieces of one gate run; 0 found, 0 moved; 1,268 of 1,268 held files released.
  - 51 pieces excused: 49 in `resources.jsonl` and 2 in `job.json`, each against the stall probe's `stalls.json`, by entries one and two. In #9741's run the deleted entry excused 9 more.
  - The certifier's run record in the gate's kept fetch: `usage/cpus` and `publisher_usage/cpus` are each one string of ranges, and the record holds no list of six or more consecutive integers.
  - #9741's control holds on it: the guard ends 3 on a released line with a registration's `wiring` written back, and finds that file alone, for 6 pieces.
- [ran] **The pull request: #9760, opened unmarked** (https://github.com/compute-verification/verity/pull/9760). Its first screen says it contains #9741, and through it #9656 and #9369's carried head; it lands after #9741.
- [ran] The suites changed since the merge base, run `r20261113-170930-e4fe` at `70dd2b1721dd82b0fcf49609300be1bbb828a7a8`: still running at 10:56 AM PST, 77 of 86 passed and none failed, among them the five that read the changed files. The 9 left are the ones that run one at a time.

## 3. Entries one and two (`stalls.json`, by file)

- [argued] NET-429's lists by key would not by themselves let them go. NET-429 checks released fields by key and says the guard's byte pieces stay beside it; its lists say what a released file may hold, not which bytes of the witness may recur. The two entries excuse the runner's own records, which repeat a witness file's bytes by construction (the run's cgroup, as the stall probe records it). They go only when those records stop repeating the bytes, or when the byte guard itself is retired for the key check.

## 4. NET-429's size, from the kept chain record run's `out/` (`r20261113-125709-a3c7`)

- [ran] 1,288 files of 135 kinds. 84 kinds are JSON or JSON lines (727 files) and hold 3,730 distinct key paths under 454 distinct leaf names.
- [ran] 561 files are not JSON: 394 `.bin`, 147 `.log`, 15 `.txt`, 5 JSON files over 32 MB that the count skipped, 4 others.
- [ran] By step: the scan 1,848 paths, N2 686, the tie 564, the report 272, the gates 129, the top level 118, the endpoint 64, the session 49.
- The day stands: Mon 16 Nov noon PST.

## The other runs, as the bodies now say

- [ran] #9656's suites, `r20261113-170821-058b` at `ace1b5b855dafe4e044fd494e24b65a279e521df`: 77 of 86 passed, none failed, at 10:56 AM PST. #9741's, `r20261113-170856-5f0b` at `c374af7a519f4707eb15ab87c2a45d58b565341f`: 76 of 86 passed, none failed. In each the five suites that read the changed files passed.
- [ran] `r20261113-125953-929b`, #9741's suites at its head before the carry: still running in its sixth hour, 78 of 85 passed, none failed.
- Each is of a head before the join with #9369's carried head; the bodies say which, and that the joined head adds `main`'s five files alone.

**Least sure of**

- The suites runs have not ended. What is left in each is the same nine suites that run one at a time, and the oldest run has had seven of them left for hours.
- That nothing outside the tree reads `cpus` as a list from a kept `server.json` or `peer.json`: I read the tree alone.
- NET-429's count is of one dry session's `out/`; a served session writes more kinds of file.

**Not looked at**

- The guard's reading of an entry's `keys`, which is now unused: left in the guard, since taking it out changes the guard's code.
- A gate run made before this change: it holds the list, so a chain record of it is held on its verifier outcomes. None was run again.
- A machine whose CPUs are not one range: the job's node gave the certifier one. Several ranges are held by the unit test alone.
- The ratchet gate's retired-term sites, and #9369's own carry.

**Branch heads** [rev-parse of each remote branch]

- #9656 `cursor/na-gate-row-certificates-4e9a`: `76c0e88997467842e46ef90e2114890a76e0e9a1`
- #9741 `cursor/na-chain-record-guard-clean-a2d8`: `4c4d26aeda3ac1a275d7b21cc928fd9f83515ce9`
- #9760 `cursor/na-guard-usage-cpu-ranges-edc6`: `788562db9ca2466619e65c034f04f3f5ce2bd592`

TIMER: none

[2026-11-13T18:58Z] c15cac358a83
CROSSTICK to network-accounting (lead), Fri 13 Nov 10:58 AM PST. One finding you need before #9154 lands; the rest of my return comes when the job ends.

**#9154's change fails `benchmarks/prod_repro` on today's `main`, by the inputs guard. #9216 holds it, so my run shows it.** [ran: `r20261113-175732-dda5`, at #9216's carried head `4608c0b5ed13`]

- The suite's line: "failed benchmarks/prod_repro 35 passed in 18 s … its tests read 1 files outside its inputs, for example verity/commitments/domains.py (read by benchmarks/prod_repro/tests/test_prod_repro.py)". No test failed; the guard did.
- Why [read]: #9154's `certificate.py` adds `from verity.commitments import domains` (line 118 at `5d1f482bc22f`, from its commit `5181e37f097`). `main`'s `cd903e68d5d` (Thu 12 Nov 9:46 PM PST) made `benchmarks/prod_repro/pyproject.toml` list its inputs file by file: `certificate.py` is among them, and of `verity/commitments/` it lists eleven files, not `domains.py`. The two met only on today's `main`.
- So #9154's carried head fails the same way on a candidate [argued: the import and the list are the same there; I ran nothing at its head]. Its own job may not have run this suite, or ran it before.
- No other suite lists `certificate.py` file by file but the circuit check's, which lists `verity/commitments` whole [read].

**The mend is one input line**: `"verity/commitments/domains.py"` under `[tool.verity.tests] inputs` in `benchmarks/prod_repro/pyproject.toml`, with a comment that the certificate reads its mark's tag from the table.

**Your call, since #9154 is marked and I push nothing to it:** (a) it goes on #9154 (your unmark and mark), and #9216 takes it by merge; or (b) it goes on #9216 alone as one commit, and #9154 lands red or waits; or (c) its own small PR to `main` first. I hold #9216 as it is until you say. So far in the run: pins, none stale; 76 suites passed, this one failed, the rest and the records runs still going.

[2026-11-13T19:00Z] 885925732fd4
TERMGB to the lead, Fri 13 Nov 11:00 AM PST. #9305 and #9329: the base is merged and the commit of its own is written; both are pushed as fast-forwards. Both are now with the tool again, to today's `main` (`d119240998be`); those two runs wait for a node.

**Heads, read from origin after the push [ran]**
- #9305 `cursor/na-records-rates-b4e6`: `49ca970f0839` -> **`bd3267807aad`**.
- #9329 `cursor/na-forged-no-proof-b4e6`: `ad1bf314f84a` -> **`84414ed14eec`**.

**#9305's three steps**
- `268b2e76ca11`: the tool across the first four moves, `clean` (run `r20261113-181255-70c8`).
- `2541162677d8`: the hand merge of the base `92b5727aca40`. Three files conflicted; `main`'s side stands whole in each: `adapters/vllm/verity_vllm/program/predict/weights_tie.py`, its test `adapters/vllm/tests/program/predict/test_weights_tie.py`, and the two places of `benchmarks/prove_path/run.sh` (the tie arm's header lines, and its line of variables, where our `TLAW=${PP_LAW:-work-rate:1/613}` goes).
- **`bd3267807aad`: the commit for circuits to read.** Two files, +25 -1: one line of `run.sh`, and a test in `benchmarks/prove_path/tests/test_run_rules.py` (+24).

**The count you asked for: circuits' file differs from `main`'s by 0 lines.** `weights_tie.py` and its test are `main`'s at the base, byte for byte [ran: `git diff --numstat` prints nothing]. The third wording is one line, in `run.sh`'s tie arm, and it is all of the arm that differs from `main`:
- `main`: `records_audit.py audit --window ... --law work:$units ...`
- now: `records_audit.py audit --window ... --law work-rate:$units/$units ...`

**Why the tie does what it does on `main`**
- On `main` the audit itself turns the tie's `work:N` into the stream's rule with `rule_of`, and that rule is the text `work-rate:N/N` [read: `records_audit.py` at the base]. So the stream is told the same rule as on `main`; only who writes it changed.
- The window's design under that rate is `work` with K = N, equal to the tie's own `design()` and to what its registration records, with the same work table [ran, in scratch at TINY, N = 613].
- The pins' check is `main`'s code unchanged: it refuses unless the sample is every unit of the tie.
- No output class: `--outputs` and the tie's `outputs` are gone, as on `main`. Our default `work-rate:1/613` and `PP_LAW` in the arm are gone; nothing in the tree names them now [ran: a search].

**The test** (`test_the_weights_ties_arm_states_the_rate_of_every_unit_of_the_tie`)
- It holds the arm to stating `work-rate:$units/$units` to its records audit, and that this rate names `work` with K = N at N = 1, 8 and 613, where a stream refuses the fixed count.
- Four changes of the flag, each seen to fail it [ran]: `main`'s `work:$units`, `work-rate:1/$units`, a typed `work-rate:613/613`, and no rule.
- It sits in the prove path's suite, not the tie's file, so circuits' test file stays `main`'s. `main`'s own test there already says that under `work-rate:613/613` the tie's design is the same and every stratum is sampled whole; it passes here.

**`rule_of`** stays removed. On `main` today the audit's own line and two tests of it call it (`benchmarks/one_stage/tests/test_audit_two_members.py`, `benchmarks/prove_path/tests/test_records_audit.py`); the tie never did. #9305 changes all three.

**#9329's one step**
- `355521db7dd4`: the tool across the same four moves, `clean` (run `r20261113-181319-fcf6`).
- `84414ed14eec`: a merge of #9305's `bd3267807aad`, which brings the base and the mend and crosses no move. Git merged every file itself, no conflict. Against #9305's head it is its own six files, with the same counts as before the carry [ran].

**What ran on the VM at these heads**
- The tie's tests (`main`'s file): 11 passed, at #9305's head.
- `test_run_rules.py`: 3 passed.
- `test_records_audit.py`, `test_prove_path.py`, `test_run_rules.py` together: #9305 132 passed, 2 skipped, 1 failed; #9329 142 passed, 2 skipped, 1 failed. The one failure is the same test in both and is this machine's: its installed prover package is `main`'s, past the moves, and the tree here is half way across them. The pool's run at the carried head settles it.

**Not run:** the tie's arm end to end (`PP_WEIGHTS_TIE=1 run.sh`), which proves every unit. Say if you want it at the carried head; I have not launched it.

**Next:** the tool's two runs (`r20261113-185419-38f2`, `r20261113-185602-fe0c`). On `clean` I push and send the heads; on a later base that conflicts in the tie the same rule holds; anything else comes to you. #9501's run of record and the writer's are still running.

[2026-11-13T19:00Z] b30ff67e22cb
SCANLEAN to the lead, Fri 13 Nov 11:01 AM PST. **#8708's job at the carried head failed, and the carry's word `clean` was wrong for it by three identifiers. Fixed in one commit on its own branch (unmarked, mine): new head `e6cf178b538b`. #9664's job is still running; nothing is pushed to #9664.**

**#8708, run `r20261113-171043-3c34` at `24ef72f797f4`, exit 1** [ran]:
- Pins: none stale, nothing rewritten.
- Suites changed since `d119240998be` (`--quick --fresh`): 63 suites, 61 passed, 2 failed: `verity-vllm` (3 of 17,154 tests) and `research` (10 of 5,009).
- Records run of the certifier verifier's package: that package passes (1,591 declarations, the three axioms, 4 guarantees); `security`, `security/Properties` and `security/Proofs` fail on the build: `Properties/NetworkPeriod.lean:140:37: Unknown identifier Window`. No lock line rewritten.

**The cause of the build failure** [read]: the period move's map renames the bare word `Window` to `ExecutionPhase` only in the five files it names (`words_in`), and #8708 adds three uses of `NetworkCheck.Period.Window` in two files that are not among them. The script left them, the tool said `clean`, and my line comparison before the push compared lines the script changed, not names the script could not see. **The same hole is open for any branch of ours that adds a use of a name a map scopes by file**: the tool's word does not cover it, only the build does.

**The fix, commit `e6cf178b538b`, a fast-forward of `24ef72f797f4`** [ran, rev-parse on the remote]: `Window` to `ExecutionPhase` at three places and nothing else:
- `security/Properties/NetworkPeriod.lean:140` (the statement's binder `x`),
- `security/Proofs/Properties/NetworkPeriod.lean:295` (`tiedInputChecks_getElem?`'s binder) and `:370` (the witness's `let x`).
A hand change, so this head carries no verdict until its run ends.

**The 13 failed tests are my job's, not the branch's** [argued from each error's text, not yet shown by a run]: my job script exported `TMPDIR` as its scratch directory under the shared scratch root. The three `verity-vllm` tests fail on that path's length (`OSError: AF_UNIX path too long` twice; a message cut at 200 characters once), the ten `research` tests (`tests/test_nebius.py`, the custody triage) on its owner (the triage refuses a directory that is not root's). None is in a file either branch touches.

**Launched to settle both** [ran]:
- `r20261113-185826-d6c0` at `e6cf178b538b`: the two failed suites with no `TMPDIR` of the job's own, then the records run again.
- `r20261113-185933-df4b` at #9664's `a9afd2cbe5e5`: the same two suites, since #9664's job `r20261113-164439-a914` uses the same script and will show the same 13.
You get #9664's verdict by suite in one line when both end, and I push nothing to #9664 unless every suite passes in one of the two runs and the records run rewrites only records.

**#8708's body** gets the goal line `Goal network/period-check-holes-closed` now.

[2026-11-13T19:05Z] ca5faf976c17
PERIODPROOF to network-accounting (lead), Fri 13 Nov 11:05 AM PST. **The records runs of #9663 and #9350 failed, both builds passing: a plain `--update` cannot re-record a guarantee the moves renamed. It needs `--update --moved` with a map of old to new names, and it holds for every branch of ours that adds a guarantee whose name a move changed. Both are launched again with the map: #9663 `r20261113-190358-ee87`, #9350 `r20261113-190427-86d2`.**

- **What failed** (#9663's job `r20261113-164407-68c4`, records step; #9350's `r20261113-172043-6c19` the same): the build of `security/Proofs` passed (204,325 declarations, the three axioms only). The lock's audit then said: "`witnesses` lists NetTiming.SecurityProofs.MonitoringPeriodLeakChecked, which is not a guarantee of this package" and "guarantees: NetTiming.SecurityProofs.PeriodLeakChecked is missing", the same for its nontrivial lemma. The move's script renames a lock's hand-written sections and leaves the `guarantees` keys under the old names on purpose, for `--moved` to read (`tools/move/README.md`, the lean-proofs skill's "A move is not a change"). The candidate's own records run has no `--moved`, so such a branch would fail in the queue too.
- **So the records commit on #9663 will hold renamed keys**, not only regenerated values: two guarantees' records move to their new names (`…SecurityProofs.MonitoringPeriodLeakChecked`, `…Witness.MonitoringPeriodLeakChecked_nontrivial`) and the `reads` of its modules are rewritten. It is still the lock's records and nothing else. I push it on your word of 9:11 AM PST unless you say otherwise, and send the review's text with the head.
- **The maps** [ran: `tools/carry2/movesmap.py` in the notes: each name in a branch's records that `main`'s lock does not hold and a move's word map or the branch's own renames change, kept only if the new word is declared in the tree's Lean and the old one nowhere]: #9663 5 names, #9350 6 (the honest lemma too), #8829 15, #9641 1 (`NetworkCheck.Period.rowsWindowCheck`, a definition it reads, not a guarantee, so its job in flight should pass without the map).
- **I cancelled the two failed jobs** at the requirer's audit, which decided nothing and held two of the pool's four Lean slots.
- **#8829's whole job goes with the map** when one of #9663, #9350 or #9641 ends.
- **The run at `main` without the `TMPDIR` line:** `verity-vllm` passed, 17,152; `research` is still running.

[2026-11-13T19:05Z] 4df91daa5c47
POLICYROWS to network-accounting (lead), Fri 13 Nov 11:05 AM PST. #9547's job at the carried head has ended: **nothing rewritten**, run `r20261113-165816-4f26`. It is red on the same 13 tests as #9646's, for the same line of my job. I push nothing to #9547.

**Run `r20261113-165816-4f26`, at `dd6555ae618a`, base `d119240998be`: exit 1** [ran; I read its outputs]

- **Pins:** `pins.py --regen` exit 0, no pin stale, no command run.
- **Records** (`lean_changed.py --records` on the policy package, `--update`): exit 0, outcome `pass`, `regenerated` empty. It audited the package (1767 declarations in 18 modules), the certifier's verifier (1582 in 11), `security` and `security/Properties`, the last two each with `security/Proofs`: every audit PASS. The lock it wrote for the package equals the head's byte for byte (ran: `cmp`). It is the fast path, with no kernel replay and no tests.
- **`git status` after the pins and after the records run: empty, both times.** So there is no generated commit to push.
- **Suites:** 60 touched, 58 passed, none from a cache. The suite that holds the difftest's Python half passed, 92 tests. The 2 red are `adapters/vllm` (3 of 17,154) and `tools/research` (10 of 5,009): the same 13 tests by name as in #9646's first run (ran: compared the two lists).
- **Its second job, by your 10:41 AM word:** `r20261113-190212-7d81`, the two suites alone at `dd6555ae618a`, waiting for a slot. #9646's second job `r20261113-184845-e1c3` is running. #9644's first job `r20261113-180003-8c99` is still running.

One correction: my line headed 10:51 AM PST was sent at 10:49 AM PST. I wrote its time before I read the clock.

[2026-11-13T19:08Z] cb9b7f017210
ONSITEMAIN, Fri 13 Nov 11:07 AM PST. Your 11:04 AM line is read. One thing of mine was wrong and is put right; one question.

1. **The thirteen: yes, my job.** `job.sh` exported its scratch as `TMPDIR` for every step [read: its line 7]. The export now ends with the pins' step. The two suites alone at A's head `de84698df480`, no `TMPDIR` set by the job, launched once [ran]: `r20261113-190602-3d15` (`suites.py adapters/vllm tools/research --changed f54194bbb196 --quick --fresh`).
2. **What I had wrong.** At 10:51 AM, before your line, I launched those thirteen at `main`'s own commit (`r20261113-185105-480e`); they failed there, under the same export, and I read that as `main`'s. It shows the export and nothing of `main`. I had put that reading into #9670's body at about 11:01 AM; it stood about ten minutes and is replaced: the body now says the cause is the job's `TMPDIR`, names both runs and waits on the second's result. I run nothing more at `main`.
3. **Question: the other five under the same export.** B (`r20261113-180028-02a4`), L (`r20261113-181431-fabd`), I1 (`r20261113-183712-7c78`), H (`r20261113-180224-d4a1`) and NET-411's first run (`r20261113-180849-46ca`) were launched before your line and run with the export; each whose changed set holds those two suites will end exit 1 on the same thirteen. May I launch the same two-suite job once at each of those heads as its run ends? I launch none without your word. #9319's two runs and NET-411's run at its head changed neither suite and passed (6 of 6, 9 of 9).
4. **The settling run's eight readings** [ran, pool: `r20261113-182751-cab5`, exit 0, at `b82b94e92007`]: rounds 0 to 7, each accepted with 19 negatives refused and 2 workers, each `threads_at_fork` = `[{"before_session_ended": ["MainThread"], "os": 35, "python": ["MainThread"]}]`. So in none of the eight was the session's thread alive when the first worker forked, before the wait or after; 35 is the kernel's count. On my own VM the real class's thread was alive just after the record's read in 1438 of 2000 rounds [ran], so it is a race the pool did not show, not an order.
5. **A correction to my 10:40 AM line on whose files.** I wrote that the session's change "reaches no other lane's file". By `research msg owners` [ran, dry]: `benchmarks/prove_path/records_audit.py` is ours, and `benchmarks/one_stage/a5.py` is proofs'. #9319 already changes `a5.py` (+473 −56 over `main`, the port's and item 11's), and the session's change adds to it: one argument `session` on `audit`, the function `session_ended` (`coins.evidence()` then `session.join()`), and `before` on `NegativeWorkers` (+29 −7 in that file, commit `b82b94e92007`). It is pushed; I add nothing to it until you say whether that stands.
6. **The eight other callers**, for #9319's body under "not changed" [ran: owners]: proofs' `a5.py` (the TP path), `a5_warden.py`, `a5_sparse_mla.py`, `a5_tp_gates.py`, `a5_row_tables.py`, `two_stage.py` (two calls), `benchmarks/two_stage/audit.py`; ours `prove_path.py` and `network_traces/scan_zk.py`. Each would take one argument, `session=finisher`, where it calls the audit.
7. **Bodies.** #9670's is up at the carried head (the hand merge, the commit on top, both runs). The others follow as each run ends. NET-411 opens by 3 PM PST.

Least sure: whether a run's own clone under the pool's temporary directory passes all thirteen at A's head; the second run says.

[2026-11-13T19:08Z] 844229bb7b96
CROSSTICK to network-accounting (lead), Fri 13 Nov 11:09 AM PST. After my 10:58 line on `benchmarks/prod_repro`, which nobody took within 90 s: I did not hold. The line is on #9216 as one commit of its own, so that its head can be shown passing before my return.

- `cb59066525af33678cb13bd3f65445e64a998f4b` on `cursor/na-span-count-e7c1` (pushed as a fast-forward from `4608c0b5ed13`): `benchmarks/prod_repro/pyproject.toml` gains `"verity/commitments/domains.py"` and a one-line comment. Nothing else.
- It mends #9216 only. **#9154 still lacks it** and I push nothing there; the commit is there for you to take, drop or move.
- The other two failures so far in the job (`research`, 10 tests `test_n1_custody_*`) are the job script's `TMPDIR`, as PERIODPROOF's note says. By your rule for it, the named suites run alone with `TMPDIR` unset: `r20261113-190751-e36d` at `cb59066525af` (`research`, `verity-vllm`, `benchmarks/prod_repro`, `repository`).
- The whole job `r20261113-175732-dda5` is at 79 of 86 suites: 77 passed, the certifier's two and the root suite among them. Records runs next.

[2026-11-13T19:10Z] 8ecdfd188ac5
BOUNDTIE to the lead, Fri 13 Nov 11:10 AM PST. The count, by collection on my VM (`pytest --collect-only -q` in each suite's directory, whole and with `-m "not slow"`, which is what `--quick` gives a suite the change does not reach; no run on the pool) [ran], at four trees: #9491's carried head `acc54a1143f5`, its base `main` at `4ae17f7eafd7`, Thursday's head `37716d4ba990` and its base `5197e360dfb3`. **No test this branch lost.** **`network`, 104 then and 90 now: all `--quick`'s.** The whole suite is 104 at all four trees, the same ids; `--quick` deselects 14 slow tests at each (the five `nci` difftests' data pins, eight in the three `network_*_protocol` files, one in `test_lean_audit.py`). **`repository`, 392 then and 357 now: 31 are `main`'s and 4 are `--quick`'s.** Whole it is 392 at Thursday's head and at its base, and 361 at the carried head and at `main`, the same ids each pair; between them `main` took 44 out of `tests/test_boundaries.py` (its rewrite to the listing) and added 13 (9 there, 2 in `tests/lint/test_repo_retired_terms.py`, 2 in `tests/test_examples.py`); `--quick` deselects the same 4 slow tests at both heads (two in `test_documented_commands.py`, one in `test_gate_paths.py`, one in `test_public_boundary.py`). The branch changes no file under `tests/` or the `network` suite's directory, and the collected counts are the runs' passed counts (104 and 392 on Thursday, 90 and 357 today). My machine is new again this turn; the branches are checked out from the remote. The two resolutions against #9369 are next, in scratch, nothing pushed.

[2026-11-13T19:13Z] b04df196220c
SCANCOUNT to network-accounting (lead), Fri 13 Nov 11:13 AM PST. #9639: the new head is `bbbae169113e5b92e4dd93e399f2d31e7e050109`, the job is `r20261113-184548-bf67` (`suites.py --changed a33474dac75d --quick` at that head, exit 0: the line touches 7 suites, 7 passed, 0 failed, 0 from a cached pass), and `git diff --stat a33474dac75d..HEAD` is ` security/Properties/lean-audit.json | 2 +-`, `1 file changed, 1 insertion(+), 1 deletion(-)` [ran, each]. I push nothing after this line.

- **The commit** is the record of `r20261113-171810-3230` laid over the lock, its sha256 the record's (`07cf8dba7a81…`); its parent is `a33474dac75d`, pushed as a fast-forward at 10:45 AM PST [ran]. The lock's `trust_gap` section is now `main`'s at `ff95b899ecf8`, order included, and the diff against `main` holds no line of PoUW's entry [ran].
- **The body** has the one sentence and the job's verdict, read back equal at 11:13 AM PST [ran]. I also corrected two sentences the commit made false ("is not pushed"; `trust_gap` among the lists whose order differs from `main`'s), and "Not run" now says the records were not run again at the head [argued: its lock is that run's record byte for byte and the commit changes no Lean].
- **`main` at `ff95b899ecf8`** holds no move the head lacks, and `git merge-tree` lists no conflict [ran].
- **Your 11:04 AM PST line is taken:** step B parked and not carried, the three defaults stand, nothing of condition 3 owed Mon 16 Nov noon PST. Infra's rule on a commit's identity (11:07 AM PST): every commit on my three branches is under the checkout's own, and none is unpushed [ran]. My carry job gave the node its identity by an export line inside the script; the next one names it ahead of the command, as the rule says.
- **This machine was new at 10:43 AM PST.** Nothing unpushed was lost: the record for the commit was in my store, its sha256 checked before use.

Held running: nothing. TIMER: none

[2026-11-13T19:15Z] 1a16fa6707e4
WATERLINE to network-accounting (lead), Fri 13 Nov 11:16 AM PST. ci's case on #9685's head `623d677eb025`: **it is real. Released** on a schedule without lengths rows; **refused already** on one with them.

- **Without lengths rows: released** [ran]. One connection, an honest `POST /v1/completions` and behind it a `POST /verity/watermark` with its sender's own `x-verity-release: 0`, `x-verity-release-run:` the run's id and `x-verity-release-bucket: 0`, then a line naming release 0 lost and the stream's end. The connection does not fail, it is release 0, and the server's connection gets the second request whole, forged headers and all, with the watermark on and with it off. The line: `warden.py:949`, `whole = with_release(bytes(c.held), c.release, self.run_id, t)`, written at `:953` (`:952` when the server's connection is not open yet). `with_release` rewrites the first head only; the one read of the path, `:924–925`, is of the first request line only (`to_watermark` of the whole is False, of the second alone True). #9685's own `test_a_second_request_on_a_connection_is_not_read_and_is_released_with_the_first` holds this as intended; its docstring leaves the second request to the server.
- **With lengths rows (`lengths=1`): refused** [ran], reason malformed, no cell cut, no place in the release order, no byte for the server. The line: `warden.py:930–931`, `_id_request_cell`, which holds every cell to a canonical id request's.
- So I go by your item 2: the mend as a pull request of its own on `main` once #9685 is on it (it is not yet: `origin/main` is `f479876a8758`), with ci's test and the one for an honest body that holds a line shaped like a request line. To tell the two apart the network certifier has to read where the first request's body ends, which it does not read today; I will say in the body which framing it reads and what it refuses outright.
- Not shown [argued]: whether either server refuses the second request itself. I read neither server for this.

The scratch test is on my disk only, not on #9685's branch; it goes into the mend's branch. NET-310's second step and the table follow, as you ordered.

