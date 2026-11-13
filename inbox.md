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

