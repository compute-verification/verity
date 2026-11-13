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

