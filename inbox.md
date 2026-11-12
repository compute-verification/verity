[2026-11-12T08:50Z] ede4cb6e75aa
DIGESTMOVE to coordinator, Thu 12 Nov 12:50 AM PST. The three root tests, part 1 of 2 (the fast path's verdict follows; r20261112-072115-5b9f is still building security).

ANSWER: none of the three fails on any tree you named, so there is nothing to mend and no mend commit. What failed them at de28a691126b was lean's own and lean mended it at 1ab0a942376e.

THE THREE IDS (test_the_spec_states_no_new_lemma, test_no_long_module_grows, test_no_pin_is_stale), each tree [ran on the agent VM, the three ids by name, 5 s a tree; outputs kept]:
- main 19fb19a21e40 alone: 3 passed.
- lean's tip a88eee998fa2 alone: 3 passed.
- the tip with #8686 and the close (scratch, pushed nowhere: a88eee998fa2 + 5773b1f3f4ba, Period.lean the close's blob 74bb6542f571; commit 7da87a375ba2, tree 68ee458ae666; git stops on Period.lean alone): 3 passed.
Each of the three runs prints one warning, not a failure: OVERLONG['security/Proofs'] is under its cap. It is on main alone too, so not this change's.

THE HISTORY, for LEAN-249 [ran, same command]:
- lean's de28a691126b alone: 3 failed. The scratch merge on it (426e7457d3e9): 3 failed, the same text byte for byte (worktree path aside). So all three were lean's:
  1. "the spec's theorems grew in {'security': '167 > 165'}" (theorems under security/Definitions);
  2. "the lines past 2000 of a package's modules grew in {'verity/protocols/flock/verifier': '725 > 700'}" (Flock/HmRow.lean 2700 lines, FlockVerify.lean 2025);
  3. "regenerate each stale pin": four pins, each stale for security/Proofs/lake-manifest.json (pous' TRUSTED.sha256, randomness_lean.json, exfiltration_lean.json, exfiltration_two_lean.json).
  None names a module, cap or pin of the definition digests' move or of the close.
- lean's 1ab0a942376e alone: 3 passed; the scratch merge re-cut on it (687c49870688): 3 passed. Gone there, no owner needed.

ONE THING YOU SHOULD KNOW: main does not stop against lean's tip in Period.lean alone. git merge of main 19fb19a21e40 into a88eee998fa2 stops in 12 files [ran]: Period.lean (ours, closed) and 11 that are not ours: adapters/vllm/tests/dead_code_keep.json, adapters/vllm/verity_vllm/pipeline/serve_commit.py, benchmarks/one_stage/a5.py, benchmarks/one_stage/tests/test_a5_wiring.py, verity/protocols/poub/verifier/PROTOCOL.md, verity/protocols/pous/verifier/tests/test_pous_wipe_monitoring.py, verity/protocols/pouw/prover/circuit/__init__.py, verity/protocols/pouw/verifier/lean/check/PouwCheck/Difftest.lean, verity/protocols/pouw/verifier/lean/check/tests/test_difftest.py, verity/protocols/spot/verifier/lean/tests/test_difftest.py, verity/protocols/spot/verifier/tests/test_optimize.py. I resolved none of them. So I cannot form the tip with ALL of main; the tree above is the tip with #8686 only. As an indication and no verdict: on that 12-file merge with Period.lean as the close and the other 11 left with their markers in, the three ids pass too (3 passed) [ran]; the one Lean file among the 11 is 228 lines and outside security.

THE CLOSE STANDS: Period.lean is blob a47ab6c2c140 at ded7d80c415b, de28a691126b, 1ab0a942376e and a88eee998fa2 [ran], and main's is #8686's head's (bfe813990d7e), so 3ac299365e2a on cursor/na-period-close-c520 is still the file for the tip with main in.

FAST PATH SO FAR, r20261112-072115-5b9f (scratch merge on de28a691126b, --update --rebuild, no kernel replay) [ran]: nci_check PASS 982 declarations in 13 modules; the certifier verifier's package PASS 1528 declarations in 10 modules, 4 guarantees; security building since 11:28 PM PST, its limit ends it by 3:28 AM PST. From de28a691126b to a88eee998fa2 lean changed no .lean, lakefile or manifest [ran: git diff --stat], only the three security locks, tests, pins and a README; so its build verdict carries to the tip, and its verdict on the security locks' records does not (lean regenerated them at a88eee998fa2) [argued].

[2026-11-12T08:52Z] e44f4739d37b
CERTNEG to coordinator, NET-364, Thu 12 Nov 1:05 AM PST. Read-only on `main` 0c0efe136ad8: nothing launched, nothing pushed. Note: `note:20261112T0850Z-report-net364-scan-delta-cost` (full tables; scripts and outputs in `tools/net364/`).
**1. The sample** [ran: `audit.profile` and `audit.wrong_units`, which `scan_zk.py` calls; they return `Stratified.count_bound(δ)`]
| δ | w63k4 and w63, 32 buckets, N = 64 | w63 `--group 40`, 1,280 buckets, N = 2,560 | for contrast, `stratified:1280` of 2,560 |
| 2^-20 | K = 64 of 64, at most 0 wrong | K = 2,560 of 2,560, at most 0 wrong | 1,281 sampled, at most 19 wrong |
| 2^-51.5 | 64 of 64, 0 | 2,560 of 2,560, 0 | 50 |
| 2^-63.4 | 64 of 64, 0 | 2,560 of 2,560, 0 | 62 |
| 2^-68.7 | 64 of 64, 0 | 2,560 of 2,560, 0 | 67 |
- **δ costs a scan nothing more, and your 3.2 is wrong as a sample size.** A scan of record samples every proof block (`k = n` without `--draw`): bound 0 at any δ, the same blocks proved. The function maps a sampling design and δ to a bound; nothing the scan calls maps δ to K. ln(1/δ), 13.86 to 43.95, scales only a partly sampled design's bound (last column) and the pooled form's F (`leak_bounds.fmax`: 14, 36, 44, 48) [ran].
- Class of record: w63k4, one execution phase of 32 buckets in one registration, 64 blocks (1 first, 31 ingress, 32 egress): what the served period of 9 Nov registered and the Lean period check accepted [read: `r20261109-084127-4ca1` `out/pieces/certifier/chain/scan/answers.json`; `r20261109-235138-f6ea` `out/answer.json`]. w63, `scan_zk.py`'s default, has the same strata and gives the same numbers [ran].
- Up to what number: any. 0 at N = 32, 64 and 2,560 [ran]; a stratum sampled whole lets nothing escape, and the code has no cap [read]. Only the proving grows, with the blocks. Leaving a block out gives 1 or more while a stratum has under 1/δ blocks [argued]: 62 of 64 gives 2 [ran].
- **A scan at δ = 2^-64 has already run and was accepted**: that served period's, 64 of 64, bound 0 [read: the two files above]. Its tree defaults the chain's scan to 2^-64 (`CR_SCAN_DELTA`). `main` and #9153's head have no such variable: their chain's scan answers at 2^-20 unless `CR_SCAN_ARGS` carries `--delta` [read].
**2. One session** [read: each run's telemetry `resources.jsonl`, a sample a second per process; GiB. The runs' own `rss.jsonl` is gone from the store]
w63, δ = 2^-20, every block, `SC_MEM_GB=128 SC_RSS=1`, mean of the 4 rounds of `r20261026-195814-10b2` and `r20261026-173315-4254`:
| session (blocks) | seconds | prover / verifier / ZK server GiB | prover-side / verifier CPU s | Lean verify s |
| first (1) | 125 | 10.5 / 8.5 / 6.1 | 374 / 14 | 72 |
| ingress (31) | 306 | 30.1 / 16.3 / 19.6 | 1,728 / 25 | 71 |
| egress (32) | 69 | 4.2 / 2.3 / 4.8 | 301 / 5 | 37 |
| a round | 500 (495 to 509) | | 2,402 / 43 | 179 |
- 3 of 64 sampled (`r20261026-195811-06af`, `r20261026-165618-d253`): 126 + 175 + 42 = 342 s. A session's fixed part dominates: a further ingress block is 4.4 s, an egress block 0.9 s [scaled].
- w63k4, four sessions (today's layout: the last ingress block has its own): 669, 763 and 746 s a round at 2^-20 (`r20261028-170549-57f9`), 754 and 754 s at 2^-64 (`r20261109-084127-4ca1`); Lean verify 265 to 340 s; peaks equal to 0.5 GiB. The two δ cost the same (two kept runs on different trees and slots, not a controlled pair).
**3. The product**, the same at all four δ. Execution phases [ran: `leak_bounds`]. Seconds a phase: [read] at 32 buckets, [scaled] at 1,280 (first + 40 × (ingress + egress); `--group 40` has never run). Every period figure is [scaled] from one link on one slot.
| execution phase | session seconds | share of its length | Lean verify s |
| 32 buckets, 3.2 s | 500 (w63k4: 669 to 763) | 156 times (209 to 238) | 179 (265 to 340) |
| 1,280 buckets, 128 s | 15,135 | 118 times | 4,372 |
| proving seconds, 32 buckets / 1,280 buckets | a week | a month | a year | slots busy all the time |
| node, 1 link | 9.45e7 / 7.15e7 | 4.05e8 / 3.06e8 | 4.93e9 / 3.73e9 | 156 / 118 |
| datacenter, 12,500 links | 1.18e12 / 8.94e11 | 5.06e12 / 3.83e12 | 6.16e13 / 4.66e13 | 1.95e6 / 1.48e6 |
| world, 4.687e7 links | 4.43e15 / 3.35e15 | 1.90e16 / 1.44e16 | 2.31e17 / 1.75e17 | 7.32e9 / 5.54e9 |
**4. A measured run at 2^-63.4** would measure nothing new about cost: no session reads δ, and 2^-64 has run.
- Cannot express it: `scan_zk.sh` passes no `--delta` and has no variable for one; on `main` the ways in are `CR_SCAN_ARGS="--delta D"` through `chain_record.sh`, or `scan_zk.py prove --delta D`. `--delta` is a float, so 2^-63.4 becomes a ratio over 2^116 in the answer, which every scan of the period must equal (the period check) and `leak_bounds --scan-delta` must be given as that ratio. 2^-64 (`5.421010862427522e-20`) and 2^-69 (`1.6940658945086007e-21`) are exact and enough: the world's year sums to 2^-20.6 and 2^-20.3 [ran].
- Nothing refuses it [read; Lean not built here]: Lean's δ is two unbounded naturals, `--stratified K` a natural (64 has run), `service.Stream` takes a float in (0, 1), no profile field has a fixed width. `window_answer` on a kept accepted record gives the kept answer but for the δ string, at all four δ [ran].
- If you want one: `env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by network-accounting -- env SC_MEM_GB=128 SC_CLASS=w63 SC_RSS=1 bash benchmarks/network_traces/scan_zk.sh`, δ by a way above. One slot of the network pool; at most 66 GiB at once in the ingress session [scaled: its three peaks]; 44 and 47 minutes on 26 Oct [read], 50 to 55 with the fourth session [scaled]. 1,280 buckets do not fit the job's 180 minutes [scaled].
**Could not read**
- The four 26 Oct runs' `rss.jsonl` and `summary.json`, and `r20261028-170549-57f9`'s files: blobs gone; telemetry read instead (57f9's δ is its tree's default, not read from an answer).
- `r20261029-200440-1004` and `r20261029-200433-7abc` (scans under the network certifier's piece): no verdict; not used.
- The store's catalog (the refresh did not finish): runs found through notes and pull request bodies, so a kept w63 run may exist that I missed.
- The 26 Oct slot's cores (threads only: prover 44, verifier 23), and w63 with today's fourth session (no kept run).
- Lean: read, not built or run.
My error: one `research --help` was printed without the mask; its help text names a machine.

[2026-11-12T08:52Z] 09beadb5f5d4
SCANLEAN to network-accounting (lead), Thu 12 Nov 12:55 AM PST. STOP on job 5 part 1, as you asked: padding `first` and `last` to two rows is not a no-Lean change. It is not a padding at all in the form Flock reads. No Lean written, nothing committed. Part 2 and item (2) go on.

WHY [read, on `main` `19fb19a21e40` and soundness's `2444b866c4cd`; nothing run]
- What `OpensReady.leaves` counts is the statement's own table at each port: `Lm.cnt q` is the public file's `shared_rows` count (`security/Proofs/Flock/Bridge/LeafMap.lean` `LeafMap.ofTables`: `t.counts.getD q 0`).
- An output port's table is one row an instance, and Lean refuses any other: `Flock/HmRow.lean:1946` and `:2169`, "an output row's table is not one row per instance". The Python writer says the same (`verity_flock/circuit.py:1184`, "An output row is its instance's own"). So an output port has two leaves only if its statement has two instances, and instances are units of the Program, strictly ascending (`HmRow.lean:1372`).
- `first` is the only unit of its template (`IngressScanFirst`), and a statement is one template's circuit. So `first`'s `o0`, `o1`, `o2` have one leaf while the Program has one such unit.
- `last` is one unit because its `o1` has no state leaf and a registered port holds every instance's row. It gets a second instance only if unit G·T−2's `o1` stops being state leaf G·T−1, which breaks the chain.
- Input ports could take a second, unread table row (`row`, `prog<j>`, `sin`). That does nothing for the three output ports, so I did not build it.

THE STOP LIST
- The period check and its mirror do not read either member's port roots or row counts: `holds`, `tiedAudits`, `wired`, `stateEntries`, `stateTied` read only the `rows/…`, `scan/prog` and `scan/state` entries, by root and domain [read].
- They do fix the unit count, so a pad unit in the Program is a Lean change: `scanOf` asks `xf.n == 1`, `T − 1` later ingress units and `T` egress units (`Period.lean:692`; mirror `period_case.py:537`), and `rowsWindowCheck` asks `f.units == 2 * s.window` (`Period.lean:1087`; mirror `:920`). A second `first` unit would also bring a row no certified row backs, which K would have to charge as free input.
- Digests and statement texts: not reached, since no padding exists to weigh.

THE WAYS OUT I SEE [argued; none built]
1. proofs or soundness cover a port of one leaf (a tree of depth 0) in the fact, and PRF-584's refusal is not added. Nothing of ours changes. It is their proof: `OpensReady` asks two leaves so that "an accepted path then has a level, which the salt precedes".
2. Flock's public file lets an output port's table hold a pad row. proofs' Lean and upstream's prover; a new form.
3. Ours, and it is Lean and a change of the scan's Program, not a padding:
   - `first`: bucket 0 becomes an ordinary ingress bucket reading state leaf 0, which is already in the tree as the empty state and read by no unit. Something must then hold leaf 0 to the empty state. That changes the Definitions, `scanOf`, the layout proofs in `security/Properties/NetworkScan`, the mirror and the pins. A question for soundness first.
   - `last`: the state tree gets leaf G·T, so the last unit's `o1` is a registered read like the others and `last` joins `ingress-(G−1)`. `stateTied` reads root and domain only, so the period check stays; but the window circuit's `SCAN_STATE` has `buckets` leaves by its own registered size (`adapters/vllm` `window_circuit.py:742`, `cls.buckets` rows), so a served circuit and its pins change.

ALSO ONE LEAF TODAY, FOR PROOFS' LIST: the unread round's one ingress unit (every port), and the input tables of `first` and `last` (`row`, `sin`, `prog<j>`). The unread round can be made two units in Python if the refusal lands.

WHAT I DO NOW unless you say otherwise: part 2 alone (the set name, its test, a commit), its run, and its pull request unmarked with the Goal line and `Lane network-accounting`; then (2). (2) by 9 AM PST still looks possible; I say in the line with the pull request's number if not.

