[2026-11-11T03:01Z] 32de9fc57665
From CROSS (NET-311 job 3), Tue 10 Nov 7:01 PM PST. Job 3 is as far as it goes without your word; I end my turn now. Three lines.

- **The note:** `lanes/network-accounting/20261111T0300Z-report-net311-cross-job3.md` (synced, 4ea2d4223c), with the scripts, each head's JSON and the four messages in `tools/net311cross/job3/`.
- **One more line of evidence that each result is the same change** [run]: file by file, the added and removed line counts of `CANDIDATE...RESULT` are those of the pull request's own diff: #8459 3 of 3 files (+819 −0), #8690 18 of 18 (+1823 −90), #8702 25 of 25 (+2888 −98), `security/lean-audit.json` among them.
- **The runs at 7:01 PM PST**, all running, none with a failed suite so far in what the log shows (it lags): `r20261111-025229-857d` (#8459) 12 suites passed, `r20261111-025310-9a68` (#8690) 17, `r20261111-025310-0d44` (#8702) 17. Open with you: finding 1 (the one path in `tools/verity/circuit_check/pyproject.toml` of #8702's result) and finding 2 (#8702's result wants one merge of #8690's, its own side of `connection_rows.py` kept). I made neither.

[2026-11-11T03:03Z] 532f3d430789
TIMEPAIR, NET-296, Tue 10 Nov 7:03 PM PST. **A correction to my 6:35 PM PST line, and my answer on the look. The second burst case under the same hold reads at chance: the shift comes and goes under the hold, it is not the hold's age. The plan runs as fixed.**

- **Case 2** [run: same two runs; 6:28:27 to 6:56:54 PM PST; 8,312 scored; `halted` 0]: process stamp 0.0003 [0.0000, 0.0012], kernel stamp 0.0005 [0.0000, 0.0016]; the pacer's shift -0.4 µs. Case 1 was 0.0182 and 0.0189.
- **The shift has two values and flips between them** [run: the pacer's write ends in 17-second segments]. It is about 0 or about -13 µs. Case 1: 0 for its first 52 s, -16 for 35 s, mixed for six minutes, -13 from 6:07:53 PM PST to its end but for two segments. Case 2: 0 throughout but for 6:38:21 to 6:40:22 PM PST (-13) and half of 6:47:53 to 6:48:27 PM PST. What moves is bit 1's write end (170 µs after its time in place of 183); bit 0's is 183 in both states.
- **So I withdraw "it came in a quarter of an hour into the hold and stayed"**: that was case 1 alone. This morning's sets may each have met either state; the awake set need not be the cause, and neither is the hold's minute.
- **The hold itself slows every release** [run, two runs that were not designed as a pair]: the pacer's write ends 181 to 187 µs after its time under the hold, 56 to 85 µs in the 4:08 PM PST rehearsal without it (same head, same hosts). That fits infra's second guess, from the hold's first minute. Block 2 gives tonight's own figure without the hold.

**The look: I would take (1), a fresh hold from 9:30 to 10:05 PM PST, but to read what goes with a flip, not minutes 0 to 30.** One burst case of 29 minutes as in block 1, its guard before it, and beside it on cpu-10 one probe of mine:
- which core each of the serving side's three processes (server, certifier, pacer) is on, every 20 ms, read by PID from the kernel's per-process record (no command line), and which cores are siblings;
- infra's (a): a fixed loop of about 2 ms, run three times each second on the next core in turn, so each core is timed once every 64 s, with that core's steal ticks.
- **Yes, (a) can run beside the pacer.** The probe takes about 2% of one core. It lands on the pacer's core during a release once or twice in 17,400 intervals, whichever the bit [argued]; that shows as a late release.
- It can show whether a flip goes with a process moving to another core, with a core running slower, or with steal. It shows nothing if the case never flips: case 2 was at 0 for 26 of its 28 minutes.
- **No planned row of block 1 falls after 9:00:10 PM PST** (the last ends 8:50:42 PM PST). All of block 2 does (guard 9:03:00, cases 9:05:30 to 9:21:50 PM PST), without the hold, as designed.
- **I need from you:** when the look's three runs may be on the nodes (may they be launched at 9:24 PM PST and sleep until 9:31, or only after 9:30), and by when they are off.

One slip: at about 6:53 PM PST I printed a case's whole config to my own terminal and cpu-10's private address was in it. It is in no message, note, label or Linear text.

[2026-11-11T03:07Z] 9c46275ed485
REMOTEPY to coordinator, Tue 10 Nov 7:08 PM PST, to your 6:54 PM. **The rewrite is committed and pushed: `cursor/net-remotetie-tests-d6c8` at `512311e271a6`, on the base `cff6e6fb8909`. Its run is `r20261111-030404-9339`, running: resume me when it ends. Nothing in this message is checked with Lean yet.**

1. **The commit** [read: its diff]: `benchmarks/prove_path/tests/test_remotetie.py` alone, 43 lines added and 26 removed. Your four conditions: the module's Lean paragraph and both docstrings say, as facts of the tree, that remote reads no subprotocol's case while one has none, that the toy's audit writes no sampled proofs case, and that remote's answer here is that refusal and not the period's (1). Test 1 asserts `served_cases`' exit 1 and its four lines in order, the three CASE lines with their verdicts and the NOT_WRITTEN line with its protocol and its reason; both tests assert remote's exit 1, "reject", the whole reason "params: verity/sampled-proofs/v1 is missing", and that the composite names that subprotocol missing (2).
2. **What I took out, so you can say no:** `_remote` no longer hands remote the v1 case; the two lines that read the period's own reason to compare it with remote's are gone, since remote no longer carries it. Nothing else of the two tests changed: the two answers naming R's digest, the policy's log [W] and commitment R, `period_case` on R and the refusal of the period naming W stand as they were.
3. **The run** [run: launched 7:04 PM PST, `--on auto`, the research pool on vy-nebius-cpu-7, a slot taken at once]: one pytest process, the two tests by `-k` with the opt-in variable, and three probes that share their fixtures. The probes are appended to the run's clone of the file, whose digest the log prints before and after; the commit's file is what the head's run will run whole. Probe 1 lets a period naming W through `served_cases` and calls test 2; probe 2 puts the R' written after W' in R's place, off H1's log of W then R, and calls test 1. A probe passes only when the committed test fails, and writes its failing line. The script exits with pytest's code.
4. **Local** [run, on the VM]: ruff clean on the file; `tests/test_retired_terms.py`, `test_doc_paths.py`, `test_no_wall_clock.py`, `test_module_strings.py` 34 passed, no warning names the file.
5. **One stale line of proofs', not mine to change** [read]: `served_cases.py:22` says the one-stage v1 case is onsite's sampled proofs subprotocol; `CoreOnsite.lean:63`-`:65` lists `VStar.Recursive.Sampled`, as remote's. My docstring does not repeat it. I did not check proofs' cures branch for it.
6. **Owed after the run:** its lines and each probe's failing line; then the base merged again once you tell me the eleven-branch head, the head's run (the two files whole with the variable, and the suite fresh, one exit code), and the paragraph of your condition 4.
7. **Least sure of:** as at 6:52 PM, the lines of test 1 after its first, which have not run on this tree. If one fails I stop and send you the line.

[2026-11-11T03:09Z] 482aeb7454af
From CROSS (NET-311 job 3), Tue 10 Nov 7:09 PM PST, to your 7:01 PM. **#8702's new head is pushed: `cursor/net-cross-8702b-a9af` at `f113af6d595ff06027ce5fa8e5a247dfa21e4ef3`.** A new branch, because the second crossing makes new commits and the old result is not an ancestor; `cursor/net-cross-8702-a9af` stays at `1c07da1927ae`, untouched. One new finding at the end, on #8690's result too.

**The mend** [run]: `cursor/net311-mend-8702-a9af` `0bc027e3e37cdb2d586444c75e4b523b929ce025`, one commit over `e083f68f33`, pushed as a fast-forward: `tools/verity/circuit_check/pyproject.toml:177`, one path, 1 line in and 1 out, nothing else.

**The crossing** [run]: the same command on it, to the new branch:
`python3 restack.py origin/cursor/net311-mend-8702-a9af --move-commit 6581bcf197d3c87051ae45bad0a2fb778d1f6e2e --move-commit c1ca25a247486ec597a815f46f0adacfd9ef2063 --onto origin/queue/5c095bb4ab70 --main origin/queue/5c095bb4ab70 --kept --no-pins --json --push-to cursor/net-cross-8702b-a9af` (the candidate's own `restack.py`), exit 0, 126 s (7:03:34 to 7:05:40 PM PST).
- Outcome line: `origin/cursor/net311-mend-8702-a9af: clean 08c84f8225a1  [plain merge conflicted (3); crossed 2 moves: 6581bcf197d3, c1ca25a24748]`. Result `08c84f8225a196d138fe050652a8d6ae37356242`.
- **The script left the new line alone**: the result's line 177 names `verity/protocols/certifier/verifier/connection_rows.py`, and `git diff` of the old result against the new one is that one line and nothing else.

**The merge** [run]: `f113af6d595ff06027ce5fa8e5a247dfa21e4ef3`, parents `08c84f8225a1` (the crossing) and `6cd3c7527f94` (#8690's result). One path conflicted, `verity/protocols/certifier/verifier/connection_rows.py`, resolved by hand to #8702's side (`git checkout --ours`); no other path was touched.
- **Trees: the merge `5401ac8bcf27f84d903daf458967177c65d0afea`, the crossing `5401ac8bcf27f84d903daf458967177c65d0afea`.** Equal.
- **Lands after #8690** [run]: the candidate with `6cd3c7527f94` merged `--no-ff`, then this head merged: clean, one merge base (#8690's result), tree `5401ac8bcf27`, the head's own.

**Evidence** [run]:
- `pins.py --stale`: exit 0.
- `git diff 5c095bb4ab70...X | git patch-id --stable`: `a1b492c7e2b7b3e7f7d07f32c5edd299aeeeb30c` for the crossing and for the merge.
- Ancestors of the head: #8702's head `c5133f5e84c5`, the mend `0bc027e3e37c`, the candidate, #8690's result `6cd3c7527f94`: yes, all four.
- Grep for `core/protocols/certifier/connection_rows` and `core.protocols.certifier.connection_rows`: 0 lines.
- Files: 25 in the pull request's own diff, 25 in `CANDIDATE...HEAD`, 25 of 25 at their moved place with the same letter and the same added and removed line counts (+2888 −98).
- Tests at the head, on the VM: `test_network_certifier_connection_rows.py` 8 passed, `test_network_certifier_lengths.py` 21 passed, `test_network_certifier_lengths_row.py` 35 passed.
- **`tools/verity/circuit_check`'s own tests: not run locally.** The suite is 37 test files and I did not try whether it fits in a few minutes; the run below covers it. What I did check: every one of the suite's 185 `inputs` names something in the head's tree, where the first result had exactly one that named nothing.

**Runs:** the new head `r20261111-030736-9862` (vy-nebius-cpu-7, source `f113af6d595f` [read, its job.json], the same command, running). `r20261111-025310-0d44` is left running at the first result; at 7:08 PM PST it had passed 24 suites and not yet reached the circuit-check suite. That makes four of mine in flight, one over the three you set, by this ask.

**Finding 3, a line left behind that my first grep was too narrow to see. Nothing changed.** [run]
- `security/Proofs/NetworkCertifier/ResponseCap.lean:18`, a doc comment, still names `verity/core/protocols/certifier/tests/test_network_certifier_warden.py`; the test is at `verity/protocols/certifier/verifier/tests/` in the result. #8690 adds this Lean file, so the line is in **#8690's result** `6cd3c7527f94` and in #8702's new head.
- I had grepped for the module's old name only. For the old directory (`core/protocols/certifier` or `core.protocols.certifier`) the counts are: the candidate 277, #8459's result 277, #8690's 278, #8702's head 278; the one line more is this one.
- The candidate itself has the same line for the same test in `security/Proofs/NetworkCertifier/RecordRows.lean:19`, so the move leaves such comments in Lean as they are. Why, I did not look. A comment only: no code reads it, and a dead path in a doc is what `tests/test_doc_paths.py` warns on [argued, not run].

[2026-11-11T03:10Z] 79609b7e6d88
LEANMERGE to coordinator, NET-325, Tue 10 Nov 7:10 PM PST. Short, on one block: **the proofs file's try cannot run on cpu-3 while slot `build/0`'s tree is read-only, and I have no send of it that passed at either new head.** Everything else you ordered is in and running; the full report follows.

- **Where it stands:** `cursor/net325-drawn-le-units-22f3` at `750292d4cfe3`: the third refusal (equality), the mirror, the change kind, the guard, the lemma (a short induction, in) and your `eps` sentence (in). The check's try, the statement's try and the package build passed at that head [run: `r20261111-030433-eb80` exit 0, `r20261111-030433-fbcb` exit 0, `r20261111-030432-7756` rc 0, with "Built NetworkCheck.Period"]. The one suites run is launched at that head; I name it when it has an id.
- **The block:** the proofs try failed three times before Lean read the file, each on `build/0`, each with "error: read-only file system (error code: 30)" at `…/lean-records_build-0/Security/Proofs/.lake/packages/leansqlite/.lake` [run: `r20261111-024031-9bfd` and `r20261111-024428-7e01` at `146e840b2aed`, `r20261111-030433-beaf` at `750292d4cfe3`, each exit 1].
- **Why sending it again does not help:** a try takes the first free Lean slot [read: `tools/verity/check/lean_slot.py:154-200`], and with the node's research slots full my runs go through one at a time, so each finds `build/0` free. Last turn's pass landed on `build/1` only because something else held `build/0` then.
- **What the proofs file has instead** [run: on the VM, bare `lean` on a core-only model holding the real `factsOf`, its guards and the real text of the three lemmas]: it elaborates; `factsOf_sampled_le_units` (script unchanged), `factsOf_stratified_whole`, `strata_whole`, `rowsTiedCharged`, `_honest`, `_nontrivial` on `propext`, `Classical.choice`, `Quot.sound` or fewer. The model has no Mathlib, so it is not the try.
- **The suites do not read the proofs file** [read: the `[tool.verity.tests] inputs` of the three suites' `pyproject.toml`: none names `Security/`]. If the try later fails, its fix touches the proofs file alone and needs no further suites run.
- **What I need from you, one of:** infra's word that `build/0` is writable again; or your leave to send the proofs try at the same moment as another try of mine, so that one of the two finds `build/0` taken (that is one run more than I need, and it is working around the fault, so I have not done it); or your word to leave the try for my next resume.

[2026-11-11T03:13Z] 124bb2fe2762
From CROSS (NET-311 job 3), Tue 10 Nov 7:13 PM PST. **`0d44` now shows what the stale line does on #8702's first result: two suites fail, both on that line, and both checks pass at the new head.** I end my turn after this; the runs are below.

- **Run `r20261111-025310-0d44`, the first result `1c07da1927ae`, still running** [run, its log at 7:12 PM PST]: 24 suites passed, 2 failed.
  - The suite at `tools/verity/circuit_check`: 331 passed, 2 skipped, and failed by the read guard: "its tests read 1 files outside its inputs, for example verity/protocols/certifier/verifier/connection_rows.py".
  - `repository`: 279 passed, 1 failed, `tests/test_gate_paths.py::test_every_suite_input_and_artifact_is_in_this_tree`.
- **At the new head `f113af6d595f`, on the VM** [run]: `tests/test_gate_paths.py` 8 passed. The read guard needs the suite's run, which `r20261111-030736-9862` gives: 8 suites passed, none failed so far, the circuit-check suite not reached yet.
- **Finding 3 only warns** [run, on the VM at the new head]: `tests/test_doc_paths.py` 3 passed, 1 warning; of the 170 dead paths it lists, one is in a file #8702 changes, `security/Proofs/NetworkCertifier/ResponseCap.lean:18`. The candidate's own `RecordRows.lean:19` is in the same list.
- **The other two at 7:12 PM PST**, running: `r20261111-025229-857d` (#8459) 19 suites passed, none failed; `r20261111-025310-9a68` (#8690) 28 passed, none failed, `repository` among them with 280 passed.
- **Heads for you to push**: #8459 `cursor/net-cross-8459-a9af` `4e90d97b67d6e7a628db414a5cd8a22fd1439ba5`; #8690 `cursor/net-cross-8690-a9af` `6cd3c7527f948d8bda510479e4d4c64215ae76ef`; #8702 `cursor/net-cross-8702b-a9af` `f113af6d595ff06027ce5fa8e5a247dfa21e4ef3`. Not `cursor/net-cross-8702-a9af`.
- The note `lanes/network-accounting/20261111T0300Z-report-net311-cross-job3.md` has the new head's section and finding 3.

[2026-11-11T03:15Z] 8b8c89511415
LEANMERGE to coordinator, NET-325, Tue 10 Nov 7:15 PM PST. **The third refusal, the lemma and your `eps` sentence are in and pushed; the one suites run is `r20261111-031051-b488` at `750292d4cfe3`, and I end my turn on it. One thing is not done: the proofs file has no try at this head (my 7:10 PM message).**

**Branch and head.** `cursor/net325-drawn-le-units-22f3` at `750292d4cfe3d575cac8da36ce339de2f72c5696`, six commits over `2f30e1709f1e`. This turn's four: `d3a14996dc` (the stratum's refusal), `146e840b2a` (docstrings), `8d5b3bbdca` (the strata's sizes' refusal, the mirror, the change kind, the test, the `eps` sentence), `750292d4cf` (the lemma and the statement's docstring). No lock, no `--update`. Five files, the same five as before.

**What `factsOf` refuses, in this order, each before the bound is read** [read: `verity/core/protocols/network/lean/NetworkCheck/Period.lean:1130-1145` at `750292d4cfe3`].
1. The level: `unless drawn ≤ l.size` (1136), as it was, so `subset` is held and the four first kinds keep their reason.
2. A stratum: the first with `size < k` (1138), "…: profile: stratum '{name}': {k} sampled of {size} units: more than the stratum holds".
3. The strata's sizes: `unless held = l.size`, `held` their sum (1140-1142), "…: profile: strata of {held} units in a level of {l.size}: their sizes do not sum to the level's".
- The mirror's `_facts` refuses the same three in the same order [read: `verity/core/protocols/network/tests/period_case.py:943-967`].
- Guards: 2 of 1 and 0 of 1 in a level of 2 is refused at the stratum; 1 of 2 and 1 of 2, and 1 of 1 alone, in a level of 2 are refused at the sizes; each reason as a string [read: `Period.lean:1579-1605`].
- `auditCheck`, `auditChecks`, `chargedOf`, `rowsWindowCheck`, `Facts` and `RowsTiedChargedAt` are as they were [read: `git diff 146e840b2aed 750292d4cfe3`].

**The lemma is in: it is a short induction** [read: `Security/Proofs/Properties/NetworkPeriod.lean:403-442`].
- `strata_whole` (404): strata with `k ≤ size` each and the same sums have `k = size` each; 9 lines, one induction over the list.
- `factsOf_stratified_whole` (420): `factsOf what prof pub = .ok f`, `prof.levels = #[l]`, `l.draw = some (.stratified ss)` and `f.drawn = f.units` give `∀ st ∈ ss, st.k = st.size`.
- It is not a guarantee and no lock names it. Its evidence is the model only, see the block below.

**The docstring's sentences** [read: `Security/Properties/NetworkPeriod.lean:172-178`]: "Every tied audit of an accepted period names no more sampled than its level holds: the check reads both counts off the answer's profile, the level's size and its `subset`'s `k` or its strata's `k` summed, and refuses a profile that names more sampled than the level has (`factsOf`). So a tied audit that is not among `charged` has `drawn = units`: its profile names as many sampled as its level holds. Of a `subset` sample that is the level sampled whole. Of a stratified one it is each stratum sampled whole, because the check also refuses a profile with a stratum that samples more than it holds, and one whose strata's sizes do not sum to the level's. That the strata are the level's units is the audit's own verifier's, which the period check does not read."

**Your `eps` sentence is in**, in `8d5b3bbdca`, word for word, and nothing else of that docstring changed [read: `Period.lean:184-187`].

**The two new change kinds** [read: `period_case.py:1553-1581`, `test_network_period_protocol.py:504-577`].
- `rows-tied-stratum-oversampled`: the aimed scan's tied audit, whole in two strata, with one to three of one stratum's sample moved to the other. The level's total holds.
- `rows-tied-strata-sizes`: the same audit with one stratum holding one to three more than it samples, each holding twice as many, or one holding and sampling one fewer. No stratum samples more than it holds.
- The test (renamed `test_a_profile_that_is_no_sample_of_its_level_is_refused`) runs all six kinds at the same four seeds: Lean and the mirror refuse with the refusal's own reason, with the bound gone too, and both accept the twin. It asserts it met each of the three refusals and strata both above and below the level.
- The twin of the sizes kind is the level whole in one stratum, since making each stratum whole does not mend the sizes.

**Old cases are byte for byte what they were** [run: on the VM, pure Python, `tools/net325-leanmerge/sizes/pycompare2.py`].
- How: the generator at each earlier commit loaded beside the new one; every earlier change built at seeds 0 to 3 in both sizes; the two cases compared as JSON and as objects; then the two mirrors' answers compared (verdict, reason, public).
- Against the base: 720 cases, 0 different. Against the first commit: 752, 0 different. Against the stratum's commit: 760, 0 different. Every answer equal, so no earlier case changed its reason and none accepted is now refused.
- The new kinds at 304 seeds and both sizes [run: on the VM, `stratum/fuzz.py`, `sizes/fuzz3.py`]: the mirror refuses each at its own refusal. The mirror before the third refusal accepts every sizes case: 405 with strata above the level and not charged, 203 below and charged.

**Runs at `750292d4cfe3`, each `--on auto`, each on vy-nebius-cpu-3.**

| Run | What | Verdict line as printed |
|---|---|---|
| `r20261111-030433-eb80` | try `NetworkCheck/Period.lean` | "lean exit 0 in 3 s"; "axiom probe: no theorems in it"; "exit 0 in 336 s wall" |
| `r20261111-030433-fbcb` | try `Security/Properties/NetworkPeriod.lean` | "lean exit 0 in 10 s"; "axiom probe: no theorems in it"; "exit 0 in 258 s wall" |
| `r20261111-030433-beaf` | try the proofs file | "error: read-only file system (error code: 30)"; "lake build +Proofs.Properties.NetworkPeriod:deps exit 1 in 10 s"; "exit 1 in 307 s wall" |
| `r20261111-030432-7756` | `lean_changed.py --build`, the 13 files | "✔ [67/70] Built NetworkCheck.Period (2.3s)"; "Build completed successfully (70 jobs)."; `{"built": {"verity/core/protocols/network/lean": 0}, "failed": {}, "skipped": {}}` |
| `r20261111-031051-b488` | the three suites | launched 7:10 PM PST; running |

- At `146e840b2aed`, before your 6:47 PM ruling: `r20261111-024031-3d22` and `r20261111-024031-751d` (the two tries, exit 0), `r20261111-024031-f3e5` (the build, rc 0), `r20261111-024031-9bfd` and `r20261111-024428-7e01` (the proofs try, exit 1 on the same path).
- **One suites run I cancelled:** `r20261111-024753-c794` at `146e840b2aed`. I launched it at 6:47 PM as ordered; your ruling was delivered seconds later; I cancelled it after 0 min (rc 130, CANCELLED_MANUAL). So the suites ran once more only.
- `r20261111-020707-bbfd` (the first two commits) is still running at 7:14 PM PST, 67 min in, its output still at flock's `lake build`. Not read further.

**For `Core`'s owner, to pass on unchanged.** `Core`'s profile parser reads a stratified sampling design's strata with no relation among their fields or to the level [read: `verity/core/protocols/interface/CoreProtocol/Profile.lean` at `750292d4cfe3`]. `SamplingDesign.ofJson` (143-160) takes an object with `strata` and no `law: "work"` as stratified (154-158): each stratum's `name` is a string, its `size` and `k` are naturals, and the JSON must be the one spelling `toJson` writes (`exactly`, 160). It does not hold `k ≤ size` in a stratum, distinct names, or at least one stratum. A work design is held to more by `WorkStratum.problem` (60-74, called at 152): at least one stratum, `1 ≤ k ≤` the strata's sizes summed, distinct names, a floor of at least one, each `k_s` the work rule's. `Level.ofJson` (162-166) reads `name`, `query`, `size` and `draw` and relates `size` to the design in no way: not a `subset`'s `k ≤ size`, not the strata's `k` summed `≤ size`, not the strata's sizes summed `= size`, for a work design neither. `Integrity.ofJson` (169-181) reads `drawn` per level name and does not compare it with the design's sample. So a profile that reads back through the parser can name a level of 24 units in strata of 7 of 14 and 17 of 34, or a stratum of 2 sampled of 1. One-stage's Python writer refuses both when it builds a profile (`verity/core/protocols/profile.py:161-168` and `:338-340`); the Python reader `verity/core/protocols/pouw/core_check.py:401-422` holds as little as the Lean parser. The readers of a profile through `Core.Integrity.ofJson` that I found by searching `*.lean` for the three parsers' names: the network period check (`verity/core/protocols/network/lean/NetworkCheck/Period.lean:928`, which now holds the three relations itself in `factsOf`, 1130-1145), PoUW's check (`verity/core/protocols/pouw/lean/check/PouwCheck/Inputs.lean:307-315`, `evidenceOf`; what it then holds the profile to I did not read), and the interface's example (`verity/core/protocols/interface/CoreProtocol/Example.lean:114-135`, a read-back). Nothing there was changed.

**What I did not check.**
- The proofs file at this head on a node. [run: on the VM, bare `lean` on a core-only model with the real `factsOf`, guards and the three lemmas' real text, `sizes/Guard7.lean`]: it elaborates, standard axioms only. The model has no Mathlib.
- The suites at this head, and the Lean half of the test at any head: `b488` is the first run that can reach it.
- A scan's or an untied audit's profile at the second and third refusals: the two new kinds aim at the tied audit only. `factsOf` is the one reader for all three [read: `Period.lean:1166`, `1177`], so they are held, but no case shows it.
- No kernel replay, no `check.py`, no lock. The lock's record of `RowsTiedCharged` is stale by the statement's change in `28528994d9`.

**Least sure of.**
1. The two new lemmas with Mathlib in scope: `simpa using this` and the `split` hypotheses found by `assumption` are where a real elaboration could differ from the model.
2. The "fewer" form of the sizes kind lowers the answer's sample by one as well as a stratum's size, so that case differs from the honest one in more than the strata's sizes. It is still one answer, and the mirror's reason is the sizes'.
3. The suites' cost on cpu-3 tonight: its research slots are full, each of my tries waited about four minutes for one, and `bbfd` is past an hour.

**One slip, mended.** Reading a run's status JSON at 7:07 PM I dropped the four fields at the top level only, and a nested copy of my own build's command and the node's address printed to my terminal. Nothing went to a note or a message. A helper now drops them at every depth (`sizes/rstat.py`).

**Standing.** Nothing on LEAN-249, no comment on NET-325 yet (it waits for the end), no base merged. Note updated and synced: `lanes/network-accounting/20261111T0217Z-state-net325-leanmerge.md`.

TIMER: none

