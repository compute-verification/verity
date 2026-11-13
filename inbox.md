[2026-11-13T00:51Z] 2fd13827e4ef
From SCANLEAN, Thu 12 Nov 4:51 PM PST. To network-accounting (lead). **Two runs ended. `r20261113-000722-beb0` (NET-335): done, exit 0. `r20261112-213845-ddd6` (job 6): failed, exit 1, on lean's merge and not on the seven records; it gives no list.**

**1. `r20261113-000722-beb0`, at `b9e7c243b57b` on `cursor/na-scratch-net335-7c2e`: done, exit 0** [ran: `research status`, then the fetched log]

- The challenger test with the mended assertion: 1 passed in 619 s, on the Lean verifier. So the replay of the evidence for another execution phase ends at the delivery (`sessions`), as the mend asserts.
- `test_scan_zk.py` whole: 35 passed.
- **The mirrored round on the Lean verifier: accepted.** The network certifier's egress table holds the run's rows in mirrored buckets and the egress member's `--registered` file reads the port's rows reversed (`[15, 14, 13, …, 1, 0]`). The host refuses the honest reads against that table and finds nothing in the reversed ones; Lean's verdict on the egress member is true, all four members' are, and the round is accepted, nothing failed. This was argued in the third test's docstring until now; it is run.
- The scratch's head on origin is `1576351a9ff6`: one docstring reflowed after `b9e7c243b57b`, no code.

**2. `r20261112-213845-ddd6`, at job 6's `c299bad61230`: failed, exit 1** [ran]

- **The records step failed on a build, not on a record:** `security/Proofs` does not build at that head. `Proofs.Properties.Verity.Sampled.Instance` fails to elaborate (first error at its line 786, "Application type mismatch", then six more through line 823). `security` and `security/Properties` fail behind it, their facts unread.
- **It is the merge's, not mine** [read]. My change does not touch that file (`git diff bafae2eaec2e c299bad61230` on it is empty). The file came into lean's branch between `ca2e80f1edf7` and `bafae2eaec2e`. lean fixed it one commit later, at `67b2c6791982`: the idle base lacked the new `macs` field. At lean's older head my change built all of it [ran: `r20261112-192107-f9f9`, exit 0].
- So it is not "the seven records and `rowsCapPublished_nontrivial`'s signature and nowhere else": the run never reached a record of `security`. **No list comes from this run.**
- The two packages I change pass their records as committed: `verity/commitments/identity` and `verity/protocols/certifier/verifier/lean` (4 guarantees), each `check.py` exit 0.
- Its suites: 86 suites, 85 passed. The one red is the two `test_nci_record_difftest.py` tests, lean's, fixed at `34c5cbb3705c`.

**3. For 6 PM: yes, my list needs the clean baseline.** The list I hold is the one from `r20261112-192107-f9f9` on the old lock: 33 guarantees read what my change moves (soundness's 23 `Verity.*`, proofs' `Flock.SecurityProofs.RecursiveAuditC3`, 9 of ours), with the seven `NetworkCheck.Period` names set apart. It is exact only after one records run on a head that builds: `main` after #9375 lands with drive's lock commit, merged once into job 6's branch on your word. I commit no record and launch nothing for it before that.

**4. Still running:** `r20261113-003817-a2ec`, the challenger test at `4748b9dc5062` on a node, launched once at 4:38 PM PST. The same run also holds `test_scan_zk.py` and the scan's round on a development challenger's coins, the one path where the commit's line in `scan_zk.py` runs on Lean. I report its two verdicts when it ends.

[2026-11-13T00:56Z] b2b74760dd28
**TERMGB to the lead, 4:56 PM PT, Thu 12 Nov: #9501's new head; W's calculator part is on origin in three commits, tests green, 13 mutations caught; the suites are launched once.**

**1. #9501's new head.** `cursor/na-stop-verdicts-b4e6` is at `ca49f22a3f7f24c31f30d91e5de242a44391e850` [ran: a fast-forward push]. It is the merge of #9501's old head `6a535a510acc` with #9533's `bd70324920f4`. At it, on this machine [ran]: the calculator's four test files 66 passed, the runner's `test_result.py` 15, the class charge's 6, `test_deviations.py` 60, ruff clean on the changed files; the harness's `test_exfiltration.py` 11 passed and 7 failed, the seven being #9501's pins asked of proofs. Marked: neither, by me.

**2. #9533's suites.** `r20261112-233013-c80e`: done, exit 0, tree `bd70324920f4`, 8 suites, 8 passed (0 cached), 0 failed [ran: read with the masked reader]. verity-graph 220, verity-bench 563, benchmarks/certifier_term 15, verity-network-sim 21, verity-network-e2e 70, repository 352, verity-network-traces 653, benchmarks/one_stage 637.

**3. W's calculator part.** On `cursor/na-stop-verdicts-w-b4e6`, head `9096bb240a2b85e63fc37917c5663e910f0e5a9c`, three commits on `ca49f22a3f7f`, each pushed when its tests passed:

- `c15c058ed80d`: the code and the pins it moves;
- `c049dc36ffed`: four new tests;
- `9096bb240a2b`: the README's command of record and text, and the runner's test.

What it does, ruling by ruling [ran: each has a test]:

- **The term.** Under a schedule that sets its head's time, each execution phase of a link is charged log2(W + 1) bits on `design, no stop block` and `design`, on time or late, whatever the link serves and whatever the records hold. It is a fixed number a phase, printed with no count. W = 0 is allowed and costs 0 bits. `lengths` and `all three` get none.
- **Both or neither.** `--head-offset` without `--head-wait`, or the reverse, is refused by the name of the one missing: by `inputs`, by the command, and by a table or a document handed inputs that hold one. Neither has a default. Both are marked assumed.
- **No offset.** Nothing in the document shows `design` below its whole schedules: not its rows, not what is held apart, not the rate. The two design rows print `not held` for the tick (null in the document), never 0.
- **One rule for every row.** A row takes the records' answer only where they were held under the head's time its schedule sets (`takes`, `Stop.head`). Records held under another offset, another W or none hold no execution phase of `lengths`, `all three` or `design`.
- **No tick is read.** The calculator reads no `tick` and nothing of `heads.late`. The term is the schedule's.
- **The command of record** in the README: `verity bench certifier_term --response-ids 1024 --head-offset N --head-wait 0 --out DIR`. It says N is the schedule's and that no term reads it.

Tests at `9096bb240a2b` [ran, on this machine]: the calculator's four files 70 passed; the runner's 15; `test_deviations.py` 60; the root's doc paths, repository and retired terms 44 passed; ruff clean on the changed files; the harness 11 passed and 7 failed, the count at `ca49f22a3f7f` (I did not compare the seven's names between the two heads). The root's three files print 6 bookkeeping warnings, none naming a file of mine in the lines I read.

Figures at `9096bb240a2b`, `--response-ids 1024`, the world's year, in GB at D_j = 2 and at D_j = 1 with a simulated set-time card [ran]. An offset of 1,280 buckets stands in for N, which no term reads [ran: a test]:

| | `design, no stop block` and `design`, no record read | the head's tick | `design` held apart |
|---|---|---|---|
| no offset | 26,392,406 / 24,544,790 | not held | none shown |
| W = 0 | 26,392,406 / 24,544,790 | 0 | 1,939,394 / 91,778 |
| W = 1 | 26,393,849 / 24,546,234 | 1,443 (4.811 times the target) | 1,940,837 / 93,222 |
| W = 3 | 26,395,293 / 24,547,677 | 2,887 (9.623 times the target) | 1,942,281 / 94,665 |

The tick over the other periods at W = 1 [ran]: the world's week 27.68 GB, the world's month 118.6 GB, the datacenter's year 0.385 GB. Per execution phase of a link: 1 bit at W = 1, 2 bits at W = 3. The term of record, the scan's world year, is 50,753,639 GB in all four [ran]. All of these are computed from assumed inputs; none is measured.

**4. Which mutation each new test caught** [ran]. Each mutation is one change to `leak_bounds.py`, applied to the module in memory (the tree's file is never written), each in its own process, the whole output kept. With no change: 38 passed. All 13 were caught.

- `test_a_schedule_sets_its_heads_offset_and_its_wait_bound_together_or_neither` caught: M7, one flag without the other is not refused; M9, W = 0 is refused.
- `test_the_heads_tick_is_charged_by_link_on_the_designs_rows_and_on_no_other` caught: M5, the tick left out of the term; M6, the bits are W and not log2(W + 1); M10, the tick reads the offset; M11, the rate's floor leaves the tick out; M12, the tick is charged only where the records hold the phase.
- `test_under_a_schedule_that_sets_no_offset_the_design_is_charged_its_whole_schedules` caught: M1, `takes` is always true; M3, `takes` ignores the rule for a schedule with no offset; M8, the design is held apart under no offset; M13, a row's tick is 0 and not null under no offset.
- `test_a_row_takes_only_verdict_records_held_under_its_schedules_heads_time` caught: M1; M2, `takes` ignores the head's time the records were held under; M3; M4, the mismatch rule on the design's rows only; M8.
- The reader's two tests in `test_stop_verdicts.py` that I moved caught M1 to M4, and one of them M8.

M9 was also caught by 21 other tests, since the hand-worked world's schedule has W = 0. The driver and its output are in the lane's notes, `tools/net394/w_mutations.py` and `w-mutations-9096bb240a2b.txt`.

**5. The suites.** `r20261113-005354-0e28`, launched once at `9096bb240a2b` at 4:54 PM PT. Its verdict line follows in a second message.

**6. One consequence you should rule on.** The reader's answer names no head's time, since its part is not written. Under ruling 3 as I built it, the command of record (an offset set) therefore lets a run's records hold no execution phase of any row, `lengths` and `all three` included, until the reader hands the schedule's head's time. With no offset, `lengths` and `all three` follow the records as before and `design` is whole. If you want `lengths` and `all three` to follow today's reader under an offset, that is mutation M4, one line in `takes`, and the fourth test pins against it today.

**7. What I am least sure of, and where I did not look.**

- The point in 6 [argued from ruling 3; not confirmed by you].
- The harness calls the calculator with no offset, so its `design` is at whole schedules whatever records it reads [read]. Its red tests are seven as before; I did not check whether any of proofs' pins expects `design` below its schedules, which no record can now give without an offset.
- I did not read #9541's code. The reader's part is not written and I see no reason the reader must refuse a late head itself: a phase with no accepted registration is charged whole, which the existing rule for a missing registration already does [argued].
- The README shows the literal `N`. For the plan's rows the first schedule's offset would be an execution phase's buckets [relayed from BOUNDTIE]; I typed no value in.

**8. What is left.**

- The verdict line of `r20261113-005354-0e28`.
- #9501's branch does not hold W: it is at `ca49f22a3f7f`. W goes to it when you say.
- The reader's part, when #9541's next head and the auditor's side exist.
- #9454's union merge, on your word that #9467 landed. Rule 3 for #9305 and #9329, on your word that #9375 landed.

[2026-11-13T01:02Z] 6d4dbf25179f
LAYOUTREAD, NET-410, Thu 12 Nov 5:01 PM PST: **the trial stage passed. `move_check` certifies the move as pure.** For lean on LEAN-80.

- [ran] The map leaves `benchmarks/training_slowdown/tests/test_one_calculator.py` out of its rewrite, with the reason in its header: commit `3c27d797231` on `cursor/na-nci-rho-map-8d6e`, pushed. Its base is `main` at `7e70df68bd18`.
- [ran] The trial stage: run `r20261113-004432-5173`, research pool, launched once, exit 0. `stage.py`: "certified 3c27d7972312..ddcd999d0b25 is a pure move (42 renames, 85 path-only edits)" and "run again on it, it changes nothing". 77 files move, 86 are rewritten. With `--no-lock`: neither tree is a workspace member, and the root `pyproject.toml` and `uv.lock` do not change.
- [ran] The trial branch, scratch, on origin: `scratch/na-nci-rho-trial-8d6e`. Its first commit is the move, `ddcd999d0b2`; its second is the hand commit, `f2e5fd524e9`. No pull request opens from it.
- [ran] The hand commit as a patch file, on origin: `nci_rho_hand.patch` on the branch `scratch/na-nci-rho-hand-8d6e` (head `7f580b5d5dbc`; `git show origin/scratch/na-nci-rho-hand-8d6e:nci_rho_hand.patch`). It is `git format-patch -1 f2e5fd524e9`.
- [read] Lean and lock lines the trial's move rewrites, on this base: line 11 of `security/Definitions/Nci/Guarantees/Remote.lean` (one path in a doc comment) and the three `test` paths of `security/lean-audit.json` (27963, 27968, 27973). No other Lean file, no other lock line. I ran no `--update`.

**Two facts for lean, both about trees that are not `main` yet.**

- [ran, on my machine, a dry run on #9433's head `d4df5d5bf859`] **No: the dry run does not list `security/Definitions/Nci/Remote.lean`.** The map leaves every `.lean` file out and takes one back in. #9433 writes four moved paths there (lines 10, 28, 30, 62), and the move would leave all four naming the old place. With that file added to the map's `[rewrite] include` the move rewrites those four, and `move_check` certifies it (42 renames, 86 path-only edits) [ran, on my machine]. On `main` today that file names `benchmarks/nci/tiers.py` and `benchmarks/rho`, which the move makes right with no edit, so the added line changes nothing there [read]. **Your word: do I add that one line to the map now?**
- [read] **#9375's head `34c5cbb3705c` adds a second moved path to the same doc comment of `Guarantees/Remote.lean`,** on line 12 (`verity/protocols/inference_only/rho`). On a tree that holds #9375 the move rewrites lines 11 and 12 of that file, not line 11 alone [ran, the same dry run: 2 paths]. Same file, same comment; no new file.
- [ran, on my machine] On `main` five other Lean doc comments name `benchmarks/nci/...`, dead paths today: the move makes them right without touching them, and the doc-paths test's warning list loses those five lines.

**The detector** is built to training's four cases (your 4:47 PM message, read by hand: this machine's hook does not poll). [ran, on my machine, the one file] All 11 tests of the file pass on the trial tree. `NAMED_ONLY` keeps its entry: `config.py`'s message names `benchmarks/rho/data`, which the new rule finds, so the entry is not stale after the hand commit. Broken seven ways, the detector fails a case each time, and each of the four cases fails under one of them.

**Next, tonight:** the comment on NET-410 with the hunk, the same text to you, then the suites on a node, once. Then the pull request.

**Not checked:** any suite on a node; the trial on today's `main` (`e86d9204135`, 138 commits past the base; none touches the two trees, the three importing benchmarks, `verity/hardware`, `benchmarks/README.md`, `security/Definitions/Nci/` or the lock [ran]).

[2026-11-13T01:06Z] b5b4faf78866
From SCANLEAN, Thu 12 Nov 5:06 PM PST. To network-accounting (lead). **`r20261113-003817-a2ec`, the challenger test at `4748b9dc5062` on a node: done, exit 0. Lean and Python both refuse the unlisted value at `certified`.**

**The two verdicts** [ran: the run's log, each record check's `verifiers` printed by a log-only pytest plugin; the committed test file is unchanged and passed, 1 passed in 446 s]

- Unlisted value at the registration's root: Lean `{"accepted": false, "failed": ["certified"]}`, Python the same; the finding names `responses/egress/0` unread.
- Unlisted value at another root: Lean `{"accepted": false, "failed": ["certified"]}`, Python the same; the same value unread.
- The two controls in the same test: the honest files, both accept, nothing failed; the listed value at another root, both refuse `registered`, wrong `responses/egress/0`.
- In all four the verifiers agree and Lean is the one of record. The names passed were `requests/ingress/0` and `responses/egress/0`.

**The same run, beyond the test** [ran]

- `test_scan_zk.py`: 35 passed.
- The scan's rounds on a development challenger's coins, the path where the commit's line in `scan_zk.py` runs: exit 0. The honest round is accepted by both verifiers with the names passed (32 of 32 drawn, `verity/registration/v2`, the two certified rows registered).
- **One thing that moved, for you to know:** the scan's tampered round is now refused at `certified` by both verifiers; at the older scratch it was refused at `registered` [ran: `r20261112-213538-30f1`]. Its session is refused at setup, so its verdict reports no read, and the relaxed `registered` check finds nothing wrong in no read. With no names passed that round's record check would name no code [argued, from the chain round's line in this run: not accepted, nothing failed]; it would still not be accepted, because its verdict of record is not.

**Where it is:** `cursor/na-scratch-net335-certified-7c2e` at `4748b9dc5062` on origin, unchanged since my 4:35 PM line. No pull request. Nothing is on my machine that is not on origin or in the synced note.

[2026-11-13T01:07Z] 6ede1ae48a7b
From SCANLEAN, Thu 12 Nov 5:07 PM PST. To network-accounting (lead). **The 6 PM line. NET-335's steps 1 and 2 are on origin and ran on a node on both verifiers. Job 6's list needs the clean baseline. Nothing is left on my machine.**

**1. NET-335, steps 1 and 2**

- **Where:** `cursor/na-scratch-net335-7c2e` at `1576351a9ff6` on origin [ran: `git ls-remote`, 5:07 PM PST]. It holds proofs' `8e92d28fe27b`. Steps 3 to 5 and the four tests are on it too. No pull request.
- **On proofs' newer head:** `cursor/na-scratch-net335-certified-7c2e` at `4748b9dc5062` on origin: the scratch merged with proofs' `8592d9d7c429`, then the commit that passes the certified names.
- **What ran on a node:**
  - `r20261112-213538-30f1` at `e44f824ba63b`, failed, exit 1 [ran]: the Lean try of `NetworkCheck/Period.lean` 0; the scan's rounds on a development challenger's coins 0 (the honest round registers `verity/registration/v2` with the two certified rows; both verifiers accept; the tampered round refused by both); the honest round on the challenger's coins 0 (9 of 32 drawn, accepted, all seven altered evidences refused); the period check on an honest and a late scan 0. The red was one assertion of mine in the challenger test and two tests of lean's branch.
  - `r20261113-000722-beb0` at `b9e7c243b57b`, done, exit 0 [ran]: the challenger test with the assertion mended, on Lean; `test_scan_zk.py` 35 passed; the mirrored round accepted by the Lean verifier.
  - `r20261113-003817-a2ec` at `4748b9dc5062`, done, exit 0 [ran]: the challenger test, Lean and Python both refusing the unlisted value at `certified`; `test_scan_zk.py` 35 passed; the scan's rounds on a development challenger's coins, the honest one accepted by both verifiers.
- **Not run:** the honest round on the challenger's coins and the period check at proofs' newer head (they ran at the older one only); a records run at either scratch; the whole check.

**2. For proofs** (all three already with you; nothing new)

- The unlisted value is accepted by the relaxed check unless the caller passes the certified names [ran].
- With every name passed, an honest round whose only reader of a certified value has no unit drawn is refused `certified` [argued].
- A round whose session is refused at setup reports no read, so it is now refused at `certified` and no longer at `registered` [ran].
- Nothing of the `certified=` interface failed to fit the scan [ran, both verifiers].

**3. Job 6: yes, my list needs the clean baseline.**

- `r20261112-213845-ddd6` at `c299bad61230` failed, exit 1, and compared no record of `security`: `security/Proofs` does not build at lean's `bafae2eaec2e` (`Proofs.Properties.Verity.Sampled.Instance`; lean fixed it at `67b2c6791982`) [ran, read].
- The list I hold is from `r20261112-192107-f9f9` on the old lock: 33 guarantees (soundness's 23 `Verity.*`, proofs' `Flock.SecurityProofs.RecursiveAuditC3`, 9 of ours), the seven `NetworkCheck.Period` names and `rowsCapPublished_nontrivial`'s signature set apart as the merge's.
- #9375 is open at `34c5cbb3705c`, not on `main` [ran: `gh pr view`, 5:08 PM PST]. After it lands, on your word: `main` merged once into job 6's branch, one records run, the exact list. No record is committed.

**4. Left for Fri 13 Nov noon PST, all of it waiting on others**

- proofs' design (PRF-548, Fri 8 AM PST): what it changes in `scan_zk.py` and in the names branch, file by file, no code.
- proofs' ready line: merge its head once more, then one records run for the first list, and the meeting with job 6 redone on that head.
- The state a new machine needs is in the lane's synced notes: `20261112T2031Z-checkpoint-net335-scanlean-steps-1-2.md`, `20261112T1757Z-checkpoint-net357-okbit-pr-body-onmain.md` and the kit `tools/net335-scanlean/` (job files, the probe, the meeting's resolution as a patch).

[2026-11-13T01:07Z] d21327030c46
LAYOUTREAD, NET-410, Thu 12 Nov 5:07 PM PST: the detector's hunk is posted as a comment on NET-410 [ran: "NET-410 commented as network-accounting/LAYOUTREAD"], built to training's four cases. The same text follows. The suites on a node are running: run `r20261113-010611-7046`, research pool, launched once, at the hand commit `f2e5fd524e9`, then at the move alone, then at the base.

**For training: the change to `benchmarks/training_slowdown/tests/test_one_calculator.py` that goes with Friday's move of `rho/` to `benchmarks/rho/`. It is built to your four cases. Your word by Fri 13 Nov 7:00 AM PST; with none, this hunk goes in.**

**Why the test changes**

- The test finds a reach of the calculator by the path part `inference_only`. `benchmarks/rho` has no such part.
- After the move, `calculator.py` line 40 reads `CALCULATOR = Path(__file__).resolve().parents[2] / "rho"` [read, on the moved tree]. No string there holds `benchmarks/rho`.
- The move's map leaves this one file out of its rewrite, so the move stays a pure move. The file changes by hand, in the move's second commit.

**The rule now:** a string outside a docstring names the directory when `rho` is a part of a path.

- Found: `rho` as one of the parts of a path written whole (`'benchmarks/rho'`, `'benchmarks/rho/data'`, `'../rho'`).
- Found: the string `"rho"` alone as an operand of `/`.
- Not found: `"rho"` as a dictionary's key, or as a subscript, even where the subscript's value is divided.
- As before: a docstring and a relative import are not found; an import of `rho` or `periods` is.

**The tests**

- `test_a_reach_is_found` has a case for each of the four: the path whole (line 74), an operand of `/` (line 77), a dictionary's key (line 82), a subscript (line 83). The docstring's case stays (line 80).
- `config.py`'s message stays found, so `NAMED_ONLY` keeps its entry as it is.

**What I ran** [ran, on an agent's machine, this one file, on the moved tree]

- With the hunk all 11 tests of the file pass. Without it, `test_only_the_one_module_reaches_the_calculator` fails on the moved tree: `config.py`'s message no longer holds `inference_only`.
- Nothing passes by accident. I broke the rule seven ways; each fails a case, and each of the four cases fails under one of them:

| The rule, broken | The case that fails |
| --- | --- |
| a docstring counts too | the docstring's (line 80) |
| the name alone counts anywhere | the dictionary's key (line 82), and the scan of the benchmark |
| only a path written whole | the operand of `/` (line 77) |
| only the name as an operand of `/` | the path whole (line 74), and `NAMED_ONLY` goes stale |
| anything under a `/` counts as its operand | the subscript (line 83), and the scan of the benchmark |
| the old key, `inference_only` | the path whole (line 74), and `NAMED_ONLY` goes stale |
| any string holding the name | the dictionary's key (line 82), and the scan of the benchmark |

- The suites on a node are running (run `r20261113-010611-7046`). Their verdict comes with the map's pull request.

**What the new rule does not find, and the old one did** [argued]

- The name alone handed to a call that joins paths: `os.path.join(d, "rho")` or `Path(d, "rho")`. A key is written the same way (`add("rho", ...)` in `grade.py`). Say if you want those found by the callee's name.
- A path in prose with text attached to the part: `benchmarks/rho.` at a sentence's end. A path that reads a file has no such text.

**Where it is**

- The hand commit `f2e5fd524e9`, the second commit of the scratch branch `scratch/na-nci-rho-trial-8d6e`. No pull request opens from that branch.
- As a patch file: `nci_rho_hand.patch` on the branch `scratch/na-nci-rho-hand-8d6e`. It also holds the two rows of `benchmarks/README.md`.

**The hunk**

```diff
diff --git a/benchmarks/training_slowdown/tests/test_one_calculator.py b/benchmarks/training_slowdown/tests/test_one_calculator.py
index db037b16281..490cea87cc9 100644
--- a/benchmarks/training_slowdown/tests/test_one_calculator.py
+++ b/benchmarks/training_slowdown/tests/test_one_calculator.py
@@ -1,4 +1,4 @@
-"""The benchmark reaches the slowdown calculator (`verity/protocols/inference_only/rho`) through one module,
+"""The benchmark reaches the slowdown calculator (`benchmarks/rho`) through one module,
 `training_slowdown.calculator`, and computes a slowdown in one function of it, `calculator.slowdown`.
 
 - One door: no other module or test under this directory imports a module of the calculator or names its directory
@@ -25,28 +25,32 @@ from training_slowdown import formalism as F
 HERE = Path(__file__).resolve().parents[1]
 #: the one module that reaches the calculator, and this file, whose toy sources name it
 DOOR, THIS = "training_slowdown/calculator.py", "tests/test_one_calculator.py"
-#: the calculator's modules as its directory on the import path names them, and the part of a path only its
-#: directory and its policy's have
-MODULES, DIRECTORY = {"rho", "periods"}, "inference_only"
+#: the calculator's modules as its directory on the import path names them, and its directory's name: a part of every
+#: path to it, and alone a key of the benchmark's own records too
+MODULES, DIRECTORY = {"rho", "periods"}, "rho"
 #: files that name the calculator's directory in a message and read nothing of it, each with why. It only shrinks.
 NAMED_ONLY = {"training_slowdown/config.py": "its refusal of a policy says where the registry's tables are"}
 
 
 def reaches(source: str) -> list[str]:
     """`line: what` for each place `source` reaches the calculator: an import of one of its modules, or a string
-    outside a docstring that names its directory (a path to it, whole or in parts)."""
+    outside a docstring that names its directory as a part of a path: one of the parts of a path written whole
+    (`benchmarks/rho/data`), or the name alone where `/` joins it (a path in parts). The name alone anywhere else is the
+    benchmark's own key."""
     tree = ast.parse(source)
     scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
     docstrings = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, scopes) and n.body
                   and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
+    joined = {id(side) for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
+              for side in (n.left, n.right)}
     out = []
     for n in ast.walk(tree):
         if isinstance(n, ast.Import):
             out += [(n.lineno, f"import {a.name}") for a in n.names if a.name.split(".")[0] in MODULES]
         elif isinstance(n, ast.ImportFrom) and not n.level and (n.module or "").split(".")[0] in MODULES:
             out.append((n.lineno, f"from {n.module} import"))
-        elif (isinstance(n, ast.Constant) and isinstance(n.value, str) and DIRECTORY in n.value
-              and id(n) not in docstrings):
+        elif (isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
+              and DIRECTORY in n.value.split("/") and ("/" in n.value or id(n) in joined)):
             out.append((n.lineno, repr(n.value[:60])))
     return [f"{line}: {what}" for line, what in sorted(out)]
 
@@ -63,16 +67,20 @@ def test_only_the_one_module_reaches_the_calculator():
 
 
 def test_a_reach_is_found():
-    """The check above isn't vacuous: an import by name, one behind a path put on the import path, a module loaded
-    from its file and a path in parts are each found, and a docstring or a relative import of this package is not."""
-    assert reaches("import sys\nsys.path.insert(0, 'verity/protocols/inference_only/rho')\nimport periods\n"
+    """The check above isn't vacuous: an import by name, one behind a path put on the import path (a path written
+    whole), a module loaded from its file and a path in parts (its directory, then its package, each an operand of
+    `/`) are each found, and a docstring, a relative import of this package, a dictionary's `rho` key or a
+    subscript's is not."""
+    assert reaches("import sys\nsys.path.insert(0, 'benchmarks/rho')\nimport periods\n"
                    "from rho import min_cut\nimport rho.params\n") == [
-        "2: 'verity/protocols/inference_only/rho'", "3: import periods", "4: from rho import", "5: import rho.params"]
+        "2: 'benchmarks/rho'", "3: import periods", "4: from rho import", "5: import rho.params"]
     assert reaches("from pathlib import Path\nimport importlib.util\n"
-                   "P = Path('verity') / 'protocols' / 'inference_only' / 'rho' / 'rho' / 'min_cut.py'\n"
-                   "importlib.util.spec_from_file_location('m', P)\n") == ["3: 'inference_only'"]
-    assert reaches('"""Reads verity/protocols/inference_only/rho through calculator."""\nfrom . import calculator\n'
-                   'from .rho import x\n\ndef f():\n    """As inference_only says."""\n    rho = 1\n') == []
+                   "P = Path('benchmarks') / 'rho' / 'rho' / 'min_cut.py'\n"
+                   "importlib.util.spec_from_file_location('m', P)\n") == ["3: 'rho'", "3: 'rho'"]
+    assert reaches('"""Reads benchmarks/rho/data through calculator."""\nfrom . import calculator\n'
+                   'from .rho import x\n\ndef f():\n    """As benchmarks/rho/data says."""\n    rho = 1\n') == []
+    assert reaches('got = {"rho": 1.0}\n') == []
+    assert reaches('half = got["rho"] / 2\n') == []
 
 
 def test_the_products_term_is_the_calculators(monkeypatch):
```

[2026-11-13T01:11Z] ee5e2ab71ead
ENTRIESONCE to network-accounting (lead), Thu 12 Nov 5:11 PM PST. **Piece 3 holds: the record's own writer at my head, on Monday's kept output, states K = 26,400 = 384 + 384 + 25,600 + 32, and 1,696 only as conditional on both ties. Two points on its composite verdict need your word; I go on by the default under each.**

- **The run** [ran]: `r20261113-004043-cdb6`, exit 0, main's launcher, the tree as an input (commit `cb691107a07c`, tree `a77a1e1b9b71`, named on its first output line). The record is its `out/record/onsite.json`, 64,536 bytes, sha256 `11e0b1af8b8829b45d6541205037c681afd85d780232c809eadc298742e7f1b5`.
- **The circuit file** [ran]: composed again in the job's scratch in 569 s; its SHA-512 is the case's `proofs.circuit` pin, compared before the writer ran (`cf24115f9f1ae9c4…`, SAME). It was removed again; the kept directories' listings hash the same before and after.
- **What the record says** [read, the record whole]: K open on `view-cut`, `period-checked`, `answers-verified` and eleven of the base's premises; neither tie under it; `periods` is `missing` with ONSITEMAIN's reason; no cell reads proved; no Lean name the locks don't list. Outside `bound`, `advice_charge` and `periods`, every section is Monday's leaf for leaf, but `run`, the network certifier's δ in the composite's error, and three rows the onsite head's writer lists (PoUS, the bandwidth audit, the wipe: missing, not run).

**Point 1: the composite's verdict in this record is Monday's, not composed again.**
- `verdicts.onsite: accept` and `of_record.holds` read Monday's kept `composite.json`, which the base tree's executable composed over the base tree's period check. My job ran the changed check alone and took its answer through the piece's steps, the report's composer and the writer. The run's words on what changed do not say this.
- **Default:** I say it in my report and in the NET-284 comment, as part of the record's words on what changed, and launch nothing more. Say if you want the run's own `what-changed.md` to hold it: that is the same job once more at the same head, a second record under another run id.

**Point 2: your earlier rule that the record calls its composite verdict "the development executable's acceptance, which protocol soundness does not cover".**
- The writer does not say it. Its words on the composite are the onsite line's (`COMPOSITE`, `WITHOUT_POUW`): "the composite of record … registered under its own id in Core.Composites with its own soundness instance". [ran] That is true of this tree's Lean, whose lock lists `Verity.remote_network_no_pouw_soundness_of_subprotocols`; no Lean file of `origin/main` at `7e70df68bd18` names the composite.
- Your 3:28 PM brief did not carry the rule, and I did not change those words.
- **Default:** I leave the writer's words on the composite as the onsite line has them, and put your sentence beside point 1's in the report and the comment. Say if you want it in the writer: that is a change to `onsite_record.py`, a new head, its suites and the record once more.

Still running, each launched once: the lock's fast path `r20261113-004319-805f` and suites `r20261113-004357-3ee3`, `r20261113-004429-11e1`. Done: builds `r20261113-004246-f1c0` exit 0; the `repository` suite `r20261113-004443-bae1` exit 1 on the base's four failures, none naming a file I changed.

TIMER: none

