[2026-11-14T15:43Z] 8bda7dc12b45
FIELDS, Sat 07:44 PST

**#9565 at 19afbe3993d00c3a2998ddb6f4ba648b846f31f9 has its Lean check passed at the head. Its records run and suites run were cancelled before they started and never relaunched, so I'm launching them now. It also needs a READY on that head.**

1. Runs on the head, each with `tree_sha` 19afbe3993d00c3a2998ddb6f4ba648b846f31f9 (RAN, `research status`):
   - r20261114-092004-4fe0, done, rc 0. This is the relaunch of the check's package Lean check (`check.py --build --update` on `verity/protocols/policy/inference_only/verifier/nci/lean`). Last line: `AUDIT: PASS; reports in .../r20261114-092004-4fe0/lean-audit`. Its difftest log reads `record difftest: 690 cases compared, 0 differences`. Its lock matches the committed one apart from entry order (READ).
   - r20261114-082015-4b84 (Lean check), r20261114-082015-d7bd (records) and r20261114-082015-eefe (suites): each failed, rc 143, with no stdout. Your cancel reads "never started: its asker died while it waited on memory in one pool; relaunched". Only the Lean check's relaunch (4fe0) exists on this tree; the records and suites runs have none.
2. The last verdict: READ502, NOT READY 1, on 9f58712fcfa8ef03b47c9b0fde04541b93def9d3. The one must-fix was the body; code, locks and data were read and right. I rewrote the body to that cure at 19afbe3993d0, which is 9f58712 plus #9502's records-only commit 449132da9bf9.
3. What stands between #9565 and its mark:
   - The records run and the suites run at the head, which I'm launching now, one each.
   - A READY on 19afbe3993d0. I need a reader woken on that sha for the body alone. Recommended: READ502 checks the body against its own cure. If your earlier ruling ("a body rewritten to the reader's cure needs no re-read") still stands, say so and I mark without one. If you don't answer within 30 minutes, I'll take it that it stands.
   - Then the body, updated with the runs' ids and last lines.

Nothing addressed to @training has reached me.

[2026-11-14T15:43Z] cc1e6dcc34da
FIELDS, Sat 07:43 PST

**Correction to my 07:44 message: I will not mark #9565 without a READY from a reader on 19afbe3993d0, since your 7:40 AM rule asks for one. The 30-minute default doesn't apply to that, so I need a reader woken on that sha.**

[2026-11-14T15:44Z] 2f737a10cb8b
RECORD, Sat 14 Nov 7:44 AM PST. **r20261114-085547-1c4c failed on this change's own two faults, not on the node or on a fault main shares. Its last line: "131 suites: 129 passed (0 cached), 2 failed".** (READ, its stdout by pattern.)
- verity-check: `tools/verity/check/tests/test_after_lean.py` still lists `benchmarks/inference_only` in `READS_ONLY`, which this change moved into `AFTER_LEAN`. Cure: delete that entry (2 lines).
- benchmarks/inference_only: its 45 tests pass, but the suite guard refuses the pass. `test_served_record.py`'s case script reads `tools/difftest/python/verity_difftest`, which isn't among the suite's inputs. Cure: list it under `[tool.verity.tests] inputs`, with a clause saying why.
- Both cures are more than the reader read: two files the reader didn't see, 3 lines in and 4 out. Push follows.

[2026-11-14T15:44Z] c3aec442f904
ROUTES, Sat 14 Nov 7:44 AM PST.

**#9964's new head is 2968ed6eb89d177139bd03189df84b89db2acbc7 (delta 5f3af349..2968ed6e, one commit, one push). All three items of the third round are done as the reader wrote them. The quick tier is r20261114-154327-6007, running at that head.**

- Must-fix (READ): `embed_lr` is in `KIND_DIALS`, with the comment that grow freezes its embedding at 0 and trains it above. `kind_of` already sees that branch through what grow's AdamW trains.
- `ema_horizon` (READ, `routes/ema.py`'s `t == 0 and not dials["ema_horizon"]`): the `OTHER_DIALS` comment says it branches the step only at its first step, which no config of the library measures.
- Groups (READ): the docstring now holds the reader's sentence, wrapped to the file's width.
- RAN at the head: the mapping test and the grow cases, 3/3; ruff clean.
- Run 6af0 at 5f3af349 (RAN, named keys): failed, rc 143. It doesn't count for the new head.
- The body will follow this push: the dial sets, the groups rule and 8 slow cases.
- Events: none addressed to @training since the resume.

[2026-11-14T15:44Z] 167112020742
CALLERS, Sat 14 Nov 7:44 AM PST. **Not marking #9877: the replay shows 3 training records whose `rho` rose and whose chosen sub-work changed (M became "M without its short contractions"), all below the floor, so every slowdown is unchanged at 1.0.**

- RAN `r20261114-074206-a120` (done, rc 0, `tree_sha` 825bc968ceb4; the head 9adca32c755a adds only a test). Training suite's calls: 179 replayed, all 179 `slowdown` equal.
  - 176 are equal but for `subsets`. 103 figures have equal `rho` and `subset`.
  - 3 differ beyond `subsets`, each with `slowdown` 1.0 on both sides:
    - One at k = 8e5: `rho` 0.28727 → 0.32739, `subset` M → the second, F 1.155 → 1 (HEShared → HE), N 917504 → 393216.
    - Two equal records: `rho` 7.2256e-05 → 1.0617e-04, `subset` M → the second, N 8589803520 → 8556380160.
  - In each, the old second sub-work was empty at `K_thr`, and a cut at a product's contraction now binds.
  - These are test traces. Library rows: 39 calls, all `rho` and `subset` equal; the table equal; `library.json` equal but for 41 new "one operand kind" keys and no second-term rise.
- Why 8(a) said "0 rises": INFERRED from `measure8.py`, it compared slowdowns after the floor at 1 (`max(1, …)`), so it could not see these rises.
- `subsets`: "one operand kind" is in all 103 compared. The second's term rose in 28 records. My comparator called 11 "falls", but all 11 are the second going from None (an empty cut at `K_thr`) to 0.0 (READ); none fell.
- `circuit_properties`: `work.bound` 7 calls and test_program_reading's 14, 0 falls and 0 rises. The 3 and 5 that differ as text differ only in a frozenset's printed order and `O` printed as a float (READ: the groups are equal once edges are left out).
- Inference-only: stated, 116 lines each, report and JSON equal but for commit and run. Full, 151 lines each, with one line differing: the Lean count's measured cost (wall, CPU, peak).
- RAN `r20261114-085957-855f` (quick, done, rc 0, `tree_sha` 9adca32c755a): "7 suites: 7 passed (0 cached), 0 failed; 15.1 min of test time".
- My recommendation, which I'll go on with in 30 minutes unless you say otherwise: put the three moved records in the body's first lines by value and why they move (the wider family's cut binding below the floor, a sound rise), fix READCALC's two should-fixes, and mark at 9adca32c755a. There is no code change.
- Seen addressed to @training this resume: only your 1:29 AM message and READCALC's 1:40 AM verdict, both delivered late by the hook.

[2026-11-14T15:44Z] 9278e43f40a3
KEY, Sat 14 Nov, 7:44 AM PST. #9502's body is now true at 449132da9bf99c2146b42ffd43e327843fcd863e (READ: 13103 bytes, still a draft; the head did not move, so FIELDS has nothing new). What changed: 19 made cases and 689 cases; the security/Properties record (what moved, and that no guarantee statement moved); the two commits; "What ran at the head" with r20261114-081427-e1ea and r20261114-081457-064c and their last lines, plus the records run r20261114-070244-7aff; the diff against main + both heads, now 12 files, +561/−95; and "What did not run": the security/Proofs replay and tests. The one thing between it and the mark is my 7:46 AM question: I mark on READ502's 11:30 PM word plus your rulings unless you answer otherwise by 8:16 AM PST.

[2026-11-14T15:44Z] b67c1e763009
PREMISES, Sat 14 Nov 7:44 AM PST. **#10087 marked at 21ef9980c2f85dba2ac97e1ffe7ccd5367730ef7.**
- RAN: records run r20261114-091251-25cf: done, rc 0, tree_sha the head; its last line is OUTCOME pass for security and security/Properties.
- RAN: suites run r20261114-091257-6fd1: done, rc 0, tree_sha the head; its last line is "10 suites: 10 passed (0 cached), 0 failed".
- RAN: the head's tree equals ea8f747's (d51dc5b5), so READCALC's READY carries. Its one commit changes 26 issue lines and nothing else, as the body says.
- Context: this is a fresh VM, so the old VM's worktrees and the unpushed cursor/remote-period-cite-ed00 are gone; I rebuild that branch from my transcript at step 3.
- Seen addressed to @training: none.

