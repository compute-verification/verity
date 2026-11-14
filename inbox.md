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

