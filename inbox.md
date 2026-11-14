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

