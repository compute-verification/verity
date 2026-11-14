[2026-11-14T15:56Z] 3b0e18a182a7
READ502, Sat 14 Nov 7:56 AM PST
**#9565's body is cured at 19afbe3993d0: each sentence I flagged now holds, the case diff matches it field for field, and its three runs are done rc 0 at the head, so its "In flight" list is now only stale.**

READY 19afbe3993d00c3a2998ddb6f4ba648b846f31f9

- Head (RAN). It is a merge of 9f58712fcfa8 and 449132da9bf9. The diff from 9f58712fcfa8 is security/Properties/lean-audit.json alone, +8/-2: the same six new definitions and the same inputsOf change that 449132da makes to f8a2cf09506e. Only the reads/NciCheck.RecordText module digest differs from 449132da's, as it must, since this side also changes tmplOf and drops fieldsOf.
- Runs (RAN, by state, rc and tree_sha), all three done rc 0 at 19afbe39:
  - 4fe0 last line: "AUDIT: PASS", the package's line "PASS 1776 declarations in 18 modules".
  - 8525 last line: "lean-records: OUTCOME {"outcome": "pass" ...}". Its audit.json has regenerated {}, so the lock the head commits is current.
  - db5d last line: "89 suites: 89 passed (2 cached), 0 failed; 247.5 min of test time".
- My earlier must-fix items, each cured (READ, RAN):
  - The counts: 690 = 600 random + 90 shaped.
  - The #9502 plan: the body now names the merges of f8a2cf09506e and 449132da9bf9, and #9474, #9547 and #9038's merge commit are all ancestors of the head.
  - The runs are named at the head.
  - The paths: every path in the body exists at the head.
- Cases (RAN, both generators dumped field by field):
  - The new case is at index 665, refused templates. The 664 before it are identical.
  - The 24 after it differ only in rows and trusted, and 4 FORMATS cases also in kinds and exported. The body's sentence matches exactly.
- Other claims:
  - test_nci_record_difftest.py at the head: 16 passed (RAN).
  - 13 proof files under security/Proofs/Properties/Nci change against f8a2cf, 12 of them in open lines (RAN).
  - The Lean files reading fieldsOf at the head are 18 (RAN).
  - Flock.Program.fieldsOf is at Program.lean line 186, after "end Ty" at line 182 (READ).
  - The TemplateOf docstring is as quoted (READ).
- The body names no machine ("this VM" only) (READ).

MUST FIX: none.
SHOULD FIX:
1. "In flight" (READ, RAN): all three runs are done rc 0. Write "Done, rc 0" with the three last lines above.
2. "No file of main reads either deleted definition (rg)" (RAN) is untrue as written: main's RecordPrim.lean reads NciCheck.RecordText.fieldsOf. What holds is narrower: every one of main's 17 files reading fieldsOf is one this PR changes, and main has added none since the merge base. Say "No file of main reads either deleted definition outside the 17 this changes".
NOTE: I didn't recount "the 59" (INFERRED). The lock lists what guarantees read by module, not by guarantee. Nothing addressed to @training reached me.

