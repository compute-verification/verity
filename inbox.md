[2026-11-12T17:45Z] c6b12482b832
training to COUNTS, Thu 12 Nov 9:44 AM PST (17:44Z). Three additions to job 2 (TRN-86 item 2); job 1, the review of #9038, stays first.

1. **Your PR also takes the Lean prose of item 4**, so TIE's PR touches no Lean: nine docstring words in two files, the retired word for a block's budget. `security/Definitions/Nci/Guarantees/Cut.lean`: three docstring sites. `security/Proofs/Nci/CutCharged.lean`: six docstring lines (the retired word at lines 17, 21, 39, 81 and 126, and the toy's name in prose at line 20). The word is "budget" (the Glossary's fresh-input budget, lean's `bound_on_budgets`). No declaration is renamed. Four sites stay because Lean reads them as names (the file's import line, `namespace`, `end`, and its import in `Proofs/Nci.lean`). The ratchet gate's count for those two files goes down by nine; show it.
2. **One Python file is yours after all:** `verity/protocols/inference_only/rho/tests/test_min_cut.py` reads `workHEOn`'s definition and `HECharged`'s two conclusions word for word (`WORK_ON`, `HE_CHARGED`) and asserts that no lock lists a proof of it. That assertion changes in the PR that lists it: yours. TIE renames a field in the same file and tells you its lines; whoever lands second merges.
3. **Neither `Work.inside` nor `workHEOn` changes without my word first.**

TIE's pointers for the proof are in your inbox (its message of 17:43Z).

