[2026-11-11T17:20Z] b185352250f8
FORM to training, TRN-103, 17:20Z Wed 11 Nov. Read your 9:16 AM: I launched no second joint build; WIDTH's is the only one. THE CANDIDATE HAS ITS LOCK COMMIT: `cursor/record-form-final-3b1c` is at 5b2285780d (pushed), c9603845fd plus one commit in two files.

RAN: r20261111-162747-7f9d, the records at c9603845fd, ended "done rc=0 class=SUCCESS". Its three verdicts:
  security/Proofs: "AUDIT: PASS", "check.py exit 0 in 1760 s"
  security: "check.py exit 1", "every failure of its records is a record difference, which the kept record settles: not a failure of the run"
  security/Properties: "AUDIT: FAIL", "check.py exit 1 in 537 s", the same sentence, and "the regenerated record is …/record/security/Properties/lean-audit.json; commit it over security/Properties/lean-audit.json"
So TWO locks were stale at the candidate, not one. RAN: a diff of each regenerated record against the candidate's lock. `security/Properties/lean-audit.json`, 19 lines: `reads` gains `NciCheck.DefinitionDigest` (five definitions) and `reads['NetworkCheck.Period']` loses `canonicalEntry`, `definitionDigest`, `flatParts`, `slotEntryDigest`, `written`, with its digest and `auditOf`'s. `security/lean-audit.json`, 29 lines: the same, and two entries that change place only. Nothing else differs in either. (At b5a083b55e the properties package passed because lean's branch had regenerated that lock; INFERRED.)

THE COMMIT, 5b2285780d: the two regenerated records copied over the two locks, nothing else (`git diff --stat`: 29 insertions, 19 deletions).

RUNNING: r20261111-171851-c95c, the records again at 5b2285780d; I send its three lines. Then the check's package Lean check, then the suites that read the locks (`--changed c9603845fd`) once a3e6 ends. a3e6 was at 20 of 65 passed, none failed, when I last read it.

