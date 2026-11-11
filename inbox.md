[2026-11-11T16:16Z] efaf19374dee
FORM to training, copy COUNTS and WIDTH, TRN-103, 16:15Z Wed 11 Nov. THE FIRST JOINT BUILD PASSES FOR THE PROOFS PACKAGE. The integration branch has moved.

RAN: r20261111-153537-2590, `lean_changed.py --records security/Proofs` at b5a083b55e (all thirteen of the width check with `DecodeKeys`, `LeafWidthsDecoded`, `arg_wide`, `RecordPrim`, `RecordProgram`, on lean's c4b1da1fdb, with `PartInner` and `BodyInner` one definition by `export`). Its lines for the package:
  "AUDIT: PASS; reports in /workspace/research/runs/r20261111-153537-2590/lean-audit"
  "lean-records: security/Proofs: check.py exit 0 in 1563 s on … (23 files changed there), without the kernel replay or the tests: a lean-fast/v1 result, which check and research merge refuse"
So: the build, the seal, the records and the axioms of the proofs package, at that head; not the kernel replay.
The run has not ended: it went on to the packages that require it. For `security` it printed "AUDIT: FAIL", then "lean-records: security: every failure of its records is a record difference, which the kept record settles: not a failure of the run". READ: that is the difference lean's fast path names on #8962 (the reads of `Flock.SecurityProofs.RecursiveAuditC3`); I have not compared the two records byte for byte. I send the run's last line when it ends.

PUSHED: `cursor/record-property-laidout-3b1c` is at b5a083b55e on origin (it was cf9b216bd9). ONE DEFINITION BY `export` stands, as you accepted on this verdict. COUNTS, WIDTH: `LeafWidths.PartInner` and `LeafWidths.BodyInner` are the `BodiesOk` ones from this head on, and `partInner_of`, `bodyInner_of` are gone; a merge must not bring them back.

NOT IN THAT BUILD, each compiled alone only: `RecordTemplates.lean` (f804), WIRED's `RecordRows.lean` and `RecordInputs.lean`, COUNTS' `RecordWire`, `RecordFields`, `RecordWide`, `RecordLeafWide`. They are merged on `cursor/record-prim-operands-3b1c` (8f25d696c2; `RecordLeafWide` goes in next). The second joint build, of that head, starts when this run frees its slot.

No file of #8962's own moved (RAN): c4b1da1fdb is an ancestor of b5a083b55e, and `git diff c4b1da1fdb b5a083b55e -- security/Properties/Nci/Record.lean security/Proofs/Properties/Nci/Record.lean` is empty. The rest of `security/Properties` differs from lean's head only by main, which the integration branch holds later than lean's branch does (INFERRED from the file list, not read file by file).

