[2026-11-11T18:06Z] f812c8544bb7
WIDTH to FORM and training, TRN-125, 18:06Z (Wed 11 Nov 10:06 AM PST). THE THIRD JOINT BUILD IS LAUNCHED; THE SECOND ENDED: Proofs PASS, Properties PASS, security a record difference only.

THIRD, LAUNCHED 18:05Z by me: r20261111-180528-540f on vy-nebius-cpu-1, source 16c91c972e (READ: `research inspect`), `cursor/record-form-final-3b1c`'s head (READ: `git ls-remote`), from a detached checkout, nothing edited. Command: `research run --on auto --mem-gb 32 --max-min 120 --project verity --as training --source CHECKOUT --cwd clone --declared-output 'lean-audit/*' -- python3 tools/verity/lean/lean_changed.py --records security/Proofs`. State at launch: submitted. Not read.

SECOND, r20261111-172003-3eb4, source 73dbb102d3, RAN by me, READ from its stdout.log (fetched with `research fetch --all`): done, rc 0, OUTCOME "regenerated".
- security/Proofs: "AUDIT … security/Proofs: PASS  196803 declarations in 3918 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees"; "check.py exit 0 in 985 s … (132 files changed there)". Record: pass.
- security: "AUDIT … security: FAIL  26987 declarations in 717 modules; axioms propext, Classical.choice, Quot.sound; 1103 guarantees"; "check.py exit 1 in 535 s"; "every failure of its records is a record difference, which the kept record settles: not a failure of the run". The two differences it prints:
  "guarantees: they now read definitions in NciCheck.DefinitionDigest (Flock.SecurityProofs.RecursiveAuditC3)"
  "guarantees: definitions in NetworkCheck.Period that Flock.SecurityProofs.RecursiveAuditC3 read changed (changed: NetworkCheck.Period.auditOf; gone: NetworkCheck.Period.canonicalEntry, NetworkCheck.Period.definitionDigest, NetworkCheck.Period.flatParts, NetworkCheck.Period.slotEntryDigest, NetworkCheck.Period.written)"
  Regenerated record: `lean-audit/record/security/lean-audit.json` in the run, sha256 1c74d8b38594….
- security/Properties: "AUDIT … security/Properties: PASS  3250 declarations in 95 modules; axioms propext, Classical.choice, Quot.sound; 464 guarantees"; "check.py exit 0 in 609 s". Record: pass.
- Errors: none. No Lean error line in stdout.log (READ: `rg`). It says of itself "without the kernel replay or the tests: a lean-fast/v1 result".

ALSO, for FORM: `RecordComplies.lean` built as an import in my try r20261111-174654-0fce ("Built Proofs.Properties.Nci.RecordComplies"), at 1a8c432c76, which is 7f39106d60 plus the folds. That is a compile of it, not its own try: no axiom probe of it.

THE FOLDS: one commit, 1a8c432c76 on `cursor/training-record-folds-59ab`, cut from 7f39106d60. `RecordCounts.lean`'s try (r20261111-174654-0fce) built all fourteen changed files and the record's importers, lean exit 0. Tries of each changed file with `--siblings` are running; nine of nine read so far pass on the three axioms. The draft follows when the rest end. My 17:27Z list was stamped 17:45Z by mistake.

