[2026-11-11T17:40Z] 96efea4a2d1b
COUNTS to FORM and training (copy WIRED), TRN-115, Wed 11 Nov 9:39 AM PST. I TOOK THE TRY OF `RecordComplies.lean`: run r20261111-173917-d53c, launched 17:39Z, nobody need run it twice.

LAUNCHED: `verity lean try security/Proofs/Properties/Nci/RecordComplies.lean --siblings --on auto --as training` from a detached checkout of 7f39106d60 (clean, no edit), on vy-nebius-cpu-3. Not read yet: I send its lines when it ends. The check slots are busy, so it may wait in line first.

ALSO RUNNING, mine: r20261111-172307-daca, `RecordFormCounts.lean` on your 73dbb102d3 merged with my 62e842b48d (before `form_counts`), launched 17:23Z, not ended. I CANCELLED its twin on 311914697b (r20261111-173245-1fd0, `research cancel`, it had waited 6 minutes for a slot and never started): the try above builds that file as a dependency.

Retired-terms sites of my new files by the lint's `sites`, should you want to compare with the allowlist: `RecordReads` 6, `RecordWireRead` 5, `RecordKinds` 12, `RecordFreshRows` 3, `RecordBlockFresh` 24, `RecordFormCounts` 8; all import paths or identifiers, none in prose (RAN: the lint's function over each file).

[2026-11-11T17:45Z] 48fe9ba31bce
COUNTS to FORM and training (copy WIRED), TRN-115, Wed 11 Nov 9:45 AM PST. JOB 1: `form_complies : Complies form` COMPILES at 7f39106d60, on the three axioms only.

RAN: run r20261111-173917-d53c on vy-nebius-cpu-3, `verity lean try security/Proofs/Properties/Nci/RecordComplies.lean --siblings --on auto --as training`, from a detached checkout of 7f39106d60, no edit. Its lines, copied:

- "Build completed successfully (10857 jobs)." Built in it, among others: `RecordBlockFresh`, `RecordFresh`, `RecordServed`, `RecordFormCounts`.
- "'NciCheck.Properties.Record.SecurityProofs.form_complies' depends on axioms: [propext, Classical.choice, Quot.sound]"
- "lean-try: security/Proofs/Properties/Nci/RecordComplies.lean: lean exit 0 in 277 s"
- "axiom probe: none of its 1 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"
- "siblings: no built module of its package that doesn't import Proofs.Properties.Nci.RecordComplies declares a name it does (640 imported in 9 s, peak 4598 MB); 3281 unchecked, with no build of their source as it is or importing a module with none, in their package or one it requires: Proofs.Accounting, Proofs.Accounting.Actual, ... and 3271 more"
- "verity lean try: run r20261111-173917-d53c on vy-nebius-cpu-3: exit 0 in 321 s wall"

THE SIBLINGS CHECK IS PARTIAL: 640 modules read, 3281 not, since that slot holds no build of them at this tree. It says nothing of a clash with a module among the 3281. A full one needs a slot with the package's records at this head (your r20261111-171851-c95c or WIDTH's r20261111-172003-3eb4, once ended).

ALSO, a try that did not run: r20261111-172307-daca (`RecordFormCounts.lean`, your 73dbb102d3 merged with my 62e842b48d, vy-nebius-cpu-1): exit 2, "lean-try: security/Proofs: no dependencies to take on this node, and the sandbox can't fetch them: nothing ran; run `lean_changed.py --records security/Proofs` here once". It says nothing of the file. The try above built that file at your newer head.

Log kept: trn115-try-complies-siblings-r20261111-173917-d53c.log.

Next, mine: job 2 (the retired-terms counts of my two files), then job 3 (the read of the five statements against the note).

