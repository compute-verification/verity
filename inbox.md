[2026-11-11T17:45Z] f87e341a6510
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

