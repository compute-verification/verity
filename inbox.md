[2026-11-11T12:21Z] fb6077f11086
WIRED, TRN-116: done early. laysOut_exists is proved as stated. I am free for more of the property.

Branch and head: cursor/training-record-wired-05f2 at cba2125f9ad0061885fff178707bbbe7caa48dc1, one commit on 9d60cc0fb6, pushed. One file: security/Proofs/Properties/Nci/RecordWired.lean (263 lines). Nothing else touched.

The theorem: NciCheck.Properties.Record.SecurityProofs.laysOut_exists, proved as stated. I ran a string compare of your brief's code block against the file from `theorem` to `:= by`: equal. All six hypotheses are used. No sorry, axiom or native_decide.

The node try that passed (ran): r20261111-121127-8dc9 on vy-nebius-cpu-3, rc 0, `lean_changed.py --try security/Proofs/Properties/Nci/RecordWired.lean --siblings` (what `verity lean try FILE --siblings` runs). Read from its job.json and stdout.log: source commit cba2125f9a, not dirty; lean exit 0 in 22 s; no warning on the file. Axioms printed: `laysOut_exists depends on axioms: [propext, Classical.choice, Quot.sound]`; probe: none of its 10 theorems goes beyond those three. Siblings: no built module declares a name it does (854 compared; 2984 not built on that node and not compared).

RecordTakes.lean compiles as its own module: the same run built it (`Built Proofs.Properties.Nci.RecordTakes`).

What I did not check: check.py's kernel replay; the whole check; whether the five hypotheses hold of the record check's accepted form (FORM's); nobody else has read the proof. The file is in no lakefile root and no lock.

Two things FORM needs when merging: (1) my header differs from the brief's. I dropped the unused RecordGrid import and put the opens inside the namespace, so the file has 3 retired-terms sites (import path, namespace, end), not 7. Those 3 need an entry in tests/lint/allowlists/repo_retired_terms.json, which I did not add because I edit no other file. (2) Every helper is private; the file adds one name to the shared namespace.

What I think of the statement and definitions: nothing is wrong, but it says less than it may read. LaysOut fixes only D's gate count, widths, input flags and reads. It says nothing of D.op or D.outputs, and my witness has constant-0 operations and no output gate. If LaidOut needs D's trace or outputs to mean anything, this theorem does not give it. Also D is some circuit, not a function of P: obtain it once. And an operand wired to a writer with no output leaf meeting it lands nowhere, so the gate reads nothing there; that is for Served's Filled clause to exclude.

Hand-off note: lanes/training/20261111T1221Z-handoff-trn116-laysout-exists-proved.md.

