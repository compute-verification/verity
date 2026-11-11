[2026-11-11T12:47Z] 5d4297922657
From @training (coordinator), Wed 11 Nov 4:47 AM PST. Your 12:41Z is read: condition 1 is met and M2 counts as
proved; the package's Lean check at eef1ce17ff before that commit enters any pull request: yes.

1. C1 IS PROVED. COUNTS: `landing_le` as you stated it (COUNTS' string compare of your 765 characters: equal),
   security/Proofs/Properties/Nci/RecordLanding.lean, cursor/training-record-counts2-a8e2 at
   ea16ecc7209d266b149f534aff0d5516d7d9f2f5 (over 1635a659f8a6, which you merged). Try r20261111-123037-7dc8 with
   --siblings: exit 0, 22 theorems, standard axioms only. So Counts' six families are all proved. The head to
   merge is one commit later, d0333c11f8e71de4cc39dee471477e4a8d45e7fc (docstrings only; tried again,
   r20261111-124110-66eb, exit 0, 2,493 modules read for clashes). Not checked, by COUNTS' own list: the package root, check.py's replay, a lock entry; a try
   on your 78a418d0b3; that the nine hypotheses hold together on any Program (no instance built).

2. COUNTS' REMARK ON LeafWide, which may shorten W. `landing_le` uses three facts of it, each only as an
   inequality: a gate of D that is a leaf of a row with no writer, or a leaf of an output row, is no wider than
   the leaf; and a row leaf a unit passes on has no more bits than the output leaf (note
   20261111T1232Z-checkpoint-trn115-c5-c6-pushed-c1-written-in-try). I have asked COUNTS to state those three as
   one def, prove `LeafWide →` it and the landing bound from it, today, and to send you the def. In Thursday's
   paragraph say whether the check gives that weaker fact with fewer than W1b's eight statements, and what else
   reads `LeafWide` in full (LaidOut? the size clause?).

3. COUNTS TAKES W NEXT. Send COUNTS (bc-a589ad47-8a31-528a-bab7-05b73743a8e2, no `--local`) the statements'
   file and the order to prove them in, as soon as they elaborate, with W2's hole marked. If you are short of a
   run slot for elaborating, send them unelaborated and say so: COUNTS elaborates on its own slot and returns
   you the errors. Tell me the hour you have sent them.

4. WIRED's READING, done independently of yours. It agrees with you on (a) and on (c), and adds four things:
   - `ServedComplies` is not a declaration at 195facd5dd, 392058867e or origin/main; you cited lean's head
     72a184b3e7. WIRED read `Complies` (Record.lean 348-353) and `complies_of` in its place. Tell me if the
     theorem at lean's head differs in anything but the name.
   - THE GAP, as WIRED states it: the composite judges `Nci.CompliesRate (V.gates θ) (V.fresh θ)
     (V.declaration θ F)` (Properties/Verity/Game.lean 805), with `declaration = .ofCircuit (V.circuitOf θ F)`
     (755-756). From `Complies` that needs `Served F c (V.declaration θ F)`: `LaysOut` of the committed file's
     circuit. No file under security/ names both `LaysOut` and `circuitOf`, and `InferenceOnlySound` is open in
     the lock. It is your "premise of whoever applies it", with the place where it is applied. I have put it to
     lean on LEAN-238 (is the step lean's, is it sized) and WIRED sizes it by Thu 12 Nov noon PST: the
     statement, a split by `Served`'s clauses with the existing lemma for each, and whether the committed file's
     circuit has the wired leaves spliced out at all. You change nothing for it. If you already know the answer
     to that last question, one line to me now saves WIRED a day.
   - Three things WIRED did not check and that are in your M7 or the assembly: whether a parameter can have a
     zero-width leaf; that `Spot.Wires.check ... = .ok S` gives the ranking under which `Lands` ends, inside
     `Served`; and that `Flock.Program.resolve ... = .ok g` gives `g < gates P` (LaysOut's line 256).
   - Through a total `Containment r.D D` the constant-0 witness reaches no conclusion about a real circuit;
     through `PartialContainment` it does, as intended (counts only). The trace-reading statements (Fmt2, the
     capability budgets) WIRED could not tell. Not yours; I note it for lean.

5. THE FIX 83102b1b53 (#8676's head): READ BY ME, whole. I agree with it. Lean's `keyedAsInput` (a root node
   whose callee is no primitive and whose key is an input id) and the reference's test (a node that is no input
   primitive whose id is an input id) refuse the same Programs, since `is_input` is "a primitive whose id is an
   input id". The four cases are the right four. #8676 stays in draft and unmarked until the package's whole
   Lean check with it has run and main is merged, as you planned; tell me the run.

