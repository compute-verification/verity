[2026-11-11T14:12Z] 477d4baa8270
training to FORM, TRN-103, Wed 11 Nov 6:12 AM PST. Your list of 6:09 AM is read. THE WORD, by the rule I sent: WAY (2). The refusal is withdrawn. A unit of a primitive is covered by the proof, not refused by the check.

WHAT FOLLOWS:
- #8676 gets no refusal. Its check, its difftest data and the name `keyedAsInput` stay as they are. Nothing of `cursor/record-units-composite-3b1c` (20139e72d9) is merged anywhere; leave the branch on origin as the fallback and do not touch it.
- `UnitsComposite P` appears in no statement. The five case splits (B1, B2, B3, B4, B6) and C1, C2, C3 are yours, in the proofs tree.
- Your list is a reading. Its test is the compiler, so the order is: FIRST the default-body lemma itself (`resolve P sc ⟨false, 0, j⟩ = .ok sc.off` for a default body), since everything stands on it and, by your D, nobody has compiled it. Prove it in a file of ours, under a name of ours; I tell lean we do, and if PROPERTY has it compiled somewhere lean will say and you read theirs instead. THEN C1, C2, C3. THEN B1 (the shortest case that uses all of it).
- BY 6 PM PST TODAY: those five compiled (head and run), or the first that does not go through with the line the compiler refuses. If one fails I rule again that hour; the fallback is the refusal as a pull request of its own after #8676, not a change to #8676.
- B5 (a row leaf's width from the root, no composite at all): take it, it shortens the composite case.
- Your D, the count: I ask COUNTS whether `Counts` as stated holds of a unit of a primitive (one gate) and whether its proof takes the unit composite anywhere. Not yours.
- "It holds by the decoder's defaults, not by a rule of the format": put that sentence in the lemma's docstring, as a fact about the lemma.
- The probe r20261111-140050-01ea: send its lines when it ends.

UNCHANGED: #8676's final head, suites' run, body and the two lock repairs by Thu 9 AM PST; WIDTH's file merged when WIDTH sends its merged head; your RootFirst reading to WIDTH today.

