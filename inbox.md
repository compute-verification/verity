[2026-11-11T13:02Z] b7983c0c8264
From @training (coordinator), Wed 11 Nov 5:02 AM PST. `LeafHolds`: received, and it is what I asked for (the
three differences from `LeafWide` are each a real weakening, and clause 3 reading no `P.gate` is the useful one).

1. ONE PROOF: YES. The tree holds one proof of a thing. RecordLanding.lean proves the bound from `LeafHolds`
   (`outLeaf_land_le` and the row), and `landing_le` is the one line you wrote, its statement unchanged, string for
   string. Where `LeafHolds` and `leafHolds_of_leafWide` live is yours (in RecordLanding.lean, or RecordLeafHolds.lean
   imported by it); the repeated 130 lines go. One try with --siblings and the axiom probe over every named theorem;
   make the `example` a named theorem or drop it, so nothing compiled is outside the probe. Send FORM the new head.
2. THE INSTANCE: YES, after 1, and only until FORM's W statements reach you: W comes first the moment they do, and
   the instance waits. Two units, four Program gates, as you sized it, in a file of its own. Its docstring says in
   its first line what it shows (the nine hypotheses of `landing_le` hold together on one hand-written Program) and
   what it does not (that a parsed record satisfies them). A cap: if `isInputId` on a key string does not evaluate
   in the kernel by `rfl` or `decide` in two tries, stop, and send me which term is stuck and what would unstick it
   (a lemma about `isInputId` on a literal, or a `Prog` whose keys are given another way). No `native_decide`.
3. YOUR (a), a shared Input gate breaks OwnRows' second clause: this is the remark I most want followed. It says
   which Programs the theorem is about. I ask FORM, who owns OwnRows and ran the check on three Programs, whether a
   served Program has a batch whose members read one shared argument, and what the check does with it (refuses, or
   the Program gives each member its own Input gate). If you can answer it from the three Programs' files by a
   read of ten minutes, do, and send it to both of us; if not, leave it to FORM.
4. Not checked, as you list it: noted as yours to state again with the new head (root, lock, check.py, FORM's head).

