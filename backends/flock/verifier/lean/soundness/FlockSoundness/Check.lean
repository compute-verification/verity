import FlockSoundness

/-!
# Axiom audit

Every proved result below depends only on Lean's three standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). A `sorryAx` or any other axiom in this output is a failure.
Named assumptions (BCHKS25 Theorem 4.6, …) are hypotheses of the theorems that use them, never
`axiom` declarations, so they cannot appear here.
-/

#print axioms FlockSoundness.Game.prob_le_value
#print axioms FlockSoundness.Game.value_bind_le
#print axioms FlockSoundness.Game.prCoin_exists_le
#print axioms FlockSoundness.Game.value_interleave_le
#print axioms FlockSoundness.Accounting.tableError_le_of_le_33
#print axioms FlockSoundness.Accounting.tableError_le_of_34_35
