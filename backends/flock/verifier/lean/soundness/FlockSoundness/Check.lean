import FlockSoundness

/-!
# Axiom audit

Run `lake env lean FlockSoundness/Check.lean`.

The proved results must depend only on Lean's three standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). Named assumptions (`Assumptions.BCHKS25Thm46`) are hypotheses
of the theorems that use them, never `axiom` declarations, so they cannot appear here.

`table_sound` still shows `sorryAx`, from the two lemmas it is proved from (`rep_sound`,
`link_sound`); that disappears when they are proved.
-/

-- the interaction model and the repetition theorem
#print axioms FlockSoundness.Game.prob_le_value
#print axioms FlockSoundness.Game.value_bind_le
#print axioms FlockSoundness.Game.prCoin_exists_le
#print axioms FlockSoundness.Game.value_interleave_le
#print axioms FlockSoundness.Game.prCoin_eval_eq_eval_le
#print axioms FlockSoundness.Game.prCoin_mvEval_eq_zero_le
-- the numbers
#print axioms FlockSoundness.Accounting.tableError_le_of_le_33
#print axioms FlockSoundness.Accounting.tableError_le_of_34_35
-- the theorem (proved from `rep_sound` and `link_sound`, both still `sorry`)
#print axioms FlockSoundness.table_sound
#print axioms FlockSoundness.table_sound_fast100
