import FlockSoundness

/-!
# Axiom audit

Run `lake env lean FlockSoundness/Check.lean`.

The proved results must depend only on Lean's three standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). Named assumptions (`Assumptions.BCHKS25Thm46`) are hypotheses
of the theorems that use them, never `axiom` declarations, so they cannot appear here.

`rep_sound` and `table_sound` still show `sorryAx`, from `ligerito_phase` alone; that disappears
when it is proved.
-/

-- the interaction model and the repetition theorem
#print axioms FlockSoundness.Game.prob_le_value
#print axioms FlockSoundness.Game.value_bind_le
#print axioms FlockSoundness.Game.prCoin_exists_le
#print axioms FlockSoundness.Game.value_interleave_le
#print axioms FlockSoundness.Game.prCoin_eval_eq_eval_le
#print axioms FlockSoundness.Game.prCoin_mvEval_eq_zero_le
-- one sumcheck round, in each of the spec's three encodings
#print axioms FlockSoundness.Rounds.residualRound_sound
#print axioms FlockSoundness.Rounds.sumRound_sound
#print axioms FlockSoundness.Rounds.lincheckRound_sound
-- the numbers
#print axioms FlockSoundness.Accounting.tableError_le_of_le_33
#print axioms FlockSoundness.Accounting.tableError_le_of_34_35
-- the compiled layer: Merkle binding, the rewinding lemma, the compilation step
#print axioms FlockSoundness.Merkle.opening_binding
#print axioms FlockSoundness.Rewinding.bad_sq_le
#print axioms FlockSoundness.Rewinding.prBad_le
#print axioms FlockSoundness.Compiled.compile_step
#print axioms FlockSoundness.Compiled.compile_step_prefix
-- the list size (ArkLib's Johnson bound), the zerocheck, lincheck and opening phases
#print axioms FlockSoundness.closeMsgs_card_le
#print axioms FlockSoundness.zerocheck_phase_of_card
#print axioms FlockSoundness.zerocheck_phase
#print axioms FlockSoundness.lincheck_phase_of_card
#print axioms FlockSoundness.lincheck_phase
#print axioms FlockSoundness.opening_phase_of_card
#print axioms FlockSoundness.opening_phase
-- Ligerito's generic pieces: the list size at every level, K coins, rounds over K, α batching
#print axioms FlockSoundness.card_close_le
#print axioms FlockSoundness.value_drawK_le
#print axioms FlockSoundness.prCoin_fromMsg_le
#print axioms FlockSoundness.prCoin_eqList_le
-- the link points
#print axioms FlockSoundness.link_sound_of_card
#print axioms FlockSoundness.link_sound
-- the theorem: `rep_sound` is proved from four phase lemmas and `table_sound` from `rep_sound` and
-- `link_sound`; the Ligerito phase lemma is still `sorry`
#print axioms FlockSoundness.rep_sound
#print axioms FlockSoundness.table_sound
#print axioms FlockSoundness.table_sound_fast100
