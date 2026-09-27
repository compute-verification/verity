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
#print axioms FlockSoundness.prCoin_positions_miss_le
-- mutual correlated agreement at the folds, from the named assumption
#print axioms FlockSoundness.prCoin_mca_le
#print axioms FlockSoundness.rowsAgreeOn_of_fold
#print axioms FlockSoundness.rowsAgreeOn_of_foldAll
#print axioms FlockSoundness.foldAllRows_eq_sum
#print axioms FlockSoundness.value_foldRounds_mca_le
#print axioms FlockSoundness.prCoin_level_mca_le
-- out-of-domain binding
#print axioms FlockSoundness.value_twoOod_collide_le
-- a level's folds against a list
#print axioms FlockSoundness.value_foldRounds_sum_le
#print axioms FlockSoundness.value_foldRounds_sum_invariant
-- the algebra of the final check's weights
#print axioms FlockSoundness.sum_eqList_prod
#print axioms FlockSoundness.sum_eqList_split
#print axioms FlockSoundness.sum_eqList_Xhat
-- the link points
#print axioms FlockSoundness.link_sound_of_card
#print axioms FlockSoundness.link_sound
-- the theorem: `rep_sound` is proved from four phase lemmas and `table_sound` from `rep_sound` and
-- `link_sound`; the Ligerito phase lemma is still `sorry`
#print axioms FlockSoundness.rep_sound
#print axioms FlockSoundness.table_sound
#print axioms FlockSoundness.table_sound_fast100
