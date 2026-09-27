import FlockSoundness

/-!
# Axiom audit

Run `lake env lean FlockSoundness/Check.lean`.

The proved results must depend only on Lean's three standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). Named assumptions (`Assumptions.BCHKS25Thm46`) are hypotheses
of the theorems that use them, never `axiom` declarations, so they cannot appear here.
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
-- the algebra of the final check's weights, and a level's fold identity
#print axioms FlockSoundness.sum_eqList_prod
#print axioms FlockSoundness.sum_eqList_split
#print axioms FlockSoundness.sum_eqList_Xhat
#print axioms FlockSoundness.foldLoL_preW
#print axioms FlockSoundness.foldLoL_W0
-- decoding a fold near a codeword
#print axioms FlockSoundness.decode
-- a level's folds, and from one level to the next
#print axioms FlockSoundness.phaseB
#print axioms FlockSoundness.phaseB0
#print axioms FlockSoundness.value_twoOod_good
#print axioms FlockSoundness.value_queryAndBatch_le
#print axioms FlockSoundness.phaseA
-- the last level, every level, Ligerito
#print axioms FlockSoundness.finalStage
#print axioms FlockSoundness.value_recursiveRounds_le
#print axioms FlockSoundness.ligerito_sound
#print axioms FlockSoundness.ligerito_phase
-- the link points
#print axioms FlockSoundness.link_sound_of_card
#print axioms FlockSoundness.link_sound
-- the theorem: `rep_sound` is proved from four phase lemmas and `table_sound` from `rep_sound` and
-- `link_sound`
#print axioms FlockSoundness.rep_sound
#print axioms FlockSoundness.table_value_sound
#print axioms FlockSoundness.table_sound
#print axioms FlockSoundness.table_sound_fast100
#print axioms FlockSoundness.table_sound_fast100_34_35
-- the executable's arithmetic: `Arith.Correct` from lane flock-verifier's level-3 theorems, so the
-- theorem for the executable has BCHKS25 Thm 4.6 as its only hypothesis
#print axioms FlockSoundness.execArith_correct
#print axioms FlockSoundness.table_sound_exec
#print axioms FlockSoundness.table_sound_exec_fast100
#print axioms FlockSoundness.table_sound_exec_fast100_34_35
-- the compiled theorem (§1.2): the lockstep coupling with the plurality prover, rewinding at every
-- cap, and binding: every counted conflict and every self-clash is a SHA-512 collision
#print axioms FlockSoundness.Rewinding.expect_bad_le
#print axioms FlockSoundness.Game.Lock.expect_le_coupled
#print axioms FlockSoundness.Game.Lock.leafE_off_le
#print axioms FlockSoundness.lockTable
#print axioms FlockSoundness.bad_split
#print axioms FlockSoundness.compiled_le_bad
#print axioms FlockSoundness.bad_le
#print axioms FlockSoundness.table_sound_compiled
#print axioms FlockSoundness.SelfClash.collision
#print axioms FlockSoundness.op₀_conflict_collision
#print axioms FlockSoundness.opR_conflict_collision
-- the audit layer: strategy-level probabilities, draw laws, partitions, the one-stage profile
#print axioms FlockSoundness.Game.prob_or_le
#print axioms FlockSoundness.Game.prob_map
#print axioms FlockSoundness.Audit.Law.escape_le_of_mem
#print axioms FlockSoundness.Audit.Law.subset_escape
#print axioms FlockSoundness.Audit.Law.subset_miss
#print axioms FlockSoundness.Audit.Law.avg_escape_subsets
#print axioms FlockSoundness.Audit.Law.choose_support
#print axioms FlockSoundness.Audit.Law.subset_minimax
#print axioms FlockSoundness.Audit.Law.subset_escape_le_prod
#print axioms FlockSoundness.Audit.Law.bernoulli_avg_prod
#print axioms FlockSoundness.Audit.Law.bernoulli_escape
#print axioms FlockSoundness.Audit.Law.demo_count_590
#print axioms FlockSoundness.Audit.Partition.compose
#print axioms FlockSoundness.Audit.Partition.compose_cone
#print axioms FlockSoundness.Audit.Partition.correct_congr
#print axioms FlockSoundness.Audit.Partition.Refines.exists_wrong_fine
#print axioms FlockSoundness.Audit.Partition.Refines.card_wrong_le
#print axioms FlockSoundness.Audit.audit_le
#print axioms FlockSoundness.Audit.audit_profile
#print axioms FlockSoundness.Audit.audit_count
#print axioms FlockSoundness.Audit.audit_drawn
#print axioms FlockSoundness.Game.prob_bind_le
#print axioms FlockSoundness.Audit.audits_seq
#print axioms FlockSoundness.Audit.Partition.audit_cone
#print axioms FlockSoundness.Audit.Partition.audit_cone_count
#print axioms FlockSoundness.Audit.Partition.audit_covered
#print axioms FlockSoundness.Audit.Partition.oracle_count
#print axioms FlockSoundness.Audit.Partition.oracle_profile
#print axioms FlockSoundness.Audit.Partition.oracle_drawn
#print axioms FlockSoundness.Audit.Partition.oracle_cone
-- two-stage audits and their relation to one-stage
#print axioms FlockSoundness.Game.prCoin_pi_forall
#print axioms FlockSoundness.Audit.twoStage_profile
#print axioms FlockSoundness.Audit.twoStage_count
#print axioms FlockSoundness.Audit.effEscape_full
#print axioms FlockSoundness.Audit.twoStage_full_profile
#print axioms FlockSoundness.Audit.effEscape_bernoulli_prod
#print axioms FlockSoundness.Audit.Partition.two_stage_a_coarse
#print axioms FlockSoundness.Audit.Partition.Refines.analysis₂
-- tightness and insecurity on NOT chains
#print axioms FlockSoundness.Audit.NotChains.effEscape_chains
#print axioms FlockSoundness.Audit.NotChains.prob_σT
#print axioms FlockSoundness.Audit.NotChains.two_stage_b_tight
#print axioms FlockSoundness.Audit.NotChains.naive_two_level_unsound
#print axioms FlockSoundness.Audit.NotChains.late_interior_insecure
#print axioms FlockSoundness.Audit.NotChains.effEscape_bernoulli
-- the Flock instantiation at the oracle layer
#print axioms FlockSoundness.Audit.value_batch_le
#print axioms FlockSoundness.Audit.FlockTable.value_le
#print axioms FlockSoundness.Audit.Partition.flockSession_le
#print axioms FlockSoundness.Audit.Partition.flock_session_sound
#print axioms FlockSoundness.Audit.Partition.flock_inputs_sound
#print axioms FlockSoundness.Audit.Partition.flock_oracle_count
#print axioms FlockSoundness.Audit.Partition.flock_oracle_count_subset
#print axioms FlockSoundness.Audit.Partition.flock_oracle_drawn
#print axioms FlockSoundness.Audit.Partition.flock_oracle_cone
