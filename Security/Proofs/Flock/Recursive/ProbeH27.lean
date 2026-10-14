import Proofs.Flock.Recursive.HPortTie

/-!
# Temporary axiom probe (H3, Accepted outcome tie) — drop once the Lean check passes

`#print axioms` on every declaration this change adds: a declaration a `variable` block silently dropped would
print `unknown constant` and fail the build here.
-/

#print axioms FlockSoundness.Game.Outcome.of_bind
#print axioms FlockSoundness.Game.Outcome.of_map
#print axioms FlockSoundness.Game.prob_mono_outcome
#print axioms FlockSoundness.Discharge.Recursion.outcome_play
#print axioms FlockSoundness.Discharge.Recursion.outcome_committed
#print axioms FlockSoundness.Discharge.Recursion.outcome_seqC
#print axioms FlockSoundness.Discharge.Recursion.outcome_auditLive_liveOfZJ
#print axioms FlockSoundness.Discharge.Recursion.ZkOuter.outcome_writes
#print axioms FlockSoundness.Discharge.Recursion.outcome_sessions
#print axioms Flock.Guarantees.checksC3_of_recordDraw
#print axioms FlockSoundness.Discharge.Recursion.ChecksC3Reached
#print axioms FlockSoundness.Discharge.Recursion.reachedChecked_of
