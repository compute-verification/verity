import FlockSoundness.Discharge.ZkHidden.Statements

/-!
# The theorems of `Statements.lean`
-/

namespace FlockSoundness.Discharge.ZkHidden

theorem acceptsZK_setupOne : AcceptsZKSetupOne := by
  sorry

theorem acceptsZK_oneAny : AcceptsZKOneAny := by
  sorry

theorem acceptsZK_retainedAny : AcceptsZKRetainedAny := by
  sorry

theorem acceptsZK_acceptedH : AcceptsZKAcceptedH := by
  sorry

theorem zk_schedOK : ZkSchedOK := by
  sorry

theorem tablesAtZKH_live {n mPts : ℕ} {Reg : Type} : TablesAtZKHLive (n := n) (mPts := mPts) (Reg := Reg) := by
  sorry

end FlockSoundness.Discharge.ZkHidden
