import FlockSoundness.Discharge.ZkHidden.Sched
import FlockSoundness.Discharge.ZkExec.Setup
import FlockSoundness.Discharge.HiddenExec.Accepts
import FlockSoundness.Discharge.HiddenExec.Flat
import FlockSoundness.Discharge.HiddenExec.Accepted
import FlockSoundness.Protocol.Discharge.ZkHidden.Theorems

/-!
# The theorems of `Statements.lean`
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock Refine Exec ZkExec HiddenExec

theorem acceptsZK_setupOne : AcceptsZKSetupOne := by
  intro I t h
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨hpos, hall⟩ := setupTables_setupOne hs
  exact ⟨spec, sts, hs, hv, hpos, fun j hj => ⟨array_forM_ok hshape j hj, hall j hj⟩⟩

theorem acceptsZK_retainedAny : AcceptsZKRetainedAny := by
  intro I t hr h
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  obtain ⟨-, sess, -, hd, -⟩ := zk_verify_ok _ _ _ _ hv
  have hsr : spec.retainRounds = true := ((setupTables_retainAny hs).trans (zkTags_retainRounds I)).trans hr
  exact ⟨spec, sts, sess, hs, hd, decode_allRetained hsr hd⟩

theorem acceptsZK_acceptedH : AcceptsZKAcceptedH := by
  intro I t hh ht c hc hv1 hc1 h
  obtain ⟨st, hst, hshape, -⟩ := acceptsZK_oneAny I t hc1 h
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  rw [setupOne_hidden hh'] at hst
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have hcst : st.c = c := Except.ok.inj ((setupHidden_parseC hst).symm.trans hc)
  refine ⟨st, hst, hshape, fun regs => ⟨_, ⟨zkTags I, I.circuitFile, I.publicFile, zkCoins I, I.tables,
    recordDraw t.record, I.partition, I.program, st, ht', hst, Hidden.hidden_layout ht' hst,
    unitRowsFlat ht' hst (by rw [hcst]; exact hv1), regs, rfl⟩, rfl, HEq.rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩⟩

end FlockSoundness.Discharge.ZkHidden
