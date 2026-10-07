import Proofs.Flock.Soundness.Discharge.ZkHidden.Theorems
import Proofs.Flock.Soundness.Discharge.ScopeW.Sites

/-!
# A one-table hidden-output `--zk` acceptance is placed, under the widened scope

`AcceptsZKAcceptedHW`: `ZkHidden.AcceptsZKAcceptedH` with `∀ p ∈ c.ports.toList, rowShapeW p = true` in place of
`∀ p ∈ c.ports.toList, p.v2 = false`, its conclusion the same. `acceptsZK_acceptedHW` proves it as
`acceptsZK_acceptedH`, with the unit slots' rows from `unitRowsFlatW`.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec ZkHidden

/-- **A one-table hidden-output `--zk` acceptance is placed, at the widened scope**: `AcceptsZKAcceptedH` at circuits
whose every row has one of `rowShapeW`'s shapes. -/
def AcceptsZKAcceptedHW : Prop :=
  ∀ (I : Inputs) (t : Exec.Transcript), I.tags.hiddenOutputs = true → I.tags.typed = false → ∀ {c : Flock.Circuit},
    HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c → (∀ p ∈ c.ports.toList, rowShapeW p = true) →
    I.count ≤ 1 → AcceptsZK I t →
    ∃ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (recordDraw t.record)
        I.partition I.program = .ok st ∧ Flock.Zk.shapeOk st = .ok () ∧
      ∀ regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)),
        ∃ (T : Model.Statement) (a : AcceptedH T), a.st = st ∧ HEq a.regs regs ∧ a.tags = zkTags I ∧
          a.cf = I.circuitFile ∧ a.pf = I.publicFile ∧ a.coins = zkCoins I ∧ a.tables = I.tables ∧
          a.draw = recordDraw t.record

theorem acceptsZK_acceptedHW : AcceptsZKAcceptedHW := by
  intro I t hh ht c hc hW hc1 h
  obtain ⟨st, hst, hshape, -⟩ := acceptsZK_oneAny I t hc1 h
  have hh' : (zkTags I).hiddenOutputs = true := (zkTags_hiddenOutputs I).trans hh
  rw [setupOne_hidden hh'] at hst
  have ht' : (zkTags I).typed = false := (zkTags_typed I).trans ht
  have hcst : st.c = c := Except.ok.inj ((setupHidden_parseC hst).symm.trans hc)
  refine ⟨st, hst, hshape, fun regs => ⟨_, ⟨zkTags I, I.circuitFile, I.publicFile, zkCoins I, I.tables,
    recordDraw t.record, I.partition, I.program, st, ht', hst, Hidden.hidden_layout ht' hst,
    unitRowsFlatW ht' hst (by rw [hcst]; exact hW), regs, rfl⟩, rfl, HEq.rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩⟩

end FlockSoundness.Discharge.ScopeW
