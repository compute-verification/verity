module

public import Proofs.Flock.SpotFlock.Accepting
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP

/-! A statement in a template's scope sets up one table; the file imports SPOT[Flock]'s analysis alone, so SPOT's
analysis, which imports none of the network attester, can read it. -/

@[expose] public section

namespace Verity.SampledTwo

theorem oneTable_of_shape {x : Claims.Msg} (hz : SpotFlock.zkOk x = true) (h1 : (SpotFlock.shapeOf x).length = 1) :
    ∃ s spec st, FlockDeploy.setupOf true x = some (s, spec, #[st]) := by
  obtain ⟨s, spec, sts, hset, -⟩ := SpotFlock.zkOk_sound hz
  have hsz : sts.size = 1 := by simpa [SpotFlock.shapeOf, hset] using h1
  obtain ⟨st, rfl⟩ := Array.size_eq_one_iff.1 hsz
  exact ⟨s, spec, st, hset⟩

theorem oneTable_of_sessionOk {h : SpotFlock.Held} {s d} (hd : SpotFlock.two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) {xS : Claims.Msg} (hs : SpotFlock.sessionOk h t xS = true) {ins : FlockDeploy.Ins}
    {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hset : FlockDeploy.setupOf true xS = some (ins, spec, sts)) : sts.size = 1 ∧ ins.read.I.count ≤ 1 := by
  have h1 : (SpotFlock.shapeOf (h.xS₀ t)).length = 1 := (SpotFlock.ref_one_table h hd ht).1
  simp only [SpotFlock.sessionOk, Bool.and_eq_true] at hs
  have e : SpotFlock.shapeOf xS = SpotFlock.shapeOf (h.xS₀ t) := eq_of_beq hs.1.1.2
  rw [← e] at h1
  have hsz : sts.size = 1 := by simpa [SpotFlock.shapeOf, hset] using h1
  refine ⟨hsz, ?_⟩
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hread : FlockDeploy.readOf true xS = some ins.read := by simp [FlockDeploy.readOf, hins]
  have hh : ins.read.tags.hiddenOutputs = true := by simpa [SpotFlock.hiddenOf, hread] using hs.2
  have hrs := FlockDeploy.Ins.setup_read hsu
  unfold Flock.Readable.Read.setup at hrs
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okHP hh hrs with ⟨hc, -⟩ | ⟨hc, hn, -⟩
  · exact hc
  · omega

end Verity.SampledTwo
