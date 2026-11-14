module

public import Proofs.Flock.SpotFlock.Accepting
public import Proofs.Flock.SpotFlock.RunsAs.SecondStageOf
public import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP

/-!
# A statement in a template's scope sets up one table

Flock's masked verifier runs a session at one table only where the session's statement sets up one table. The held
copies' two-stage design decides it of every statement in a template's scope:

* `oneTable_of_shape`: a statement in Flock's masked scope (`SpotFlock.zkOk`) whose list of table sizes
  (`SpotFlock.shapeOf`) has one entry sets up one table.
* `oneTable_of_sessionOk`: where the design evaluates, any statement in template `t`'s scope sets up one table, at a
  table count of at most one. The design evaluates only where each template's reference statement sets up one
  (`SpotFlock.ref_one_table`), a statement in scope has its reference statement's table sizes (`SpotFlock.sessionOk`'s
  shape clause), and the masked verifier sets a count above one up as that many tables
  (`HiddenExec.setupTables_okHP`).

This file imports SPOT[Flock]'s analysis alone and nothing of the sessions' layout (`SessionCoins`), so a file of
SPOT's analysis, which imports none of the network certifier, can read it. What the layout makes of one table is
`OneTable`'s. Its two theorems are in the namespace `Verity.SampledTwo`, where the layout's files read them.
-/

@[expose] public section

namespace Verity.SampledTwo

/-- **A statement in Flock's masked scope whose list of table sizes has one entry sets up one table.** -/
theorem oneTable_of_shape {x : Claims.Msg} (hz : SpotFlock.zkOk x = true) (h1 : (SpotFlock.shapeOf x).length = 1) :
    ∃ s spec st, FlockDeploy.setupOf true x = some (s, spec, #[st]) := by
  obtain ⟨s, spec, sts, hset, -⟩ := SpotFlock.zkOk_sound hz
  have hsz : sts.size = 1 := by simpa [SpotFlock.shapeOf, hset] using h1
  obtain ⟨st, rfl⟩ := Array.size_eq_one_iff.1 hsz
  exact ⟨s, spec, st, hset⟩

/-- **A statement in a template's scope sets up one table, at a table count of at most one**: where the held copies'
two-stage design evaluates, a statement that passes template `t`'s scope check has the one table size of the
template's reference statement, and the masked verifier sets a table count above one up as that many tables. -/
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
