import Proofs.Flock.Soundness.Discharge.ZkHidden.NonVacuousJ
import Proofs.Flock.Soundness.Discharge.ZkReg.SoundHJCustody

/-!
# The registered `J`-table statements' draw hypotheses hold together at a `Layout.scopeOk` circuit

`ZkHidden/NonVacuousJ.lean` for the registered-output statements: `zk_session_soundHJR` and
`zk_session_soundHJR_custody` take the hidden-output `y₀ : DrawSetupZHJ` at table 0's inputs and the draw hypotheses
of `zk_session_soundHJ` and its custody form, and `drawHypsZHJ` builds them together at one accepted session over a
circuit `Layout.scopeOk` admits. The statements themselves take `ZkRegV2.scopeOkHV2` circuits, since the registered
check refuses every registered port of a `Layout.scopeOk` circuit (`ZkRegV2.check_refuses_scopeOk`):
`ZkRegV2/NonVacuousHJ.lean` builds the same hypotheses there and applies the statements to them.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n : ℕ}

/-- **The `J`-table draw hypotheses `y₀` and `hDraw` hold together at a `Layout.scopeOk` circuit.** At the coin
server's draw file (`DrawFileZK`, `subset kd`, the tables dividing `kd`) and one session `verify --zk` accepts at the
draw `dj S`, on the inputs `ht`, `hTags`, `hc`, `hU` and `hscope` describe: at `mPts = mPtsOf c.g (kd / nTab I)` there
is a `y₀` (table 0's setup), every table is set up, and `DrawOkZKHJ` holds at every draw, so at every draw of the law
(`hDraw`). `zk_session_soundHJR_nonvacuousV2` is the same at the `scopeOkHV2` circuits `zk_session_soundHJR` takes.
Not covered: the session's existence, `hRec`, `hpub`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJR_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **The custody form's `y₀` and draw file hold together at a `Layout.scopeOk` circuit.** At the draw file `hCust.2`
is (`DrawFileZK`, `subset kd`), the tables dividing `kd` (`hdiv`), and a session `verify --zk` accepts at the draw
`dj S`, on the inputs `ht`, `hTags`, `hc`, `hU` and `hscope` describe: there is a `y₀` at `mPtsOf c.g (kd / nTab I)`
link points (table 0's setup), every table is set up there, and `DrawOkZKHJ`, which the statement derives from the draw
file, holds at every draw at the same count. `zk_session_soundHJR_custody_nonvacuousV2` is the same at the `scopeOkHV2`
circuits `zk_session_soundHJR_custody` takes. Not covered: the session's existence, `hCust.1` (`RecordsLiveZKJ`),
`hpub`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJR_custody_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  exact ⟨⟨y₀⟩, hy, hok⟩

end FlockSoundness.Discharge.ZkReg
