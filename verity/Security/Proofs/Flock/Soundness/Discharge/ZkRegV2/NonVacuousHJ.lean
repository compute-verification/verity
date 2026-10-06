import Proofs.Flock.Soundness.Discharge.ZkReg.NonVacuousHJ

/-!
# `zk_session_soundHJR`'s draw hypotheses hold together

`ZkReg/NonVacuousHJ.lean` at a `ZkRegV2.scopeOkHV2` circuit, the circuits the statements take: `drawHypsZHJ` reads of
the circuit only that its unit returns the outputs, which `scopeOkHV2` gives as `scopeOk` does, so `drawHypsZHJV2`
builds `zk_session_soundHJR`'s and `zk_session_soundHJR_custody`'s `y₀` and draw hypotheses together at one accepted
session. With
`ZkRegV2.registered_v2_nonvacuousH` (the circuit and the registered check) this is what the red team's B1 asks of the
`J`-table statements: their circuit and draw premises are met by a real session.
-/

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n : ℕ}

/-- **The hidden-output `J`-table draw hypotheses hold together at an accepted session over `sha512/row/v2` rows**:
`drawHypsZHJ` at a `scopeOkHV2` circuit. -/
theorem drawHypsZHJV2 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : ZkRegV2.scopeOkHV2 c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0) (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    (∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S') ∧
      ∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  ⟨drawOkZKHJ_of_file I ht hc kd n hdiv dj hdj,
    setupZHJ_of_acceptsZK I hTags.1 ht hc (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 dj h hd
      (drawOkZKHJ_of_file I ht hc kd n hdiv dj hdj S)⟩

/-- **`zk_session_soundHJR`'s `y₀` and `hDraw` hold together**: `zk_session_soundHJR_nonvacuous` at a `scopeOkHV2`
circuit. The proof applies `zk_session_soundHJR` to this `y₀` and `hDraw`, at every registered-port list `own`. Not
covered: the session's existence, `hRec`, `hpub`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJR_nonvacuousV2 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : ZkRegV2.scopeOkHV2 c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJV2 I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_soundHJR (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- **`zk_session_soundHJR_custody`'s `y₀` and the custody's draw file hold together**:
`zk_session_soundHJR_custody_nonvacuous` at a `scopeOkHV2` circuit. The proof applies `zk_session_soundHJR_custody` to
this `y₀`, at every `own`. Not covered: the session's existence, `hCust.1` (`RecordsLiveZKJ`), `hpub`, `hkd`,
`hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundHJR_custody_nonvacuousV2 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : ZkRegV2.scopeOkHV2 c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : DrawFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (DrawSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', DrawOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := drawHypsZHJV2 I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) =>
    zk_session_soundHJR_custody (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy, hok⟩

end FlockSoundness.Discharge.ZkReg
