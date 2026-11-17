module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.NonVacuousJ
public import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJJson
public import Proofs.Flock.Soundness.Discharge.ZkHidden.View
public import Proofs.Flock.Soundness.Discharge.ZkLink.SampleOk
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepted
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkHidden.Sched
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundJ

/-! The hidden-output `J`-table statements' sample hypotheses hold together at one accepted session. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n mPts : ℕ}

theorem setupZHJ_of_acceptsZKI (I : Inputs) (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : SampleOkZKHJ I dj mPts S) (j : Fin (nTab I)) :
    Nonempty (SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJH hh hd hs
  have hj : j.val < sts.size := by have := j.isLt; omega
  have hst := hstJ j.val hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, _, hz0, -⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[j.val].wideLeaf sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨_, hf⟩ := fast100_of_schedule hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  have hcJ : HmOut.parse (inputsJ I j.val).tags (inputsJ I j.val).circuitFile (inputsJ I j.val).tables = .ok c :=
    (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  obtain ⟨_, -, hdr⟩ := setupHidden_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec (c := { sts[j.val].c with outPorts := #[] }) hdr
  obtain ⟨x, hx⟩ := sampleSetupH_of_setupHiddenI (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hcJ hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, -, -, -⟩ := sampleSetupZHJ_of x (by rw [hx]; exact array_forM_ok hshape j.val hj)
    (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

set_option linter.unusedVariables false in
theorem setupZHJ_of_acceptsZK (I : Inputs) (hh : I.tags.hiddenOutputs = true) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : Assumptions.Zk.SampleAtZKHJ I dj mPts S)
    (j : Fin (nTab I)) :
    Nonempty (SampleSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  setupZHJ_of_acceptsZKI I hh ht hc dj h hd hok j

theorem sampleHypsZHJ (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0) (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    (∀ S', SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S') ∧
      ∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  ⟨sampleOkZKHJ_of_file I ht hc kd n hdiv dj hdj,
    setupZHJ_of_acceptsZK I hTags.1 ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj h hd
      (sampleOkZKHJ_of_file I ht hc kd n hdiv dj hdj S)⟩

/-- Not covered: the session's existence, `hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters and
`TableCRZ`. -/
theorem zk_session_soundHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_soundHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- Not covered: the session's existence, `hCust.1` (`RecordsLiveZKJ`), `hpub`, `hpt`, `hkd`, `hA3`, the count curve's
parameters and `TableCRZ`. -/
theorem zk_session_soundHJ_custody_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) =>
    zk_session_soundHJ_custody (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- Not covered: the session's existence, `hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters and
`TableCRZ`. -/
theorem zk_session_soundHJ_custody_json_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ :=
    sampleHypsZHJ I ht hTags hc hU hscope kd hdiv dj (sampleFileZK_of_toJson_subset dj kd n hdj) h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (k Rw M : ℕ) =>
    zk_session_soundHJ_custody_json (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
      (Ω := Unit) (hdj := hdj) (k := k) (Rw := Rw) (M := M)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- Not covered: the session's existence, `hRec`, `hpub`, `hpt`, `hkd`, `hA3`, the count curve's parameters, `TableCRZ`,
`hdm` (`avoidsMask`, which no check implies) and `hKey` (a named assumption). -/
theorem zk_session_composedHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_composedHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- Not covered: the session's existence and `hdm` (`avoidsMask`, which no check implies); `hT1` holds at the pinned
key's hiding gap (`CROnly.hm96Hiding_gap`). -/
theorem zk_session_viewHJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    (hscope : Layout.scopeOkH c = true) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ j : Fin (nTab I),
        Nonempty (SampleSetupZHJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨-, hy⟩ := sampleHypsZHJ I ht hTags hc hU hscope kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := zk_session_viewHJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy⟩

theorem sampleSetupZKH_whole_mPts (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) {kd pop : ℕ}
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) : mPts = mPtsOf c.g kd :=
  y₀.x.hmPts.symm.trans (setupHidden_mPts_of_proved hc (sampleFileZK_proved_size hdj S₀) y₀.x.h)

theorem sampleSetupZKH_whole_empty_16_2_1 (I : Inputs) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset 16) n) (S₀ : Finset (Fin n)) :
    IsEmpty (SampleSetupZKH I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) :=
  ⟨fun y₀ => by
    have h := sampleSetupZKH_whole_mPts I hc dj hdj y₀
    rw [hJ, hg] at h
    exact mPtsOf_16_ne_8 h.symm⟩

theorem y0_binderH_16_2_1 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZHJ I) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) (hscope : Layout.scopeOkH c = true)
    (hJ : nTab I = 2) (hg : c.g = 1) (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset 16) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    mPtsOf c.g (16 / nTab I) = 27 ∧
      Nonempty (SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (16 / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ S₀ : Finset (Fin n), IsEmpty (SampleSetupZKH I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) := by
  refine ⟨?_, (sampleHypsZHJ I ht hTags hc hU hscope 16 (by simp [hJ]) dj hdj h hd).2 ⟨0, nTab_pos I⟩,
    sampleSetupZKH_whole_empty_16_2_1 I hc hJ hg dj hdj⟩
  rw [hJ, hg]
  exact mPtsOf_16_8.2

end FlockSoundness.Discharge.ZkHidden
