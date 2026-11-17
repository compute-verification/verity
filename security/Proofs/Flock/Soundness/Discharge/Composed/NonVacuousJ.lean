module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJJson
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
import Proofs.Flock.Soundness.Discharge.Composed.SoundJ
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-! The `J`-table statements' reference setup `y₀` and sample hypotheses hold together, and a whole-draw `y₀` does not exist; that a session `verify --zk` accepts exists at a sample of the file is not covered. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {n mPts : ℕ}

theorem setupZJ_of_acceptsZK (I : Inputs) (hh : I.tags.hiddenOutputs = false) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t)
    (hd : recordDraw t.record = some (dj S)) (hok : Assumptions.Zk.SampleAtZKJ I dj mPts S)
    (j : Fin (nTab I)) :
    Nonempty (SampleSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) := by
  obtain ⟨spec, sts, hs, -, hv⟩ := h
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hh hd hs
  have hj : j.val < sts.size := by have := j.isLt; omega
  have hst := hstJ j.val hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, _, hz0, -⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Zk.schedule sts[j.val].wideLeaf sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ ht j.val
  obtain ⟨_, -, hdr⟩ := Layout.setupH_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec hdr
  obtain ⟨x, hx⟩ := sampleSetup_of_setupH_g (I := inputsJ I j.val) (S := partS I (dj S) S j) (mPts := mPts)
    ht' hst hf hD ((hok j).1 D hD) ((hok j).2 _ hst)
  obtain ⟨y, -, -, -⟩ := sampleSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

theorem sampleHypsZJ (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    (∀ S', SampleOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S') ∧
      ∀ j : Fin (nTab I),
        Nonempty (SampleSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j)) :=
  ⟨sampleOkZKJ_of_file I ht hc kd n hdiv dj hdj,
    setupZJ_of_acceptsZK I hTags.1 ht dj h hd (sampleOkZKJ_of_file I ht hc kd n hdiv dj hdj S)⟩

/-- The proof applies `zk_session_soundJ` to this `y₀` and `hDraw`, so a binder that no longer admits them fails the build.
Not covered: the session's existence, `hRec`, `hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's
parameters and `TableCRZ`. -/
theorem zk_session_soundJ_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZJ I ht hTags hc kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (Lb : Fin 1 → ℕ) (src : Unit → (i : Fin 1) → Fin (Lb i) → Fin 256) =>
    zk_session_soundJ (Reg := Unit) (I := I) (dj := dj) (mPts := mPtsOf c.g (kd / nTab I)) (y₀ := y₀) (c := c)
      (pub := pub) (kd := kd) (Ω := Unit) (Lb := Lb) (src := src) (hDraw := fun _ => hok _)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- The proof applies `zk_session_soundJ_custody` to this `y₀`. Not covered: the session's existence, `hCust.1`
(`RecordsLiveZKJ`), `hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundJ_custody_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZJ I ht hTags hc kd hdiv dj hdj h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) =>
    zk_session_soundJ_custody (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
  exact ⟨⟨y₀⟩, hy, hok⟩

/-- The proof applies `zk_session_soundJ_custody_json` to the `y₀` it builds and to `hdj`. Not covered: the session's
existence, `hRec`, `hU`, `hpub`, `hpt`, `hscope`, `hkd`, `hA3`, the count curve's parameters and `TableCRZ`. -/
theorem zk_session_soundJ_custody_json_nonvacuous (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : ∀ S, dj S = (drawOf (.subset kd) n S).toJson)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      (∀ j : Fin (nTab I),
        Nonempty (SampleSetupZJ (inputsJ I j) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S j) (drawJ I (dj S) j))) ∧
      ∀ S', SampleOkZKJ I dj (mPtsOf c.g (kd / nTab I)) S' := by
  obtain ⟨hok, hy⟩ := sampleHypsZJ I ht hTags hc kd hdiv dj (sampleFileZK_of_toJson_subset dj kd n hdj) h hd
  obtain ⟨y₀⟩ := hy ⟨0, nTab_pos I⟩
  have _ := fun (pub : Public) (k Rw M : ℕ) =>
    zk_session_soundJ_custody_json (Reg := Unit) (I := I) (c := c) (pub := pub) (kd := kd) (dj := dj) (y₀ := y₀)
      (Ω := Unit) (hdj := hdj) (k := k) (Rw := Rw) (M := M)
  exact ⟨⟨y₀⟩, hy, hok⟩

theorem sampleFileZK_proved_size {kd pop : ℕ} {dj : Finset (Fin n) → Lean.Json} (hdj : SampleFileZK dj (.subset kd) pop)
    (S : Finset (Fin n)) : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → D.proved.size = kd := by
  intro D hD
  have hDe := hdj S D hD
  have hsz : D.units.size = kd := (ofJson_shape hD).2.2 kd (by rw [hDe]; rfl)
  rw [hDe, sampleOf_proved, ← hDe, hsz]

theorem sampleSetupZK_whole_mPts (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) {kd pop : ℕ}
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZK I mPts S₀ (dj S₀)) : mPts = mPtsOf c.g kd := by
  obtain ⟨ud, hud, hm⟩ := setupH_mPts_sampled y₀.x.h
  have hcst := (setupH_facts y₀.x.h).1
  rw [zkTyped ht] at hcst
  simp only [Bool.false_eq_true, ↓reduceIte] at hcst
  have hst : y₀.x.st.c = c := Except.ok.inj (hcst.symm.trans hc)
  rw [← y₀.x.hmPts, hm, hst, sampleFileZK_proved_size hdj S₀ ud hud]

theorem e2e_forces_mPts (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd : ℕ)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : SampleSetupZK I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
    (hdj : SampleFileZK dj (.subset kd) n) :
    mPtsOf c.g kd = mPtsOf c.g (kd / nTab I) :=
  (sampleSetupZK_whole_mPts I ht hc dj hdj y₀).symm

theorem mPtsOf_16_8 : mPtsOf 1 16 = 28 ∧ mPtsOf 1 8 = 27 := by decide

theorem mPtsOf_16_ne_8 : mPtsOf 1 16 ≠ mPtsOf 1 8 := by decide

theorem sampleSetupZK_whole_empty_16_2_1 (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset 16) n) (S₀ : Finset (Fin n)) :
    IsEmpty (SampleSetupZK I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) :=
  ⟨fun y₀ => by
    have h := e2e_forces_mPts I ht hc 16 dj y₀ hdj
    rw [hJ, hg] at h
    exact mPtsOf_16_ne_8 h⟩

theorem y0_binder_16_2_1 (I : Inputs) (ht : I.tags.typed = false) (hTags : TagsOkZJ I) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hJ : nTab I = 2) (hg : c.g = 1)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset 16) n)
    {S : Finset (Fin n)} {t : Exec.Transcript} (h : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    mPtsOf c.g (16 / nTab I) = 27 ∧
      Nonempty (SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (16 / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
        (drawJ I (dj S) 0)) ∧
      ∀ S₀ : Finset (Fin n), IsEmpty (SampleSetupZK I (mPtsOf c.g (16 / nTab I)) S₀ (dj S₀)) := by
  refine ⟨?_, (sampleHypsZJ I ht hTags hc 16 (by simp [hJ]) dj hdj h hd).2 ⟨0, nTab_pos I⟩,
    sampleSetupZK_whole_empty_16_2_1 I ht hc hJ hg dj hdj⟩
  rw [hJ, hg]
  exact mPtsOf_16_8.2

end FlockSoundness.Discharge.Composed
