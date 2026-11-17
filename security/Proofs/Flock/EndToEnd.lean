module

public import Proofs.Flock.EndToEnd.Statement
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
import Proofs.Flock.Soundness.Audit.SamplingDesign
import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
import Proofs.Flock.Soundness.Discharge.Layout.Drawn
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-! The proof of `Flock.Guarantees.EndToEnd`: `zk_session_soundJ_custody_reg` with its first and last terms in closed
form. -/

@[expose] public section

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- **Flock's one-stage audit, end to end.** -/
theorem EndToEnd : Flock.Guarantees.EndToEnd := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody_reg I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb src hA3
    hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [SamplingDesign.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

open FlockSoundness.Discharge FlockSoundness.Discharge.Exec FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
  FlockSoundness.Discharge.Composed in
theorem EndToEnd_refSetup {n : ℕ} (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : Flock.HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) (hT : TagsOkZJ I)
    {S : Finset (Fin n)} {t : Exec.Transcript} (hacc : AcceptsZK I t) (hd : recordDraw t.record = some (dj S)) :
    Nonempty (SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) (partS I (dj S) S ⟨0, nTab_pos I⟩)
      (drawJ I (dj S) 0)) := by
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hT.1 hd hs
  have hj : 0 < sts.size := hsz ▸ nTab_pos I
  have hst := hstJ 0 hj
  obtain ⟨-, _, schs, -, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨_hs0, _, hz0, -⟩ := hall 0 (by simpa using hj)
  simp only [Array.getElem_map] at hz0
  have hz' : Flock.Zk.schedule sts[0].wideLeaf sts[0].m = .ok (schs[0].1, schs[0].2) := hz0
  obtain ⟨-, -, _, hf⟩ := schedule_spec hz'
  have hDraw := sampleOkZKJ_of_file I ht hc kd pop hdiv dj hdj S ⟨0, nTab_pos I⟩
  obtain ⟨_, -, hdr⟩ := Layout.setupH_sampled hst
  obtain ⟨D, hD, -⟩ := Layout.sampled_spec hdr
  obtain ⟨x, hx⟩ := sampleSetup_of_setupH_g (I := inputsJ I 0) (S := partS I (dj S) S ⟨0, nTab_pos I⟩)
    (mPts := mPtsOf c.g (kd / nTab I)) (typedJ ht 0) hst hf hD (hDraw.1 D hD) (hDraw.2 _ hst)
  obtain ⟨y, -⟩ := sampleSetupZJ_of (typedJ ht 0) x (sch := schs[0].1) (tPad := schs[0].2) (by rw [hx]; exact hz')
  exact ⟨y⟩

end Flock.SecurityProofs
