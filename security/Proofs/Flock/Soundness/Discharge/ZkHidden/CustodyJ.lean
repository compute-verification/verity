module

public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundJ
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
import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt

/-! `zk_session_soundHJ` at the challenger's live-coin part's custody. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

theorem sampleOkZKHJ_of_sizeG (I : Inputs) {g : ℕ}
    (hg : ∀ (j : ℕ) (d : Lean.Json) (st : Flock.Stmt), Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile
      I.publicFile (zkCoins I) I.tables (some d) I.partition I.program = .ok st → st.c.g = g)
    (samplingDesign : Draw.SamplingDesign) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj samplingDesign pop) (S : Finset (Fin n))
    (hsz : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → D.units.size = kd) :
    SampleOkZKHJ I dj (mPtsOf g (kd / nTab I)) S := by
  intro j
  by_cases hD : ∃ D, Draw.UnitSample.ofJson (dj S) = .ok D
  swap
  · have hj : ∀ D', Draw.UnitSample.ofJson (drawJ I (dj S) j) ≠ .ok D' := by
      intro D' h'
      apply hD
      unfold drawJ at h'
      split_ifs at h'
      · exact ⟨_, h'⟩
      · unfold partJson at h'
        split at h'
        · exact ⟨_, ‹_›⟩
        · exact ⟨_, h'⟩
    refine ⟨fun D' h' => (hj D' h').elim, fun st hst => ?_⟩
    obtain ⟨ud, hud, -⟩ := setupHidden_mPts_sampled hst
    exact (hj ud hud).elim
  obtain ⟨D, hDj⟩ := hD
  have hDe := hdj S D hDj
  have hsz : D.units.size = kd := hsz D hDj
  have hS : D.units = ((S.image Fin.val).sort (· ≤ ·)).toArray := by rw [hDe]; rfl
  by_cases h1 : I.count ≤ 1
  · have hn : nTab I = 1 := by simp [nTab, h1]
    have hdr : drawJ I (dj S) j = dj S := by simp [drawJ, h1]
    have hpart : partS I (dj S) S j = S := by
      refine Finset.filter_true_of_mem fun u _ => Fin.ext ?_
      have h1 := (partOf I (dj S) u.val).isLt
      have h2 := j.isLt
      omega
    refine ⟨fun D' h' u hu => ?_, fun st hst => ?_⟩
    · rw [hdr, hDj] at h'
      rw [← Except.ok.inj h', hDe, sampleOf_proved, sampleOf_mem]
      exact ⟨u, by rwa [hpart] at hu, rfl⟩
    · rw [setupHidden_mPts_of_provedG (hg j _ _ hst) (fun D' h' => ?_) hst]
      rw [hdr, hDj] at h'
      rw [← Except.ok.inj h', hn, Nat.div_one, hDe, sampleOf_proved, ← hDe, hsz]
  · have hn : nTab I = I.count := by simp [nTab, h1]
    have hdr : drawJ I (dj S) j = Tables.partSample D.population D.units I.count j := by
      simp only [drawJ, h1, ite_false, partJson, hDj]
    have hq : kd = I.count * (kd / I.count) := by
      rw [hn] at hdiv
      exact (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)).symm
    have hjq : j.val * (kd / I.count) + kd / I.count ≤ kd := by
      have hj : j.val < I.count := by have := j.isLt; omega
      have := Nat.mul_le_mul_right (kd / I.count) (Nat.succ_le_of_lt hj)
      rw [Nat.succ_mul] at this
      omega
    have hpd : ∀ D', Draw.UnitSample.ofJson (drawJ I (dj S) j) = .ok D' →
        D'.proved = D.units.extract (j.val * (kd / I.count)) (j.val * (kd / I.count) + kd / I.count) := by
      intro D' h'
      rw [hdr] at h'
      rw [ofJson_partSample h', hsz]
    refine ⟨fun D' h' u hu => ?_, fun st hst => ?_⟩
    · rw [hpd D' h']
      obtain ⟨huS, hpu⟩ := Finset.mem_filter.1 hu
      have hmem : u.val ∈ D.units.toList := by
        rw [hS]
        simpa [Finset.mem_sort] using ⟨u, huS, rfl⟩
      have hi : D.units.toList.idxOf u.val < D.units.toList.length := List.idxOf_lt_length_of_mem hmem
      have hget : D.units.toList[D.units.toList.idxOf u.val] = u.val := List.getElem_idxOf hi
      have hpo := congrArg Fin.val hpu
      simp only [partOf, hDj, hsz] at hpo
      generalize D.units.toList.idxOf u.val = i at hi hget hpo
      rw [Array.length_toList, hsz] at hi
      have hq0 : 0 < kd / I.count := by
        rcases Nat.eq_zero_or_pos (kd / I.count) with h0 | h0
        · rw [h0, Nat.mul_zero] at hq; omega
        · exact h0
      have hlt : i / (kd / I.count) < I.count := (Nat.div_lt_iff_lt_mul hq0).2 (by rw [← hq]; exact hi)
      have hji : i / (kd / I.count) = j.val := by omega
      have h1 := Nat.div_add_mod i (kd / I.count)
      have h2 := Nat.mod_lt i hq0
      rw [hji] at h1
      rw [Array.mem_def, Array.toList_extract, List.extract_eq_take_drop]
      refine List.mem_iff_getElem.2 ⟨i - j.val * (kd / I.count), ?_, ?_⟩
      · simp only [List.length_drop, List.length_take, Array.length_toList, hsz]
        rw [Nat.mul_comm] at h1
        omega
      · rw [List.getElem_take, List.getElem_drop]
        convert hget using 2
        rw [Nat.mul_comm] at h1
        omega
    · rw [setupHidden_mPts_of_provedG (hg j _ _ hst) (fun D' h' => ?_) hst]
      rw [hpd D' h', Array.size_extract, hsz, Nat.min_eq_left hjq, Nat.add_sub_cancel_left, hn]

theorem sampleOkZKHJ_of_size (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (samplingDesign : Draw.SamplingDesign) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj samplingDesign pop) (S : Finset (Fin n))
    (hsz : ∀ D, Draw.UnitSample.ofJson (dj S) = .ok D → D.units.size = kd) :
    SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S :=
  sampleOkZKHJ_of_sizeG I (fun _ _ _ h => congrArg Flock.Circuit.g (setupHidden_c_of_parse
    ((parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc) h)) samplingDesign kd pop hdiv dj hdj S hsz

theorem sampleOkZKHJ_of_fileG (I : Inputs) {g : ℕ}
    (hg : ∀ (j : ℕ) (d : Lean.Json) (st : Flock.Stmt), Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile
      I.publicFile (zkCoins I) I.tables (some d) I.partition I.program = .ok st → st.c.g = g)
    (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    SampleOkZKHJ I dj (mPtsOf g (kd / nTab I)) S :=
  sampleOkZKHJ_of_sizeG I hg (.subset kd) kd pop hdiv dj hdj S fun D hD =>
    (ofJson_shape hD).2.2 kd (by rw [hdj S D hD]; rfl)

theorem sampleOkZKHJ_of_file (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S :=
  sampleOkZKHJ_of_fileG I (fun _ _ _ h => congrArg Flock.Circuit.g (setupHidden_c_of_parse
    ((parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc) h)) kd pop hdiv dj hdj S

theorem sampleOkZKHJ_of_fileSt (I : Inputs) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (kd pop : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (hdj : SampleFileZK dj (.subset kd) pop) (S : Finset (Fin n)) :
    SampleOkZKHJ I dj (mPtsOf c.g (kd / nTab I)) S :=
  sampleOkZKHJ_of_fileG I (fun j _ _ h => (stmt_g_of_setupHidden (typedJ ht j)
    ((parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc) h).symm) kd pop hdiv dj hdj S

theorem zk_session_soundHJ_custody
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJ I ht dj y₀ wr dg N hc hU hpub hpt hscope kd hkd Lb src hA3
    (fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.ZkHidden
