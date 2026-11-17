module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.BindInst
public import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJ
public import Proofs.Flock.Soundness.Discharge.ZkLink.Public
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
import Proofs.Flock.Soundness.Discharge.Composed.SampledJ
import Proofs.Flock.Soundness.Discharge.Composed.PublicJ
import Proofs.Flock.Soundness.Discharge.Tables.Headline
import Proofs.Flock.Soundness.Discharge.ZkLink.Count
import Proofs.Flock.Soundness.Game

@[expose] public section

set_option Elab.async false

/-! One hidden-output `--zk` sampled session of `J` tables at the template instance's program `instProg c n`, with no
condition on the unit's input groups, on where the outputs are returned or on shared-row tables. -/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockSoundness.Discharge.Inst

theorem instProg_readsOwn (c : Flock.Circuit) (n : ℕ) (outs : Finset (Fin (instN c n))) (u : Fin n) :
    ((instProg c n).partition outs).ReadsOwn u :=
  Layout.readsOwn_of_rows (fun u => Const.Prog.instC (instProg c n) outs u)
    (fun u y hy => instProg_reads outs u y hy) u

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

theorem zk_session_soundHJI
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hDraw : ∀ ω, SampleOkZKHJ I dj mPts ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω))
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hRec hDraw)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).incorrect
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hRec hDraw)) τ _ _ ?_
  refine (zk_flock_countC execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

theorem zk_session_soundHJ_custodyI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
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
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((instProg c n).partition (outsOfI c n)).incorrect
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJI I ht dj y₀ wr dg N hc hU hpub hrf hpr hw kd hkd Lb src hA3
    (fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

theorem rsZHJI_incorrectRegZ (dj : Finset (Fin n) → Lean.Json) {I : Inputs} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hrf : Hidden.Any.RowsFitS c) {pub : Public}
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hU : Built c.unit) (hpr : Layout.PortReads c hU)
    (hw : ProvedScope.wiresAcyclic c = true) (R R' : Reg) (Xv : Fin (instN c n) → Bool) :
    incorrectRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) R Xv =
      incorrectRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) R' Xv := by
  apply incorrectRegZ_congr
  intro p
  rfl

theorem zk_session_soundHJ_custody_regI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
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
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (incorrectRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
        fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ _ _ ?_
  refine (zk_flock_countC_reg execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    (instProg_readsOwn c n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

/-- The sample is the session's, so no sampling design over it is assumed. -/
theorem zk_session_soundHJ_custody_sampled_regI
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hrf : Hidden.Any.RowsFitS c) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)
    (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t')
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
          fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧ ∃ u ∈ o.2.1,
        u ∈ incorrectRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ)
          (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
            (reg σ) (cont σ) k Rw))
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ksAvgStrictZ (planZHJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  let W := incorrectRegZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
    ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)) (reg σ)
    (XplurZC execArith hs512 ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
      ((rsZHJI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHJI_computes dj y₀ ht hc hrf hpub hU hpr hw))
      (reg σ) (cont σ) k Rw)
  refine (execRefinesZK_tables_sampled execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_liveI I ht hc dj y₀ wr _ hTags hCust.1
        fun _ => sampleOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (fun S => ∃ u ∈ S, u ∈ W)).trans ?_
  refine (prob_mono (q := fun o : Bool × Finset (Fin n) => o.1 = true ∧ ¬ Disjoint W o.2)
    (fun o h => ⟨h.1, Finset.not_disjoint_iff.2 (let ⟨u, hu, hw⟩ := h.2; ⟨u, hw, hu⟩)⟩) _ _).trans ?_
  refine (zk_flock_sampledC_reg execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
    (instProg_readsOwn c n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' σ).trans ?_
  exact add_le_add (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR) le_rfl

end FlockSoundness.Discharge.ZkHidden
