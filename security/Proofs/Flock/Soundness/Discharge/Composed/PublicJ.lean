module

public import Proofs.Flock.Soundness.Discharge.Composed.CustodyJ
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
import Proofs.Flock.Soundness.Discharge.Tables.Headline

/-! `zk_session_soundJ_custody` with "wrong" measured against the public file (`incorrectRegZ`), at the same bound. -/

@[expose] public section

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

theorem readsOwn_of_rows {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {R : Fin n → Rows}
    (I : ∀ u, P.IsRowsUnit u (R u))
    (hread : ∀ u (y : Fin (R u).w), (∀ i, y ≠ (R u).comp i) → P.unit ((I u).wire y) = none) (u : Fin n) :
    P.ReadsOwn u := by
  intro h x hh hx
  obtain ⟨i, rfl⟩ := (I u).onto h hh
  rw [(I u).op i] at hx
  simp only [Verity.Boolean.Op.args, List.mem_append, List.mem_map] at hx
  obtain ⟨y, rfl⟩ : ∃ y, (I u).wire y = x := by
    rcases hx with ⟨y, -, e⟩ | ⟨y, -, e⟩ <;> exact ⟨y, e⟩
  by_cases hcomp : ∃ j, y = (R u).comp j
  · obtain ⟨j, rfl⟩ := hcomp
    exact Or.inl ((I u).mem j)
  · exact Or.inr (hread u y fun j e => hcomp ⟨j, e⟩)

theorem keyProg_readsOwn (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (outs : Finset (Fin (keyN c hU n)))
    (u : Fin n) : ((keyProg c hU n).partition outs).ReadsOwn u :=
  readsOwn_of_rows (fun u => Const.Prog.instC (keyProg c hU n) outs u)
    (fun u y hy => keyProg_reads c hU n outs u y hy) u

end FlockSoundness.Discharge.Layout

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

theorem rsZJ_incorrectRegZ (dj : Finset (Fin n) → Lean.Json) {I : Inputs} {mPts : ℕ} {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit) {pub : Public}
    (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true) (R R' : Reg) (Xv : Fin (Layout.keyN c hU n) → Bool) :
    incorrectRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
        ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) R Xv =
      incorrectRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
        ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) R' Xv := by
  apply incorrectRegZ_congr
  intro p
  rfl

theorem zk_session_soundJ_custody_reg
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : SampleSetupZJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => sampleOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZJ dj y₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (incorrectRegZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope)) (reg σ)
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
            ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZJ dj y₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZJ dj y₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
          ((rsZJ dj y₀ ht hc hU hpub hpt hscope).bindingZ (rsZJ_computes dj y₀ ht hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) dg
    (decAtZJ (planZJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJ_live I ht dj y₀ wr _ hTags hCust.1 fun _ => sampleOkZKJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    _ _ ?_
  refine (zk_flock_countC_reg execArith hs512 _ (planZJ dj y₀) (tabZJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZJ dj y₀) (tabZJ dj y₀)
      ((sitesZJ dj y₀ ht).placeKey hU (sitesZJ_parse dj y₀ ht hc) (Layout.outsOf c hU n)))
    (Layout.keyProg_readsOwn c hU n _)
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

end FlockSoundness.Discharge.Composed
