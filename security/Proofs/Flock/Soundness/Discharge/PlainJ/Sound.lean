module

public import Proofs.Flock.Soundness.Discharge.PlainJ.Decode
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

/-! The soundness half of one plain sampled session of `J` tables (`plain_session_soundJ`). -/

@[expose] public section

namespace FlockSoundness.Discharge.PlainJ

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.Tables
open FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Composed (TagsOkZJ)
open FlockSoundness.Discharge.DecodeZLJ (decJ RecordsLiveJ)
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

theorem plain_session_soundJ
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (x₀ : SampleSetup (inputsPJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planPJ (Reg := Reg) ht dj x₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {c : Flock.Circuit} (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic I.tags c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hDraw : ∀ ω, SampleOkPJ I dj mPts ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω))
    (hTags : TagsOkZJ I) (hRec : RecordsLiveJ I dj (fun S R => (planPJ (Reg := Reg) ht dj x₀ S R).length) wr)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (planPJ (Reg := Reg) ht dj x₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPJ ht dj x₀) N wr)))
    {qF qS : ℝ} :
    let σ := simAuditJ H512 enc512 leaf512 (planPJ ht dj x₀) I wr (opnPJ I (planPJ ht dj x₀) dj) dg
      (decJ (planPJ ht dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnPJ_exec ht dj x₀ _ wr hTags hRec hDraw) τ
    -- ecr/sha-512 for the link finders of its simulation
    LinkCRL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planPJ ht dj x₀)
      (tabPJ ht dj x₀) ((rsPJ ht dj x₀ hc hpub hpt hU hscope).binding (rsPJ_computes ht dj x₀ hc hpub hpt hU hscope))
      (reg σ) (cont σ) k Rw M t' →
    -- cr/sha-512 for its knowledge finders
    (∀ ω j, TableCRL execArith H512 enc512 leaf512
      (planPJ ht dj x₀ ((SamplingDesign.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (batchStratL H512 enc512 leaf512 (planPJ ht dj x₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAccepts I o ∧
        K ≤ (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).incorrect
          (Xpub {Layout.oneG} {Layout.zeroG}
            (XplurL ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) H512 enc512 leaf512 (planPJ ht dj x₀)
              (tabPJ ht dj x₀)
              ((rsPJ ht dj x₀ hc hpub hpt hU hscope).binding (rsPJ_computes ht dj x₀ hc hpub hpt hU hscope))
              (reg σ) (cont σ) k Rw))).card)
        (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfJ dg (planPJ ht dj x₀) N wr)) τ ≤
      (SamplingDesign.subset n kd hkd).miss K +
        StrictCR.ksAvgStrict (planPJ ht dj x₀) (SamplingDesign.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).linkBoundE
          (L := SamplingDesign.execOS (n := n) src (.subset kd) n) (planPJ (Reg := Reg) ht dj x₀) (tabPJ (Reg := Reg) ht dj x₀)
          ((rsPJ ht dj x₀ hc hpub hpt hU hscope (Reg := Reg)).binding (rsPJ_computes ht dj x₀ hc hpub hpt hU hscope)) k Rw M
          ρ t' := by
  intro σ hCR hKS
  exact (Placed.keyProg c hU n).flock_headline_live_tables_subset_miss (Layout.outsOf c hU n) kd hkd Lb src hA3
    (planPJ (Reg := Reg) ht dj x₀) (tabPJ ht dj x₀)
    (Placed.Sites.progPlaces_keyProg hU (sitesPJ_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.oneG} (Const.constCols_keyProg hU (sitesPJ_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    {Layout.zeroG} (Zero.zeroCols_keyProg hU (sitesPJ_parse ht dj x₀ hc) (Layout.outsOf c hU n))
    (layPJ ht dj x₀ hc hU (pub := pub)) (rsPJ ht dj x₀ hc hpub hpt hU hscope)
    (rsPJ_computes ht dj x₀ hc hpub hpt hU hscope) hk hRw hM hρ hr ht' K I wr
    (opnPJ I (planPJ (Reg := Reg) ht dj x₀) dj) dg (decJ (planPJ (Reg := Reg) ht dj x₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N (decodesOnPJ_exec ht dj x₀ _ wr hTags hRec hDraw) τ hCR hKS

end FlockSoundness.Discharge.PlainJ
