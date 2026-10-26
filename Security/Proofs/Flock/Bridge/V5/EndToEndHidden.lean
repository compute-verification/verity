import Proofs.Flock.Bridge.V5.Statement
import Proofs.Flock.Bridge.EndToEndHiddenC

/-!
# `EndToEndHiddenV5`, from `OpeningsCanonTargetV5`

`EndToEndHiddenC.lean` at the verifier's public inputs:

* `Layout.keyProgP_readsOwn`: `keyProgP`'s units read only their own gates and input gates (`keyProgP_reads`);
* `zk_session_soundHJ_custody_regCP`: `zk_session_soundHJ_custody_regC` at flock-verify `--zk --public-inputs`'s
  acceptance (`zk_headline_exec_tablesA` at `AcceptsZKP I own`, through `decodesZCJH_custodyP`), at `keyProgP` and the
  binding `rsZHJP`;
* `Flock.Bridge.endToEndHiddenV5_of`: `OpeningsCanonTargetV5` gives `EndToEndHiddenV5`, with its first and last terms
  in closed form as `endToEndHiddenC_of`'s.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

/-- **`keyProgP`'s units read only their own gates and input gates** (`keyProg_readsOwn`). -/
theorem keyProgP_readsOwn (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports)
    (outs : Finset (Fin (keyN c hU n))) (u : Fin n) : ((keyProgP c pi hU n).partition outs).ReadsOwn u :=
  readsOwn_of_rows (fun u => Const.Prog.instC (keyProgP c pi hU n) outs u)
    (fun u y hy => keyProgP_reads c hU n pi outs u y hy) u

end FlockSoundness.Discharge.Layout

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session of `J` tables at the verifier's public inputs is sound against the
registration with canonical openings** (`zk_session_soundHJ_custody_regC` at `own`). -/
theorem zk_session_soundHJ_custody_regCP
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJP I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (canon : Fin ((keyProgP c pi hU n).circuit (Layout.outsOfP c hU n pi)).N →
      ((rsZHJP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)).Opening → Prop)
    (hcan : OpeningsCanon execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
      (tabZHJP dj y₀) ((rsZHJP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)) canon)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht hc kd hdiv dj y₀ wr _ hTags hCust) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKP I own o ∧
        K₀ ≤ (wrongRegZc execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)) canon
          (reg σ)
          (XplurZC execArith hs512 ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
            (tabZHJP dj y₀)
            ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJP dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tablesA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr
    (opnAtZHJP (planZHJP dj y₀) I own dj) dg (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N (decodesZCJH_custodyP I own ht hc kd hdiv dj y₀ wr _ hTags hCust) τ _ _ ?_
  refine (zk_flock_countC_regC execArith hs512 _ (planZHJP dj y₀) (tabZHJP dj y₀) _ canon
    (loweringSoundZK_of_place execArith _ (planZHJP dj y₀) (tabZHJP dj y₀) (placeZHJP dj y₀ ht hc hscope hU hpi))
    (Layout.keyProgP_readsOwn c hU n pi _) hcan
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJP dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJP dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

end FlockSoundness.Discharge.ZkHidden

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.ZkLink

/-- **`OpeningsCanonTargetV5` gives `EndToEndHiddenV5`.** -/
theorem endToEndHiddenV5_of (hOC : OpeningsCanonTargetV5) : EndToEndHiddenV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw
    M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ Cc Lm rows κ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_regCP I own ht (Flock.Desc.parse_flat hD) hU hpi hpub hscope
    kd hkd hdiv dj y₀ wr dg N Lb src hA3 hTags hCust hk hRw hM hρ hr ht' K₀
    (fun g o => canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
    (hOC n Reg I own ht D hD hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀) τ
    (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

end Flock.Bridge
