module

public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.EndToEndHiddenC

/-!
# `EndToEndHiddenV5`, from `OpeningsCanonTargetV5`

`EndToEndHiddenC.lean` at the verifier's public inputs:

* `Layout.keyProgP_readsOwn`: `keyProgP`'s proof blocks read only their own gates and input gates (`keyProgP_reads`);
* `zk_session_soundHJ_custody_escCP`: at flock-verify `--zk --public-inputs`'s acceptance (`zk_headline_exec_tablesA`
  at `AcceptsZKP I own`, through `decodesZCJH_custodyP` under the registration's law), at `keyProgP` and the binding
  `rsZHJP`, a set `W` of proof blocks is wrong with probability at most the sample's escape of `W` and the extraction's
  terms (`zk_flock_escapeC_regC`), at any sample `L`;
* `Flock.Bridge.endToEndHiddenV5_of`: `OpeningsCanonTargetV5` gives `EndToEndHiddenV5`, at the verifier's sample
  `execSel` under the sampling design, whose escape of `W` is at most the sampling design's `P_a` of `W` under A3 (`execSel_escape_le`), and the
  last term in closed form as `endToEndHiddenC_of`'s.
-/

@[expose] public section

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Audit Placed

/-- **`keyProgP`'s proof blocks read only their own gates and input gates** (`keyProg_readsOwn`). -/
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

/-- **One hidden-output `--zk` sampled session of `J` tables at the verifier's public inputs, under the registration's
law, is sound against the registration with canonical openings, at a set of proof blocks**: every proof block of `W` is
wrong with probability at most the sample's escape of `W` and the extraction's terms (`zk_flock_escapeC_regC` at `own`). -/
theorem zk_session_soundHJ_custody_escCP
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi)
    {pub : Public} (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)
    (hscope : Layout.scopeOkH c = true)
    (law : Draw.Law) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) (L : Law n)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    (hsz : ∀ S D, Draw.UnitDraw.ofJson (dj S) = .ok D → D.units.size = kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (W : Finset (Fin n))
    (canon : Fin ((keyProgP c pi hU n).circuit (Layout.outsOfP c hU n pi)).N →
      ((rsZHJP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)).Opening → Prop)
    (hcan : OpeningsCanon execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
      (tabZHJP dj y₀) ((rsZHJP (Reg := Reg) dj y₀ ht hc hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)) canon)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht hc law kd hdiv dj y₀ wr L hTags hCust hsz) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZKP I own o ∧
        W ⊆ wrongRegZc execArith ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub)) canon
          (reg σ)
          (XplurZC execArith hs512 ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
            (tabZHJP dj y₀)
            ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub))
            (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      L.escape W +
        ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProgP c pi hU n).partition (Layout.outsOfP c hU n pi)) (planZHJP dj y₀)
          (tabZHJP dj y₀)
          ((rsZHJP dj y₀ ht hc hscope hU hpi hpub).bindingZ (rsZHJP_computes dj y₀ ht hc hscope hU hpi hpub))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tablesA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr
    (opnAtZHJP (planZHJP dj y₀) I own dj) dg (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => [])
    (fun _ _ rs j => capOfRoot rs[j.val]!) N (decodesZCJH_custodyP I own ht hc law kd hdiv dj y₀ wr L hTags hCust hsz)
    τ _ _ ?_
  refine (zk_flock_escapeC_regC execArith hs512 _ (planZHJP dj y₀) (tabZHJP dj y₀) _ canon
    (loweringSoundZK_of_place execArith _ (planZHJP dj y₀) (tabZHJP dj y₀) (placeZHJP dj y₀ ht hc hscope hU hpi))
    (Layout.keyProgP_readsOwn c hU n pi _) hcan
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJP dj y₀) hk) ht' W σ).trans ?_
  exact add_le_add (add_le_add le_rfl
    (ksAvgZC_le_strict execArith hs512 (planZHJP dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

end FlockSoundness.Discharge.ZkHidden

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.ZkLink

/-- **`OpeningsCanonTargetV5` gives `EndToEndHiddenV5`**: the session's bound at the verifier's sample under the sampling design,
whose escape of `W` is at most the sampling design's `P_a` of `W` under A3 (`execSel_escape_le`). -/
theorem endToEndHiddenV5_of (hOC : OpeningsCanonTargetV5) : EndToEndHiddenV5 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope law kd ns hns hlaw hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags
    hCust k Rw M hk hRw hM ρ hρ hr t' ht' W τ qF qS L P vb σ Cc Lm rows κ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_escCP I own ht (Flock.Desc.parseStmt_stmt hD) hU hpi hpub
    hscope law kd hdiv dj y₀ wr dg N L hTags hCust (lawOn_size hlaw hCust.2) hk hRw hM hρ hr ht' W
    (fun g o => canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
    (hOC n Reg I own ht D hD hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀) τ
    (qF := qF) (qS := qS) hCR
  rw [Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h.trans (add_le_add (add_le_add (execSel_escape_le src hA3 hlaw hns W) le_rfl) le_rfl)

end Flock.Bridge
