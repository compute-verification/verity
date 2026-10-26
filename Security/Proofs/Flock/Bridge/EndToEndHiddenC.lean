import Proofs.Flock.Bridge.Canon
import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ

/-!
# `EndToEndHiddenC`, from `OpeningsCanonTarget`

`EndToEndHidden`'s argument with "wrong" `wrongRegZc`:

* `zk_session_soundHJ_custody_regC`: `zk_session_soundHJ_custody_reg` at `zk_flock_countC_regC`, for any `canon`
  under `OpeningsCanon`, at the same bound;
* `Flock.Bridge.endToEndHiddenC_of`: `OpeningsCanonTarget` gives `EndToEndHiddenC`. It is
  `zk_session_soundHJ_custody_regC` at `Bridge.canonAt` with its first and last terms in closed form, as
  `EndToEndHidden` is `zk_session_soundHJ_custody_reg`'s (`Audit.Law.subset_miss`, and `linkBoundZC` at A2);
* `Flock.Bridge.endToEndHidden_of_c`: `EndToEndHiddenC` gives `EndToEndHidden` (`wrongRegZ_subset_wrongRegZc`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec Composed
  Placed StrictCR DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n : ℕ} {Reg : Type}

/-- **One hidden-output `--zk` sampled session of `J` tables is sound against the registration with canonical
openings.** `zk_session_soundHJ_custody_reg` with "wrong" `wrongRegZc canon`, under `OpeningsCanon` at `canon`: a drawn
unit's satisfying witnesses open each of its committed wires canonically. -/
theorem zk_session_soundHJ_custody_regC
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOkH c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf c.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (canon : Fin ((keyProg c hU n).circuit (Layout.outsOf c hU n)).N →
      ((rsZHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)).Opening → Prop)
    (hcan : OpeningsCanon execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
      (tabZHJ dj y₀) ((rsZHJ (Reg := Reg) dj y₀ ht hc hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)) canon)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (wrongRegZc execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt)) canon
          (reg σ)
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJ dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJ_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
        fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ _ _ ?_
  refine (zk_flock_countC_regC execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _ canon
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJ dj y₀ ht hc hscope hU))
    (Layout.keyProg_readsOwn c hU n _) hcan
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

end FlockSoundness.Discharge.ZkHidden

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.ZkLink

/-- **`OpeningsCanonTarget` gives `EndToEndHiddenC`.** -/
theorem endToEndHiddenC_of (hOC : OpeningsCanonTarget) : EndToEndHiddenC := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ Cc rows κ hCR hA2
  have h := Discharge.ZkHidden.zk_session_soundHJ_custody_regC I ht (Flock.Desc.parse_flat hD) hU hpub hpt hscope kd
    hkd hdiv dj y₀ wr dg N Lb src hA3 hTags hCust hk hRw hM hρ hr ht' K₀
    (fun g o => canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)
    (hOC n Reg I ht D hD hU pub hpub hpt hscope kd dj S₀ d₀ y₀) τ
    (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, Discharge.ZkLink.linkBoundZC, ite_eq_left hA2] at h
  exact h

/-- **`EndToEndHiddenC` gives `EndToEndHidden`**: a unit wrong on the registration is wrong on it with canonical
openings (`wrongRegZ_subset_wrongRegZc`), so `EndToEndHidden`'s event implies `EndToEndHiddenC`'s. -/
theorem endToEndHidden_of_c (h : EndToEndHiddenC) : Flock.Guarantees.EndToEndHidden := by
  intro n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk
    hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h' := h n Reg I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M
    hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS hCR hA2
  refine le_trans (prob_mono ?_ _ _) h'
  intro o ho
  exact ⟨ho.1, ho.2.trans (Finset.card_le_card (wrongRegZ_subset_wrongRegZc _ _ _ _ _ _ _ _))⟩

end Flock.Bridge
