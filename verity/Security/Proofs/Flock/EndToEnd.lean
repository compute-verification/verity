import Specs.Flock.Guarantees.EndToEnd

/-!
# Proof of C-Flock's end-to-end guarantee

`Flock.SecurityProofs.EndToEnd : Flock.Guarantees.EndToEnd` is `zk_session_soundJ_custody` with its two last terms in
closed form:
* `zk_session_soundJ_custody`: one `--zk` session of `J` tables at the record's custody, at the live subset draw, is
  sound, `Pr[flock-verify accepts ∧ at least K₀ wrong units] ≤ miss(K₀) + ε_ks + δ_link`;
* `Audit.Law.subset_miss`: the uniform `kd`-subset's `miss(K₀)` is `C(n − K₀, kd) / C(n, kd)` at `K₀ ≤ n`;
* `linkBoundZC_of_linkCR`: at a prover whose link finders satisfy A2 (`LinkCRZC`), `linkBoundZC` is its closed form.
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Model FlockSoundness.Discharge.ZkLink
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
  {D Sl : Type} [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ}
  (P : Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) {L : Law n}
  {decode : DecoderZK (C := C) planZ tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : ValueBindingZ A P planZ tab decode Hc) (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R)

open Classical in
/-- **`δ_link` at a prover whose link finders satisfy A2** is `linkBoundZC`'s closed form. -/
theorem linkBoundZC_of_linkCR {k Rw M : ℕ} {ρ t' : ℝ} (h : LinkCRZC A Hs P planZ tab vb R τ k Rw M t') :
    linkBoundZC A Hs P planZ tab vb k Rw M ρ t' R τ =
      ENNReal.ofReal ((∑ p : vb.Pos, (Fintype.card (ΩcZ A P planZ tab vb L p) : ℝ) / Fintype.card L.Ω) *
        (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) := by
  unfold linkBoundZC
  exact if_pos h

/-- **C-Flock's one-stage audit, end to end.** -/
theorem EndToEnd : Flock.Guarantees.EndToEnd := by
  intro n Reg I ht c hc hU pub hpub hpt hscope kd hkd hdiv dj S₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags hCust k Rw M hk hRw
    hM ρ hρ hr t' ht' K₀ hK₀ τ qF qS L P vb σ hCR hA2
  have h := Discharge.Composed.zk_session_soundJ_custody I ht hc hU hpub hpt hscope kd hkd hdiv dj y₀ wr dg N Lb src hA3
    hTags hCust hk hRw hM hρ hr ht' K₀ τ (qF := qF) (qS := qS) hCR
  rw [Law.subset_miss hkd hK₀, linkBoundZC_of_linkCR _ _ _ _ _ _ _ _ hA2] at h
  exact h

end Flock.SecurityProofs
