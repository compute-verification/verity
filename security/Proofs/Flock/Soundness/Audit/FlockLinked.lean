module

public import Proofs.Flock.Soundness.Audit.FlockLink

/-! The compiled-layer audit with `ε_bind` discharged by the link theorem. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

/-- **The binding error the link theorem gives**: `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`. -/
noncomputable def linkBoundE (k Rw M : ℕ) (ρ t' : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω) *
    (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

theorem linkBoundCR_eq {k Rw M : ℕ} {ρ t' : ℝ} {R : Reg} {τ : Cont L Reg (batchedSession H E plan) R}
    (hCR : P.LinkCR H E plan tab vb R τ k Rw M t') :
    P.linkBoundCR H E plan tab vb k Rw M ρ t' R τ = P.linkBoundE (L := L) plan tab vb k Rw M ρ t' := by
  unfold linkBoundCR
  simp only [hCR, ↓reduceIte]
  rfl

theorem flock_batched_count_linked (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_countE H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE H E plan tab vb hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

theorem flock_batched_sampled_linked (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.incorrect (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_sampledE H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE H E plan tab vb hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

end Partition

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n}

/-- The decoder that reads each sampled proof block from its place in its table (`loweringSoundB_of_place`). -/
noncomputable def placeDecoder (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) :
    DecoderB (C := C) (Reg := Reg) plan tab := fun S R u M =>
  if h : u ∈ S then (place S R u h).decode
    (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
      (tabOf plan tab S R u).sch.level₀.logCols M)
  else fun _ => false

end Partition

end FlockSoundness.Audit
