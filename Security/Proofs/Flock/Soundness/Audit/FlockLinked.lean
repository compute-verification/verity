import Proofs.Flock.Soundness.Audit.FlockLink

/-!
# The compiled-layer audit with `δ_link` discharged

The batched Flock audit with the expected-time extractor (`flock_batched_countE`), composed with the link theorem
(`flock_batched_linkSoundE`) at the plurality value layer (`Xplur`). The count curve and the drawn-unit bound no longer
carry `δ_link` as a hypothesis:
$$\Pr[\text{accept} \wedge \text{at least } K \text{ wrong}] \le \mathrm{miss}_L(K) + \varepsilon_{ks}(\sigma) +
\frac{Q_s}{1-\rho}\Big(\frac{2t'(1+k)}{2^{256}} + \frac{1}{eM} + \frac{k}{eR_w}\Big).$$

What it rests on, all named:
* A2, expected-time collision resistance of SHA-512 for the link theorem's explicit finders of the prover bounded
  (`hCR : LinkCR … (reg σ) (cont σ) …`);
* `ValueBinding`, the binding of wires to registered commit strings, which the circuit prover's `hm96-sha512` leaf
  layout discharges;
* the lowering, `LoweringSoundB`, or in the `_placed` forms each drawn unit's place in its table (`UnitPlace`: the row
  placement of #154).

The numeric side conditions are `N₀/(ek) ≤ ρ < 1` for every table (`hr`, `hρ`) and `t'`, the SHA-512 evaluations of one
session run.
-/

namespace FlockSoundness.Audit

open Game Model Accounting Extract LinkFinder
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

/-- **The link term the link theorem gives**: `Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`. -/
noncomputable def linkBoundE (k Rw M : ℕ) (ρ t' : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ c : vb.Pos, (Fintype.card (P.Ωc plan tab vb L c) : ℝ) / Fintype.card L.Ω) *
    (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

/-- At a prover whose link finders satisfy A2, the prover-by-prover link bound is `linkBoundE`. -/
theorem linkBoundCR_eq {k Rw M : ℕ} {ρ t' : ℝ} {R : Reg} {τ : Cont L Reg (batchedSession H E plan) R}
    (hCR : P.LinkCR H E plan tab vb R τ k Rw M t') :
    P.linkBoundCR H E plan tab vb k Rw M ρ t' R τ = P.linkBoundE (L := L) plan tab vb k Rw M ρ t' := by
  unfold linkBoundCR
  simp only [hCR, ↓reduceIte]
  rfl

/-- **The compiled-layer count curve, with `δ_link` discharged**: at the plurality value layer,
`Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + linkBoundE`, given A2 for the link finders, `ValueBinding`
and the lowering. -/
theorem flock_batched_count_linked (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_countE H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE H E plan tab vb hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

/-- **Drawn units of the compiled-layer audit, with `δ_link` discharged**, are correct at the plurality value layer
up to `ε_ks(σ) + linkBoundE`. -/
theorem flock_batched_drawn_linked (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (P.flock_batched_drawnE H E plan tab decode (fun R τ => P.Xplur H E plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (P.flock_batched_linkSoundE H E plan tab vb hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [P.linkBoundCR_eq H E plan tab vb hCR])

end Partition

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- The decoder that reads each drawn unit from its place in its table (`loweringSoundB_of_place`). -/
noncomputable def placeDecoder (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) :
    DecoderB (C := C) (Reg := Reg) plan tab := fun S R u M =>
  if h : u ∈ S then (place S R u h).decode
    (witnessOf execArith (tabOf plan tab S R u).S.m (tabOf plan tab S R u).sch.k0
      (tabOf plan tab S R u).sch.level₀.logCols M)
  else fun _ => false

/-- **The compiled-layer count curve from the drawn units' places**: `flock_batched_count_linked` with the lowering
discharged by each drawn unit's place in its table (`UnitPlace`: the row placement). What remains named is A2, `ValueBinding` and the places. -/
theorem flock_batched_count_placed
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab (P.placeDecoder plan tab place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSession H E plan)) σ ≤
      L.miss K + ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  P.flock_batched_count_linked H E plan tab vb (P.loweringSoundB_of_place plan tab place) hk hRw hM hρ hr
    ht K σ hCR

/-- **Drawn units, from the drawn units' places**: `flock_batched_drawn_linked` with the lowering discharged by
`UnitPlace`. -/
theorem flock_batched_drawn_placed
    (place : ∀ S R u, u ∈ S → UnitPlace P (tabOf plan tab S R u).S u) {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab (P.placeDecoder plan tab place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSession H E plan)))
    (hCR : P.LinkCR H E plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (P.Xplur H E plan tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSession H E plan)) σ ≤
      ksAvgBE H E plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  P.flock_batched_drawn_linked H E plan tab vb (P.loweringSoundB_of_place plan tab place) hk hRw hM hρ hr
    ht σ hCR

end Partition

end FlockSoundness.Audit
