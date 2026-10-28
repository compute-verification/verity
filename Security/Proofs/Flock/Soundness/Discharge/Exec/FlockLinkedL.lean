import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
import Proofs.Flock.Soundness.Audit.FlockLinked

/-!
# The compiled-layer audit with `δ_link` discharged

The batched Flock audit with the expected-time extractor (`flock_batched_countEL`), composed with the link theorem
(`flock_batched_linkSoundEL`) at the plurality value layer (`XplurL`). The count curve and the drawn-unit bound no longer
carry `δ_link` as a hypothesis:
$$\Pr[\text{accept} \wedge \text{at least } K \text{ wrong}] \le \mathrm{miss}_L(K) + \varepsilon_{ks}(\sigma) +
\frac{Q_s}{1-\rho}\Big(\frac{2t'(1+k)}{2^{256}} + \frac{1}{eM} + \frac{k}{eR_w}\Big).$$

What it rests on, all named:
* A2, expected-time collision resistance of SHA-512 for the link theorem's explicit finders of the prover bounded
  (`hCR : LinkCRL … (reg σ) (cont σ) …`);
* `ValueBinding`, the binding of wires to registered commit strings, which the circuit prover's `hm96-sha512` leaf
  layout discharges;
* the lowering, `LoweringSoundB`, or in the `_placed` forms each drawn unit's place in its table (`UnitPlace`: the row
  placement).

The numeric side conditions are `N₀/(ek) ≤ ρ < 1` for every table (`hr`, `hρ`) and `t'`, the SHA-512 evaluations of one
session run.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type} {Hc : List UInt8 → Dh}
  (vb : P.ValueBinding plan tab decode Hc)

/-- At a prover whose link finders satisfy A2, the prover-by-prover link bound is `linkBoundE`. -/
theorem linkBoundCR_eqL {k Rw M : ℕ} {ρ t' : ℝ} {R : Reg} {τ : Cont L Reg (batchedSessionL H E Lf plan) R}
    (hCR : LinkCRL P H E Lf plan tab vb R τ k Rw M t') :
    linkBoundCRL P H E Lf plan tab vb k Rw M ρ t' R τ = P.linkBoundE (L := L) plan tab vb k Rw M ρ t' := by
  unfold linkBoundCRL
  simp only [hCR, ↓reduceIte]
  rfl

/-- **The compiled-layer count curve, with `δ_link` discharged**: at the plurality value layer,
`Pr[accept ∧ at least K wrong units] ≤ miss K + ε_ks(σ) + linkBoundE`, given A2 for the link finders, `ValueBinding`
and the lowering. -/
theorem flock_batched_count_linkedL (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_countEL P H E Lf plan tab decode (fun R τ => XplurL P H E Lf plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundEL P H E Lf plan tab vb hlow hk hRw hM hρ hr ht) K σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

/-- **Drawn units of the compiled-layer audit, with `δ_link` discharged**, are correct at the plurality value layer
up to `ε_ks(σ) + linkBoundE`. -/
theorem flock_batched_drawn_linkedL (hlow : P.LoweringSoundB plan tab decode)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  (flock_batched_drawnEL P H E Lf plan tab decode (fun R τ => XplurL P H E Lf plan tab vb R τ k Rw) hk
    (fun S R j => (hr S R j).trans_lt hρ) hlow
    (flock_batched_linkSoundEL P H E Lf plan tab vb hlow hk hRw hM hρ hr ht) σ).trans_eq
    (by rw [linkBoundCR_eqL P H E Lf plan tab vb hCR])

variable {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : Law n}

/-- **The compiled-layer count curve from the drawn units' places**: `flock_batched_count_linkedL` with the lowering
discharged by each drawn unit's place in its table (`UnitPlace`: the row placement). What remains named is A2, `ValueBinding` and the places. -/
theorem flock_batched_count_placedL
    (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u) {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab (P.placeDecoder plan tab place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (K : ℕ) (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ K ≤ (P.wrong (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw)).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  flock_batched_count_linkedL P H E Lf plan tab vb (P.loweringSoundB_of_place plan tab place) hk hRw hM hρ hr
    ht K σ hCR

/-- **Drawn units, from the drawn units' places**: `flock_batched_drawn_linkedL` with the lowering discharged by
`UnitPlace`. -/
theorem flock_batched_drawn_placedL
    (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u) {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : P.ValueBinding plan tab (P.placeDecoder plan tab place) Hc)
    {k : ℕ} (hk : 1 ≤ k) {Rw M : ℕ} (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateB (plan S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t')
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan)))
    (hCR : LinkCRL P H E Lf plan tab vb (reg σ) (cont σ) k Rw M t') :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.wrong (XplurL P H E Lf plan tab vb (reg σ) (cont σ) k Rw)) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBEL H E Lf plan k (reg σ) (cont σ) + P.linkBoundE (L := L) plan tab vb k Rw M ρ t' :=
  flock_batched_drawn_linkedL P H E Lf plan tab vb (P.loweringSoundB_of_place plan tab place) hk hRw hM hρ hr
    ht σ hCR

end FlockSoundness.Discharge.Exec
