import FlockSoundness.Protocol.Audit.FlockLink

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
* `ValueBinding`, the binding of wires to registered commit strings, which M0's `hm96-sha512` leaf layout discharges;
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

end Partition

end FlockSoundness.Audit
