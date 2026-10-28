import Definitions.Nci.Assumptions.Packed
import Definitions.Nci.Dark

/-!
# NCI's fan-out bound on a chain with several products to a gate

`ContainedFanOutOn` for a `PackedChain`, so that it says something about a declaration that contains a reference circuit
computing in `Dot16` steps (`k = 16`). It counts products, not the gates that hold them, and its bound does not read `k`.
At `k = 1` it gives `ContainedFanOutOn` (`Nci.SecurityProofs.containedFanOutOn_of_packed`).
-/

namespace Nci.Guarantees

open Verity

/-- **Products done in the declaration are bounded by their fan-out, at most `k` products to a gate**
(`ContainedFanOutOn` on a `PackedChain`): a declaration `d` that complies at `(Γ, α)` and contains `c` partly computes the
chain's products whose gates are in `S`, `G_S` of them, with `b_a / Λ · G_S ≤ α size(d)`,
`Λ = max e_max (K / (4 b_w dd))`, whatever `k`. Each IU makes at most `Λ` products per `b_a` bits it reads: `e_max` per
input entry it uses at a position it completes no entry of, one for each row of the multiplication, whichever gates hold
them, and, at positions it completes entries of, `K / (4 b_w dd)` per `b_a` bits by AM–GM on its weight and input reads.
It needs only `PackedActivationsIncompressible` on `c` at read cap `K ≥ αΓ`, besides the θ premise
`ThetaRegistered`. -/
def PackedContainedFanOutOn : Prop :=
  ∀ (k Γ : ℕ) (α K b_w b_a : ℝ) (c d : Registered) (ch : PackedChain c.D k) (dd emax : ℕ)
    (M : PartialContainment c d),
    0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → α * Γ ≤ K →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies Γ α d → Assumptions.ThetaRegistered c d M →
    Assumptions.PackedActivationsIncompressible c ch Γ K b_w b_a →
      b_a / max (emax : ℝ) (K / (4 * b_w * dd)) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      α * d.D.size d.D.computed

end Nci.Guarantees
