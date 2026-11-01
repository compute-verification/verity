import Definitions.Nci.Assumptions.Packed
import Definitions.Nci.Dark

/-!
# The inference-only policy's fan-out bound on a chain with several products to a gate

`ContainedFanOutOn` for a `PackedChain`, so that it says something about a declaration that contains a reference circuit
computing in `Dot16` steps (`k = 16`). It counts products, not the gates that hold them, and its bound does not read `k`.
At `k = 1` it gives `ContainedFanOutOn` (`Nci.SecurityProofs.containedFanOutOn_of_packed`).
-/

namespace Nci.Guarantees

open Verity

/-- **Products done in the declaration are bounded by their fan-out, at most `k` products to a gate**
(`ContainedFanOutOn` on a `PackedChain`): a declaration `d` that complies at `(c, k)` and contains `r` partly computes
the chain's products whose gates are in `S`, `G_S` of them, with `b_a / Λ · G_S ≤ (k / c) size(d)`,
`Λ = max e_max (K / (4 b_w dd))`, whatever `k`. Each inference block makes at most `Λ` products per `b_a` bits it reads:
`e_max` per input entry it uses at a position it completes no entry of, one for each row of the multiplication,
whichever gates hold them, and, at positions it completes entries of, `K / (4 b_w dd)` per `b_a` bits by AM–GM on its
weight and input reads. It needs only `PackedActivationsIncompressible` on `r` at read cap `K ≥ k`, besides the
registered-information premise `ThetaRegistered`. -/
def PackedContainedFanOutOn : Prop :=
  ∀ (p c k : ℕ) (K b_w b_a : ℝ) (r d : RegisteredCircuit) (ch : PackedChain r.D p) (dd emax : ℕ)
    (M : PartialContainment r d),
    0 < c → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → (k : ℝ) ≤ K →
    (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → Complies c k d → Assumptions.ThetaRegistered r d M →
    Assumptions.PackedActivationsIncompressible r ch c K b_w b_a →
      b_a / max (emax : ℝ) (K / (4 * b_w * dd)) *
        (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
      (k / c : ℝ) * d.D.size d.D.computed

end Nci.Guarantees
