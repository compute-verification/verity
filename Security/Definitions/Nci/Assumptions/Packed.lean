import Definitions.Nci.Assumptions
import Definitions.Nci.Packed

/-!
# NCI's activations assumption on a chain with several products to a gate

`ActivationsIncompressible`, restated for a `PackedChain`, so that it applies to a reference circuit that computes in
`Dot16` steps. At `k = 1` it is `ActivationsIncompressible` on the chain itself (`Chain.toPacked`).
-/

namespace Nci.Assumptions

open Verity

open Classical in
/-- **Incompressibility of weights and activations, at most `k` products to a gate** (`ActivationsIncompressible` on a
`PackedChain`). Every IU `R` of `D` (a set of at most `Γ` computed gates) that reads at most `K` bits of novel information
reads at least `b_w` bits for each weight entry it uses and `b_a` bits for each input entry it uses, other than an input
entry linked to an entry whole in `R`:
`b_w · #(weight entries R uses) + b_a · #(input entries R uses, not linked to an entry whole in R) ≤ width(in(R) \ θ)`.
"Uses" means that `R` holds the gate of a product of the entry, a gate that may hold up to `k` products.

On an honest `Dot16` circuit (`k = 16`) it reads as the per-product form does. An IU that holds one step gate, the 16
products `W_ℓ[i, j] x_t^{(ℓ)}[j]` of one entry at 16 consecutive `j`, uses 16 weight entries and 16 input entries, and is
charged `16 b_w + 16 b_a`; the gate reads the previous accumulator, the 16 activations and the 16 weights, which is
`16 b_w + 16 b_a` plus the accumulator's width. None of those input entries is linked to an entry whole in a single
step gate unless the step completes an entry of the multiplication before. As in `ActivationsIncompressible`, the
rationale is that `R` must read each input entry it uses, its linked output entry or a partial sum of it, and each weight
entry it uses, whichever of its gates hold them.

It is adopted, with `PackedTrainingIncompressible`, as the `k`-generalization of the per-product premises
`ActivationsIncompressible` and `TrainingIncompressible`. It
covers a reference training circuit with at most `k` of the chain's products to a gate, such as one in `Dot16` steps,
whose weights and first inputs depend on more than `Γ` work of training since the commitment, as
`ActivationsIncompressible` does, and is false on a declared circuit, whose wiring the prover controls. -/
def PackedActivationsIncompressible {k : ℕ} (D : Declaration) (c : PackedChain D.D k) (Γ : ℕ) (K b_w b_a : ℝ) : Prop :=
  ∀ R ⊆ D.D.computed, D.D.size R ≤ Γ → (novelReads D R : ℝ) ≤ K →
    b_w * (c.weightsUsed R).card +
        b_a * ((c.inputsUsed R).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card ≤ novelReads D R

end Nci.Assumptions
