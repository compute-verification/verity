module

public import Definitions.Nci.BitView
public import Definitions.Nci.Dark
public import Definitions.Nci.Assumptions.Baseline
public import Definitions.Nci.Assumptions
public import Definitions.Nci.Assumptions.Cover
public import Definitions.Nci.Assumptions.CoverTraining

/-!
# The cover theorem on a declaration that complies with the inference-only policy

The cover theorem (`Nci.Cover.Guarantees`) bounds what a block of a circuit of bits finishes. This guarantee
joins it to the inference-only policy at `(c, k)`, `Nci.Complies c k`, through a reference training circuit
`r` that the declared circuit `d` contains: `d`'s wiring is the prover's, so the premises are on `r`, and `d` reaches
`r` through containment with its registered information free of the reference circuit's fresh input
(`ThetaRegistered`). The premises of the cover theorem (`UpdateHardness`, the upstream certificate) are on `r`'s circuit
bit by bit (`Circuit.ofDecl r`), where a block is a set of bits.

The route: a compliant division of `d` pulled back through the containment divides `r` into pieces, each no larger than
its inference block and reading no more fresh input. A piece `R`'s block, the bits of `R` and of what it reads, computes
every bit of `R` that reads something and is handed at most `novelReads R ≤ α σ` bits outside the free bits, where
`σ ≤ Γ` is its inference block's size; it has at most `w₀ Γ + α Γ + width(θ)` bits when no gate of `r` is wider than
`w₀`. So the blocks cover `r`'s bits, each finishes at most `(k² + b k) / b` counted products at its own `k ≤ α σ`, and
summed over the inference blocks the counted products are at most `(α² Γ + b α) / b · size(d)`.

With `k = α Γ` this is `size(d) ≥ α₀ b Γ / (k² + b k) · size(r)`: the covering bound's per-block count at `c_w = b`,
`c_a = 1` (with the upstream certificate in place of activation-chaining), at `s = Γ` word gates, when the counted
products are a fraction `α₀` of `r`'s gates. The products are counted at the granularity of `r`'s gates (a Dot16 step,
not a scalar product), and `UpdateHardness` is needed at the block size in bits, `s = w₀ Γ + α Γ + width(θ)`.
-/

@[expose] public section

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the cover theorem's factor**: a declaration `d` that contains a
reference training circuit `r` and complies at `(Γ, α)` (`CompliesRate`), with its registered information free of the
reference circuit's fresh input (`ThetaRegistered`), has `α₀ b size(r) ≤ (α² Γ + b α) size(d)` when the counted
products of `r`'s bit view are a fraction `α₀` of `r`'s gates, under `UpdateHardness` at the block size
`s ≥ w₀ Γ + α Γ + width(θ)` (no gate of `r` wider than `w₀`) and the upstream certificate at depth `D` with
`α Γ < b D`. -/
def CoverTraining : Prop :=
  ∀ (Γ : ℕ) (α b α₀ : ℝ) (r d : Verity.RegisteredCircuit) (M : Verity.Containment r.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl r) J L)
    (A : J → Finset (Fin (Circuit.ofDecl r).N)) (s D w₀ : ℕ),
    0 < b → 0 ≤ α → Nci.CompliesRate Γ α d →
    Nci.Assumptions.ThetaRegistered r d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → Assumptions.CapBelowDepth (α * Γ) b D →
    Assumptions.BlockFits r Γ (α * Γ) w₀ s → Nci.Assumptions.Coverage α₀ r Mb.P.card →
      α₀ * b * r.D.size r.D.computed ≤ (α * α * Γ + b * α) * d.D.size d.D.computed

end Nci.Cover.Guarantees
