import Definitions.Nci.BitView
import Definitions.Nci.Dark
import Models.Nci.Assumptions.Baseline
import Models.Nci.Assumptions.Cover

/-!
# The cover theorem on a declaration that complies with the NCI policy

The cover theorem (`Nci.Cover.Guarantees`) bounds what a block of a circuit of bits finishes. This guarantee joins it to
the NCI policy, `Nci.Complies Γ α`, through a reference training circuit `c` that the declared circuit `d` contains, as
`Nci.Guarantees.ContainedTraining` does: `d`'s wiring is the prover's, so the premises are on `c`, and `d` reaches `c`
through containment with its baseline registered (`BaselineRegistered`). The premises of the cover theorem
(`UpdateHardness`, the upstream certificate) are on `c`'s circuit bit by bit (`Circuit.ofDecl c`), where a block is a set
of bits.

The route: a compliant division of `d` pulled back through the containment divides `c` into pieces, each no larger than
its IU and reading no more novel information. A piece `R`'s block, the bits of `R` and of what it reads, computes every
bit of `R` that reads something and is handed at most `novelReads R ≤ α σ` bits outside the free bits, where `σ ≤ Γ` is
its IU's size; it has at most `w₀ Γ + α Γ + width(θ)` bits when no gate of `c` is wider than `w₀`. So the blocks cover
`c`'s bits, each finishes at most `(k² + b k) / b` counted products at its own `k ≤ α σ`, and summed over the IUs the
counted products are at most `(α² Γ + b α) / b · size(d)`.

With `k = α Γ` this is `size(d) ≥ α₀ b Γ / (k² + b k) · size(c)`: the cover theorem's area factor at `s = Γ` word gates,
when the counted products are a fraction `α₀` of `c`'s gates. The products are counted at the granularity of `c`'s gates
(a Dot16 step, not a scalar product), and `UpdateHardness` is needed at the block size in bits,
`s = w₀ Γ + α Γ + width(θ)`.
-/

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the cover theorem's factor**: a declaration `d` that contains a
reference training circuit `c` and complies at `(Γ, α)`, with its baseline registered (`BaselineRegistered`), has
`α₀ b size(c) ≤ (α² Γ + b α) size(d)` when the counted products of `c`'s bit view are a fraction `α₀` of `c`'s gates,
under `UpdateHardness` at the block size `s ≥ w₀ Γ + α Γ + width(θ)` (no gate of `c` wider than `w₀`) and the upstream
certificate at depth `D` with `α Γ < b D`. -/
def CoverTraining : Prop :=
  ∀ (Γ : ℕ) (α b α₀ : ℝ) (c d : Verity.Declaration) (M : Verity.Containment c.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl c) J L)
    (A : J → Finset (Fin (Circuit.ofDecl c).N)) (s D w₀ : ℕ),
    0 < b → 0 ≤ α → Nci.Complies Γ α d →
    Nci.Assumptions.BaselineRegistered c d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → α * Γ < b * D →
    (∀ g, c.D.w g ≤ w₀) → w₀ * Γ + α * Γ + c.D.width c.θ ≤ s →
    α₀ * c.D.size c.D.computed ≤ Mb.P.card →
      α₀ * b * c.D.size c.D.computed ≤ (α * α * Γ + b * α) * d.D.size d.D.computed

end Nci.Cover.Guarantees
