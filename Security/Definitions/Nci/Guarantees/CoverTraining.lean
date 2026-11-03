import Definitions.Nci.BitView
import Definitions.Nci.Dark
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions.Cover

/-!
# The cover theorem on a declaration that complies with the NCI policy

The cover theorem (`Nci.Cover.Guarantees`) bounds what a block of a circuit of bits finishes. This guarantee joins it to
the NCI policy, `Nci.Complies Γ α`, through a reference training circuit `c` that the declared circuit `d` contains, as
`Nci.Guarantees.ContainedTraining` does: `d`'s wiring is the prover's, so the premises are on `c`, and `d` reaches `c`
through containment with its θ registered (`ThetaRegistered`). The premises of the cover theorem
(`UpdateHardness`, the upstream certificate) are on `c`'s circuit bit by bit (`Circuit.ofDecl c`), where a block is a set
of bits.

The route: a compliant division of `d` pulled back through the containment divides `c` into pieces, each no larger than
its IU and reading no more novel information. A piece `R`'s block, the bits of `R` and of what it reads, computes every
bit of `R` that reads something and is handed at most `novelReads R ≤ α σ` bits outside the free bits, where `σ ≤ Γ` is
its IU's size; it has at most `w₀ Γ + α Γ + width(θ)` bits when no gate of `c` is wider than `w₀`. So the blocks cover
`c`'s bits, each finishes at most `(k² + b k) / b` counted products at its own `k ≤ α σ`, and summed over the IUs the
counted products are at most `(α² Γ + b α) / b · size(d)`.

With `k = α Γ` this is `size(d) ≥ α₀ b Γ / (k² + b k) · size(c)`: the covering bound's per-block count at
`c_w = b`, `c_a = 1` (with the upstream certificate in place of activation-chaining), at `s = Γ` word gates, when the
counted products are a fraction `α₀` of `c`'s gates. The products are counted at the granularity of `c`'s gates
(a Dot16 step, not a scalar product), and `UpdateHardness` is needed at the block size in bits,
`s = w₀ Γ + α Γ + width(θ)`.

`CoverTrainingAligned` is the width variant when every activation region is a set of whole words at least `f` bits wide:
the blocks are whole words too, so each use of their whole products costs them `f` bits of its region, and the factor is
`α₀ b f Γ / (k² + b f k)`, up to `f` times `CoverTraining`'s, with no premise beyond `UpdateHardness`.
-/

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the cover theorem's factor**: a declaration `d` that contains a
reference training circuit `c` and complies at `(Γ, α)`, with its θ registered (`ThetaRegistered`), has
`α₀ b size(c) ≤ (α² Γ + b α) size(d)` when the counted products of `c`'s bit view are a fraction `α₀` of `c`'s gates,
under `UpdateHardness` at the block size `s ≥ w₀ Γ + α Γ + width(θ)` (no gate of `c` wider than `w₀`) and the upstream
certificate at depth `D` with `α Γ < b D`. -/
def CoverTraining : Prop :=
  ∀ (Γ : ℕ) (α b α₀ : ℝ) (c d : Verity.Declaration) (M : Verity.Containment c.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl c) J L)
    (A : J → Finset (Fin (Circuit.ofDecl c).N)) (s D w₀ : ℕ),
    0 < b → 0 ≤ α → Nci.Complies Γ α d →
    Nci.Assumptions.ThetaRegistered c d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → α * Γ < b * D →
    (∀ g, c.D.w g ≤ w₀) → w₀ * Γ + α * Γ + c.D.width c.θ ≤ s →
    α₀ * c.D.size c.D.computed ≤ Mb.P.card →
      α₀ * b * c.D.size c.D.computed ≤ (α * α * Γ + b * α) * d.D.size d.D.computed

/-- **On whole-word activations, training through a compliant declaration pays the width factor**: `CoverTraining` with
each use's activation region the bits of a set `Aw j` of `c`'s gates (`wordBits`), each at least `f` bits wide, has
`α₀ b f size(c) ≤ (α² Γ + b f α) size(d)`, under `UpdateHardness` alone. A block of whole words handed a bit of a region
of whole words is handed all of that word's bits, so each use of its whole products costs it `f` bits of its region. -/
def CoverTrainingAligned : Prop :=
  ∀ (Γ : ℕ) (α b α₀ : ℝ) (c d : Verity.Declaration) (M : Verity.Containment c.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl c) J L)
    (Aw : J → Finset (Fin c.D.N)) (s D w₀ f : ℕ),
    0 < b → 0 ≤ α → Nci.Complies Γ α d →
    Nci.Assumptions.ThetaRegistered c d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream (fun j => wordBits c (Aw j)) D → α * Γ < b * D →
    (∀ j ∈ Mb.uses, ∀ a ∈ Aw j, f ≤ c.D.w a) →
    (∀ g, c.D.w g ≤ w₀) → w₀ * Γ + α * Γ + c.D.width c.θ ≤ s →
    α₀ * c.D.size c.D.computed ≤ Mb.P.card →
      α₀ * b * f * c.D.size c.D.computed ≤ (α * α * Γ + b * f * α) * d.D.size d.D.computed

end Nci.Cover.Guarantees
