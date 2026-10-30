import Definitions.Nci.BitView
import Definitions.Nci.Dark
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions
import Definitions.Nci.Assumptions.Cover
import Definitions.Nci.Assumptions.CoverTraining

/-!
# The cover theorem on a declaration that complies with the inference-only policy

The cover theorem (`Nci.Cover.Guarantees`) bounds what a block of a circuit of bits finishes. This guarantee
joins it to the inference-only policy at `(c, k)`, `Nci.Complies c k`, through a reference training circuit
`r` that the declared circuit `d` contains, as `Nci.Guarantees.ContainedTraining` does: `d`'s wiring is the
prover's, so the premises are on `r`, and `d` reaches `r` through containment with its registered information
free of the reference circuit's fresh input (`ThetaRegistered`). The premises of the cover theorem (`UpdateHardness`,
the upstream certificate) are on `r`'s circuit bit by bit (`Circuit.ofDecl r`), where a block is a set of bits.

The route: a compliant division of `d` pulled back through the containment divides `r` into pieces, each no larger than
its inference block and reading no more fresh input. A piece `R`'s block, the bits of `R` and of what it reads, computes
every bit of `R` that reads something and is handed at most `novelReads R ≤ k` bits outside the free bits; it has at
most `w₀ c + k + width(θ)` bits when no gate of `r` is wider than `w₀`. So the blocks cover `r`'s bits, each finishes at
most `(k² + b k) / b` counted products, and summed over the `size(d) / c` inference blocks the counted products are at
most `(k² + b k) / (b c) · size(d)`.

That is `size(d) ≥ α₀ b c / (k² + b k) · size(r)`: the covering bound's per-block count at `c_w = b`, `c_a = 1` (with
the upstream certificate in place of activation-chaining), at `s = c` word gates, when the counted products are a
fraction `α₀` of `r`'s gates. The products are counted at the granularity of `r`'s gates (a Dot16 step, not a scalar
product), and `UpdateHardness` is needed at the block size in bits, `s = w₀ c + k + width(θ)`.

`CoverTrainingAligned` is the width variant when every activation region is a set of whole words at least `f` bits wide:
the blocks are whole words too, so each use of their whole products costs them `f` bits of its region, and the factor is
`α₀ b f c / (k² + b f k)`, up to `f` times `CoverTraining`'s, with no premise beyond `UpdateHardness`.
-/

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the cover theorem's factor**: a declaration `d` that contains a
reference training circuit `r` and complies at `(c, k)`, with its registered information free of the reference circuit's
fresh input (`ThetaRegistered`), has `α₀ b c size(r) ≤ (k² + b k) size(d)` when the counted products of `r`'s bit
view are a fraction `α₀` of `r`'s gates, under `UpdateHardness` at the block size `s ≥ w₀ c + k + width(θ)` (no gate of
`r` wider than `w₀`) and the upstream certificate at depth `D` with `k < b D`. -/
def CoverTraining : Prop :=
  ∀ (c k : ℕ) (b α₀ : ℝ) (r d : Verity.Declaration) (M : Verity.Containment r.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl r) J L)
    (A : J → Finset (Fin (Circuit.ofDecl r).N)) (s D w₀ : ℕ),
    0 < b → Nci.Complies c k d →
    Nci.Assumptions.ThetaRegistered r d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → Assumptions.CapBelowDepth k b D →
    Assumptions.BlockFits r c k w₀ s → Nci.Assumptions.Coverage α₀ r Mb.P.card →
      α₀ * b * c * r.D.size r.D.computed ≤ (k * k + b * k) * d.D.size d.D.computed

/-- **On whole-word activations, training through a compliant declaration pays the width factor**: `CoverTraining` with
each use's activation region the bits of a set `Aw j` of `r`'s gates (`wordBits`), each at least `f` bits wide, has
`α₀ b f c size(r) ≤ (k² + b f k) size(d)`, under `UpdateHardness` alone. A block of whole words handed a bit of a region
of whole words is handed all of that word's bits, so each use of its whole products costs it `f` bits of its region. -/
def CoverTrainingAligned : Prop :=
  ∀ (c k : ℕ) (b α₀ : ℝ) (r d : Verity.Declaration) (M : Verity.Containment r.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl r) J L)
    (Aw : J → Finset (Fin r.D.N)) (s D w₀ f : ℕ),
    0 < b → Nci.Complies c k d →
    Nci.Assumptions.ThetaRegistered r d (Nci.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream (fun j => wordBits r (Aw j)) D → Assumptions.CapBelowDepth k b D →
    Assumptions.RegionsAtLeast Mb Aw f → Assumptions.BlockFits r c k w₀ s →
    Nci.Assumptions.Coverage α₀ r Mb.P.card →
      α₀ * b * f * c * r.D.size r.D.computed ≤ (k * k + b * f * k) * d.D.size d.D.computed

end Nci.Cover.Guarantees
