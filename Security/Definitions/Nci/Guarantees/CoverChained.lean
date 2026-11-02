import Definitions.Nci.BitDecl
import Definitions.Nci.Dark
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions.CoveringTwoCosts

/-!
# The cover theorem on a compliant declaration, counting the covering bound's chained multiplication blocks

`CoverTraining` (`Definitions.Nci.Guarantees.CoverTraining`) joins the cover theorem to the NCI policy, and counts the
products of the cover theorem's vocabulary under its upstream certificate. `CoverTrainingChained` is the same join with
the covering bound's count (`Nci.Guarantees.CoveringBoundTwoCosts`): `|𝓜*|`, the multiplication blocks of the reference
training circuit `c` that are chained at depth `D` and activation cost `c_a` (`Nci.Assumptions.chainedBlocksAt`), under
parameter-hardness at weight cost `c_w` (`Nci.Assumptions.ParameterHardAt`). Both are read on `c` bit by bit as a
declaration (`bitDecl c`), so a block's size, what it is handed and every cut are counted in bits, the unit of the NCI
policy's novel reads.

Why bits and not `c`'s gates: on the gates, `|in(S) \ θ|` counts gates while an IU's novel reads count bits, so a block
handed `k` bits is handed up to `k / w_min` gates, and the bound would lose up to `w_max / w_min` unless every gate had
one width; and a constant gate (computed, reading nothing) is a source outside `θ`, so every block of gates holding it
is handed it, while an IU holding it computes it for free. On the bits every gate is one bit wide and a constant's bits
are free, so neither needs a hypothesis.

The route is `CoverTraining`'s. A compliant division of the declared circuit `d`, pulled back through the containment,
divides `c` into pieces. Piece `u`'s block, the bits of the piece and of what it reads, is a subcircuit of `bitDecl c`
with at most `s ≥ w₀ Γ + α Γ + width(θ)` bits that is handed `k_u ≤ α σ_u` bits outside the free bits, where `σ_u ≤ Γ`
is its IU's size; the blocks compute every bit that reads something. The covering bound's per-block count gives each
block at most `(k_u / c_w)(k_u / c_a) + k_u` chained blocks whose output bit it computes, and summed over the IUs
`c_w c_a |𝓜*| ≤ (α² Γ + c_w c_a α) size(d)`.

In the covering bound's form, with `k = α Γ`: `|𝓜*| ≤ (size(d) / Γ) · ((k / c_w)(k / c_a) + k)`, as if `d` were covered
by `size(d) / Γ` blocks; equivalently, with `α₀ = |𝓜*| / size(c)`, the slowdown `size(d) / size(c)` is at least
`α₀ Γ / ((k / c_w)(k / c_a) + k)`. At `c_w = b`, `c_a = 1` this is `CoverTraining`'s factor with `𝓜*` in place of the
certified products.
-/

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the covering bound's two-cost factor on its chained multiplication
blocks**: a declaration `d` that contains a reference training circuit `c` and complies at `(Γ, α)`, with its θ
registered (`ThetaRegistered`), has `c_w c_a |𝓜*| ≤ (α² Γ + c_w c_a α) size(d)`, where `𝓜*` is the set of
multiplication blocks of `c`'s bits chained at `(D, c_a)` (`Nci.Assumptions.chainedBlocksAt (bitDecl c)`), under
parameter-hardness at `(s, c_w)` on `c`'s bits (`ParameterHardAt (bitDecl c)`) at the block size
`s ≥ w₀ Γ + α Γ + width(θ)` (no gate of `c` wider than `w₀`), with `α Γ < c_w D` and `c_a > 0`. Equivalently its
slowdown `size(d) / size(c)` is at least `(|𝓜*| / size(c)) Γ / ((k / c_w)(k / c_a) + k)` at `k = α Γ`; in that form a
division by zero is zero. -/
def CoverTrainingChained : Prop :=
  ∀ (Γ : ℕ) (α : ℝ) (c d : Verity.Declaration) (M : Verity.Containment c.D d.D)
    (B : Nci.Covering.MulBlocks (bitDecl c)) (s D w₀ : ℕ) (cw ca : ℝ),
    0 ≤ α → Nci.Complies Γ α d →
    Nci.Assumptions.ThetaRegistered c d (Nci.PartialContainment.ofContainment M) →
    Nci.Assumptions.ParameterHardAt (bitDecl c) B s cw → α * Γ < cw * D → 0 < ca →
    (∀ g, c.D.w g ≤ w₀) → w₀ * Γ + α * Γ + c.D.width c.θ ≤ s →
      cw * ca * (Nci.Assumptions.chainedBlocksAt (bitDecl c) B D ca).card ≤
          (α * α * Γ + cw * ca * α) * d.D.size d.D.computed ∧
        ((Nci.Assumptions.chainedBlocksAt (bitDecl c) B D ca).card : ℝ) / c.D.size c.D.computed * Γ /
            (α * Γ / cw * (α * Γ / ca) + α * Γ) ≤ (d.D.size d.D.computed : ℝ) / c.D.size c.D.computed

end Nci.Cover.Guarantees
