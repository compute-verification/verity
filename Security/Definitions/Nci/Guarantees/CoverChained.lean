import Definitions.Nci.BitDecl
import Definitions.Nci.Dark
import Definitions.Nci.Assumptions.Baseline
import Definitions.Nci.Assumptions.CoveringTwoCosts
import Definitions.Nci.Assumptions.CoverTraining

/-!
# The cover theorem on a compliant declaration, counting the covering bound's chained multiplication blocks

`CoverTraining` (`Definitions.Nci.Guarantees.CoverTraining`) joins the cover theorem to the inference-only
policy, and counts the products of the cover theorem's vocabulary under its upstream certificate.
`CoverTrainingChained` is the same join with the covering bound's count
(`Nci.Guarantees.CoveringBoundTwoCosts`): `|𝓜*|`, the multiplication blocks of the reference training circuit
`r` that are chained at depth `D` and activation cost `c_a` (`Nci.Assumptions.chainedBlocksAt`), under
parameter-hardness at weight cost `c_w` (`Nci.Assumptions.ParameterHardAt`). Both are read on `r` bit by bit
as a declaration (`bitDecl r`), so a block's size, what it is handed and every cut are counted in bits, the unit of the
policy's fresh input reads.

Why bits and not `r`'s gates: on the gates, `|in(S) \ θ|` counts gates while an inference block's fresh input reads
count bits, so a block handed `k` bits is handed up to `k / w_min` gates, and the bound would lose up to `w_max / w_min`
unless every gate had one width; and a constant gate (computed, reading nothing) is a source outside `θ`, so every block
of gates holding it is handed it, while an inference block holding it computes it for free. On the bits every gate is
one bit wide and a constant's bits are free, so neither needs a hypothesis.

The route is `CoverTraining`'s. A compliant division of the declared circuit `d` at `(c, k)`, pulled back through the
containment, divides `r` into pieces. Piece `u`'s block, the bits of the piece and of what it reads, is a subcircuit of
`bitDecl r` with at most `s ≥ w₀ c + k + width(θ)` bits that is handed at most `k` bits outside the free bits; the
blocks compute every bit that reads something. The covering bound's per-block count gives each block at most
`(k / c_w)(k / c_a) + k` chained blocks whose output bit it computes, and summed over the `size(d) / c` inference blocks
`c_w c_a c |𝓜*| ≤ (k² + c_w c_a k) size(d)`.

In the covering bound's form: `|𝓜*| ≤ (size(d) / c) · ((k / c_w)(k / c_a) + k)`, as if `d` were covered by
`size(d) / c` blocks; equivalently, with `α₀ = |𝓜*| / size(r)`, the slowdown `size(d) / size(r)` is at least
`α₀ c / ((k / c_w)(k / c_a) + k)`. At `c_w = b`, `c_a = 1` this is `CoverTraining`'s factor with `𝓜*` in place of the
certified products.
-/

namespace Nci.Cover.Guarantees

/-- **Training through a compliant declaration pays the covering bound's two-cost factor on its chained multiplication
blocks**: a declaration `d` that contains a reference training circuit `r` and complies at `(c, k)`, with its registered
information free of the reference circuit's fresh input (`ThetaRegistered`), has
`c_w c_a c |𝓜*| ≤ (k² + c_w c_a k) size(d)`, where `𝓜*` is the set of multiplication blocks of `r`'s bits chained at
`(D, c_a)` (`Nci.Assumptions.chainedBlocksAt (bitDecl r)`), under parameter-hardness at `(s, c_w)` on `r`'s
bits (`ParameterHardAt (bitDecl r)`) at the block size `s ≥ w₀ c + k + width(θ)` (no gate of `r` wider than `w₀`), with
`k < c_w D` and `c_a > 0`. Equivalently its slowdown `size(d) / size(r)` is at least
`(|𝓜*| / size(r)) c / ((k / c_w)(k / c_a) + k)`; in that form a division by zero is zero. -/
def CoverTrainingChained : Prop :=
  ∀ (c k : ℕ) (r d : Verity.RegisteredCircuit) (M : Verity.Containment r.D d.D)
    (B : Nci.Covering.MulBlocks (bitDecl r)) (s D w₀ : ℕ) (cw ca : ℝ),
    Nci.Complies c k d →
    Nci.Assumptions.ThetaRegistered r d (Nci.PartialContainment.ofContainment M) →
    Nci.Assumptions.ParameterHardAt (bitDecl r) B s cw → Assumptions.CapBelowDepth k cw D → 0 < ca →
    Assumptions.BlockFits r c k w₀ s →
      cw * ca * c * (Nci.Assumptions.chainedBlocksAt (bitDecl r) B D ca).card ≤
          (k * k + cw * ca * k) * d.D.size d.D.computed ∧
        ((Nci.Assumptions.chainedBlocksAt (bitDecl r) B D ca).card : ℝ) / r.D.size r.D.computed * c /
            (k / cw * (k / ca) + k) ≤ (d.D.size d.D.computed : ℝ) / r.D.size r.D.computed

end Nci.Cover.Guarantees
