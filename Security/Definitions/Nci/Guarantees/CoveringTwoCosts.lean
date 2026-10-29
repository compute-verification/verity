import Definitions.Nci.Assumptions.CoveringTwoCosts

/-!
# The covering bound at two costs: `c_w` a weight, `c_a` an activation

`CoveringBound` (`Definitions.Nci.Guarantees.Covering`) with the weights' cost and the activations' cost apart: the
parameter-hardness `c_w` bounds the weights a block completes, the activation-chaining `c_a` bounds its tokens. It is a
statement beside `CoveringBound`, which is it at `c_w = c_a = r · b`.
-/

namespace Nci.Guarantees

open Verity Nci.Covering

/-- **The covering bound at two costs**: if `C` is `(s, c_w)`-parameter-hard (`Assumptions.ParameterHardAt`),
`c_w D > k` and `c_a > 0`, every `(s, k)`-covering `𝓘` of `C` by `n` blocks has `|𝓜*| ≤ n · ((k / c_w)(k / c_a) + k)`,
where `𝓜*` is the set of `(D, c_a)`-chained multiplication blocks (`Assumptions.chainedBlocksAt`); equivalently its
slowdown is `ρ(𝓘) = s n / |C| ≥ (|𝓜*| / |C|) · s / ((k / c_w)(k / c_a) + k)`.

Each chained block is charged to a block `S` computing its output gate. At most `k` of those charged to `S` are not
completed by it. The completed ones have their weight in `θ'_S`, `|θ'_S| ≤ k / c_w` by parameter-hardness, so fewer than
`D`; and their token in a set of at most `k / c_a` tokens, since `in(S) ∩ T_j` cuts the activation off from the weights
outside `θ'_S` and so has at least `c_a` gates. `c_w D > k` makes `c_w` positive; `c_a > 0` is needed, since at
`c_a ≤ 0` every cut is wide enough and the bound's `k / c_a` is not a count.

At fine-tuning's measured costs, one shared `c` is capped by the weights' entropy and gives `ρ` about 1.1; the two
costs give about 45 (`Definitions.Nci.Assumptions.CoveringTwoCosts`). The count is stated multiplied out, with no division by
`(k / c_w)(k / c_a) + k`, so it says something when `k = 0`; in the slowdown form a division by zero is zero. -/
def CoveringBoundTwoCosts : Prop :=
  ∀ (d : Declaration) (B : MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
    (k : ℝ) < cw * D → 0 < ca → Assumptions.ParameterHardAt d B s cw → Covers d s k I →
      ((Assumptions.chainedBlocksAt d B D ca).card : ℝ) ≤ n * (k / cw * (k / ca) + k) ∧
      ((Assumptions.chainedBlocksAt d B D ca).card : ℝ) / d.D.N * s / (k / cw * (k / ca) + k) ≤ s * n / d.D.N

end Nci.Guarantees
