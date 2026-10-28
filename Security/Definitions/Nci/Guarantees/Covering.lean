import Definitions.Nci.Assumptions.Covering

/-!
# The covering bound at one width `b` and one compression rate `r`

A lower bound on inference blocks: a circuit that computes many chained multiplications of updated
weights cannot be covered by few small blocks that each read little. It is a statement beside `Complies`, `Chain` and
`F`, not a specialization of them: a covering's blocks may overlap and recompute, so it is not a division into IUs, and
its multiplication blocks are any disjoint gate sets with an output gate, not a chain's products and sums.
-/

namespace Nci.Guarantees

open Verity Nci.Covering

/-- **The covering bound** ([COVERING-BOUND]): if `C` is `(s, r, b)`-parameter-hard (`Assumptions.ParameterHard`) and
`c D > k` with `c = r · b`, every `(s, k)`-covering `𝓘` of `C` by `n` blocks has `|𝓜*| ≤ n · ((k / c)² + k)`, where `𝓜*`
is the set of `(D, r, b)`-chained multiplication blocks (`Assumptions.chainedBlocks`); equivalently its slowdown is
`ρ(𝓘) = s n / |C| ≥ (|𝓜*| / |C|) · s / ((k / c)² + k)`.

Each chained block is charged to a block `S` computing its output gate. At most `k` of those charged to `S` are not
completed by it, since each puts a gate of its own in `in(S) \ θ`; the completed ones have their weight in `θ'_S`,
`|θ'_S| ≤ k / c` by parameter-hardness, and their token in a set of at most `k / c` tokens, since `in(S) ∩ T_j` cuts the
activation off for each of them.

In a transformer `k ≈ 10⁷` bits and `c ≈ 16`, so `D ≈ 6 · 10⁵` suffices, and all but layer 1's Q/K/V products (about
0.1% of a 70B model's) are chained. The count is stated multiplied out, with no division by `(k / c)² + k`, so it says
something when `k = 0`; in the slowdown form a division by zero is zero. -/
def CoveringBound : Prop :=
  ∀ (d : Registered) (B : MulBlocks d) (s k D : ℕ) (r b : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
    (k : ℝ) < r * b * D → Assumptions.ParameterHard d B s r b → Covers d s k I →
      ((Assumptions.chainedBlocks d B D r b).card : ℝ) ≤ n * ((k / (r * b)) ^ 2 + k) ∧
      ((Assumptions.chainedBlocks d B D r b).card : ℝ) / d.D.N * s / ((k / (r * b)) ^ 2 + k) ≤ s * n / d.D.N

end Nci.Guarantees
