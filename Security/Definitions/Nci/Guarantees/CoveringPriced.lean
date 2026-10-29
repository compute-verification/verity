import Definitions.Nci.Assumptions.CoveringPriced

/-!
# The covering bound at a price per weight

`CoveringBoundTwoCosts` (`Definitions.Nci.Guarantees.CoveringTwoCosts`) with each updated weight at its own price `p i` in
place of one weight cost `c_w`, beside it. The activations keep one cost `c_a`. At `p = fun _ => c_w` it gives
`CoveringBoundTwoCosts`.
-/

namespace Nci.Guarantees

open Verity Nci.Covering

/-- **The covering bound at a price per weight**: if `C` is `(s, p)`-parameter-hard
(`Assumptions.ParameterHardPricedAt`), every price is nonnegative, any `D` weights cost more than `k` together, and
`c_a > 0`, then every `(s, k)`-covering `𝓘` of `C` by `n` blocks splits `𝓜*`, the `(D, c_a)`-chained multiplication
blocks (`Assumptions.chainedBlocksAt`), into a set `U` of at most `n k` blocks and the rest, whose weights' prices add up
to at most `n · k · (k / c_a)`. And at every price ceiling `t` (`p i ≤ t` for all `i`), its slowdown is
`ρ(𝓘) = s n / |C| ≥ (∑_{b ∈ 𝓜*} p_b / |C|) · s / (k · (k / c_a) + t k)`, where `p_b` is the price of `b`'s weight.

Each chained block is charged to a block `S` that computes its output gate. `U` is the chained blocks that no such `S`
completes: each has one of its gates among the inputs of `S` outside `θ`, and the multiplication blocks are disjoint, so
at most `k` are charged to `S`. A block that `S` completes has its weight in `θ'_S`, whose prices add up to at most `k`
by parameter-hardness, so `θ'_S` has fewer than `D` weights; and its token in a set of at most `k / c_a` tokens, since
`in(S) ∩ T_j` cuts the activation off from the weights outside `θ'_S` and so has at least `c_a` gates. The completed
blocks charged to `S` then cost at most `(∑_{i ∈ θ'_S} p i) · (k / c_a) ≤ k · (k / c_a)`.

`D` has the role it has in `CoveringBoundTwoCosts`, where `c_w D > k` gives it: the weights a block completes number
fewer than `D`. Chaining needs no price of its own, since its `D` counts weights. `k < p i · D` for every `i` gives both
conditions on the prices; the weaker form also lets a weight the prover can rebuild be priced at zero, as long as fewer
than `D` weights are that cheap. `c_a > 0` is needed as in `CoveringBoundTwoCosts`. `t` charges each block of `U` at most
`t`. In practice `t` is at most a weight's width, 16 bits at bf16, and `t k` is under `10⁻⁴` of `k · (k / c_a)` at
`k = 1` MB and `c_a = 16`. At `p = fun _ => c_w` and `t = c_w` the slowdown form is `CoveringBoundTwoCosts`'s, and `U`
and the rest give its count.

The counts are stated multiplied out, with no division, so they say something when `k = 0`; in the slowdown form a
division by zero is zero. -/
def CoveringBoundPriced : Prop :=
  ∀ (d : Declaration) (B : MulBlocks d) (s k D : ℕ) (p : Fin B.nw → ℝ) (ca : ℝ) (n : ℕ) (I : Fin n → Subcircuit d.D),
    (∀ i, 0 ≤ p i) → (∀ W : Finset (Fin B.nw), D ≤ W.card → (k : ℝ) < ∑ i ∈ W, p i) → 0 < ca →
    Assumptions.ParameterHardPricedAt d B s p → Covers d s k I →
      (∃ U ⊆ Assumptions.chainedBlocksAt d B D ca, (U.card : ℝ) ≤ n * k ∧
        ∑ b ∈ Assumptions.chainedBlocksAt d B D ca \ U, p b.1 ≤ n * (k * (k / ca))) ∧
      ∀ t : ℝ, (∀ i, p i ≤ t) →
        (∑ b ∈ Assumptions.chainedBlocksAt d B D ca, p b.1) / d.D.N * s / (k * (k / ca) + t * k) ≤ s * n / d.D.N

end Nci.Guarantees
