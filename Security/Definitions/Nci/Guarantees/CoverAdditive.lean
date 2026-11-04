import Definitions.Nci.Assumptions.CoverAdditive

/-!
# The additive covering bound

The covering bound at two costs (`Nci.Guarantees.CoveringBoundTwoCosts`) with the weights' bits paid outside the
token computations (`Assumptions.ConditionalParameterHardAt`): a block's two charges, `c_w` per completed weight and
`c_a` per token, then come from disjoint inputs and add up to at most `k`, so AM–GM bounds its completed multiplications
by a quarter of `(k / c_w)(k / c_a)`. Activation-chaining at cost `c_a` charges a block `c_a` inputs per token: an
activation `c_a` bits wide is such a cut.
-/

namespace Nci.Cover.Guarantees

open Verity

/-- **The additive covering bound**: if `C` is conditionally `(s, c_w)`-parameter-hard
(`Assumptions.ConditionalParameterHardAt`), `c_w D > k` and `c_a > 0`, every `(s, k)`-covering `𝓘` of `C` by `n` blocks
has `|𝓜*| ≤ n · ((k / c_w)(k / c_a) / 4 + k)`, where `𝓜*` is the set of `(D, c_a)`-chained multiplication blocks
(`Nci.Assumptions.chainedBlocksAt`); equivalently its slowdown is
`ρ(𝓘) = s n / |C| ≥ (|𝓜*| / |C|) · s / ((k / c_w)(k / c_a) / 4 + k)`.

Each chained block is charged to a block `S` computing its output gate, and at most `k` of those charged to `S` are not
completed by it, as in `CoveringBoundTwoCosts`. The completed ones lie in `θ'_S × J_S`. Each token `j ∈ J_S` costs `S`
at least `c_a` inputs inside `T_j`, which are disjoint, and each weight of `θ'_S` costs it `c_w` bits outside them, so
`c_a |J_S| + c_w |θ'_S| ≤ k`, and `4 c_w c_a |θ'_S| |J_S| ≤ k²`. That needs no bound on `|J_S|` from the cut alone,
which is where `CoveringBoundTwoCosts` spends a second `k`.

The factor is four times `CoveringBoundTwoCosts`'s for `k ≫ c_w c_a`, under a stronger premise. Kimi K2 (docstring
arithmetic, not Lean), at `k = 2·10⁷` bits, `s = 4.52·10¹⁶`, `α₀ = |𝓜*| / |C| = 0.147`, `c_w = 2.5` and `c_a = 4` (each
counted product's output reading all four bits of its FP4 activation): `α₀ s c_w c_a / (k² / 4)` is about 660, against
`CoveringBoundTwoCosts`'s 166. Three corrections apply: `s/k` is fixed by serving, so the factor falls as `1/k`; `s` and
`α₀` are counted in scalar operations, and in Boolean gates the factor is multiplied by serving's gates per multiply-add
over training's (`1` for equal formats, `0.42` for FP8 serving with BF16 training, `0.23` for FP4 serving with BF16
training); counting the backward products into activations doubles `α₀`. The count is stated multiplied out, so it says
something when `k = 0`; in the slowdown form a division by zero is zero. -/
def CoveringBoundAdditive : Prop :=
  ∀ (d : RegisteredCircuit) (B : Nci.Covering.MulBlocks d) (s k D : ℕ) (cw ca : ℝ) (n : ℕ)
    (I : Fin n → Nci.Covering.Subcircuit d.D),
    (k : ℝ) < cw * D → 0 < ca → Assumptions.ConditionalParameterHardAt d B s cw → Nci.Covering.Covers d s k I →
      ((Nci.Assumptions.chainedBlocksAt d B D ca).card : ℝ) ≤ n * (k / cw * (k / ca) / 4 + k) ∧
      ((Nci.Assumptions.chainedBlocksAt d B D ca).card : ℝ) / d.D.N * s / (k / cw * (k / ca) / 4 + k) ≤
        s * n / d.D.N

end Nci.Cover.Guarantees
