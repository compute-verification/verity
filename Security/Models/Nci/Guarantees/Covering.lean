import Models.Nci.Assumptions.Covering

/-!
# The covering bound

Daniel's standard form (22 Oct 2026), at one width `b` for a weight and for an activation and one compression rate `r`
for both, `c = r·b`: if `cD > k` and `C` is parameter-hard at `(s, r, b)` (`Assumptions.ParameterHard`), every
`(s,k)`-covering `𝓘` of `C` (`Cover.Covers`) has `|𝓘| ≥ |𝓜*| / ((k/c)² + k)` blocks, `𝓜*` being the multiplication
blocks chained at `(D, r, b)` (`Assumptions.ActivationChained`), so that its slowdown is
`ρ(𝓘) ≥ (|𝓜*|/|C|) · s / ((k/c)² + k)`.

It is a statement beside `Nci.Complies`, `Nci.Chain` and `Nci.F`, not a case of them. A block's size `s` counts its
inputs, the registered ones too, where an IU's size counts its computed gates and its reads of `θ` are free and
unbounded, so a division that complies is a covering only with a bound on each IU's reads of `θ`. `Chain` fixes a
matrix multiplication's products, entries and links and `F` charges weights and activations at two widths `b_w`, `b_a`;
here a multiplication block is any set of gates with an output gate, a token's computation is any set of gates outside
`θ`, and weights and activations cost one `c = r·b`.
-/

namespace Nci.Guarantees

open Verity

open Classical in
/-- **The covering bound** (Daniel's standard form, 22 Oct 2026): at width `b > 0` and compression rate `0 < r ≤ 1`,
`c = r·b`, with `cD > k`, if `C` is parameter-hard at `(s, r, b)` (`Assumptions.ParameterHard`), every `(s,k)`-covering
`𝓘` of `C` (`Cover.Covers`) has `|𝓘| ≥ |𝓜*| / ((k/c)² + k)` blocks, written multiplied out, and slowdown
`ρ(𝓘) ≥ (|𝓜*|/|C|) · s / ((k/c)² + k)` (`Cover.slowdown`), where `𝓜*` is the multiplication blocks chained at
`(D, r, b)` (`Assumptions.ActivationChained`).

A block completes at most `(k/c)²` chained multiplication blocks: their weights number at most `k/c` by
parameter-hardness, and their tokens at most `k/c`, since it reads at least `c` bits of each such token. It computes the
output gate of at most `k` others, since each such block it doesn't complete has a gate among its at most `k` inputs
outside `θ`. -/
def CoveringBound : Prop :=
  ∀ {ι τ : Type} [DecidableEq ι] (d : Declaration) (M : Cover.Mults d ι τ) (b r s k D : ℝ),
    0 < b → 0 < r → r ≤ 1 → k < r * b * D → Assumptions.ParameterHard d M s r b →
    ∀ 𝓘 : Finset (Cover.Subcircuit d.D), Cover.Covers d s k 𝓘 →
      ((M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2).card : ℝ) ≤
          𝓘.card * ((k / (r * b)) ^ 2 + k) ∧
        ((M.P.filter fun p => Assumptions.ActivationChained d M D r b p.1 p.2).card : ℝ) / d.D.N * s /
            ((k / (r * b)) ^ 2 + k) ≤ Cover.slowdown d s 𝓘

end Nci.Guarantees
