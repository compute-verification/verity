import Definitions.Nci

/-!
# NCI in reads form: the read factor, and reads in a set of gates

`readFactor K b_w b_a d` is the actual-circuit model's `f(K)`: the least number of bits of novel information an IU that
reads at most `K` bits pays a product of an updated matrix multiplication chain, at widths `b_w` of a weight entry and `b_a` of an activation entry and smallest inner
dimension `d`. NCI's slowdown factor is `F = f(K) / (2α)`.

`readsIn d R G` counts what `R` reads physically, baseline included, but only in the gates `G`: the width of
`in(R) ∩ G`. At `G = univ` it is at least `novelReads d R`, which leaves the baseline out.
-/

namespace Nci

open Verity

/-- **The read factor** `f(K) = max_{0<μ<1} min{(1 + √μ)² b_w b_a d / K, √2 (1 + √μ) √(1 − μ) b_a √(b_w / K)}`, the
maximum written as a supremum: `F`'s supremum without its `1/(2α)`. -/
noncomputable def readFactor (K b_w b_a d : ℝ) : ℝ :=
  ⨆ μ : Set.Ioo (0 : ℝ) 1,
    min ((1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / K) (√2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / K))

/-- **The reads of `R` in `G`**: the width of the gates of `G` outside `R` that `R` reads, baseline included. -/
def readsIn (d : RegisteredCircuit) (R G : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R ∩ G)

end Nci
