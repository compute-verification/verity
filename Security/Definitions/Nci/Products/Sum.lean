import Definitions.Nci.Products.System

/-!
# Products made whole through chains, several in a block: P-share on a set and the bound summed over it

A block may do several of a whole-making system's products at once, a linear layer's forward, `dX` and `dW` among them,
reading an array once for every product that reads it. Two premises on a set `G` of the system's products, facts of the
given circuit, let the cap count them together:

* **P-share on a set** (`System.SharedOn`) at `c_G`: in every inference block, `G`'s products have
  `Σ_{p ∈ G} |hadA_p| |hadB_p| ≤ c_G |hadValues G|²`. Products that share arrays only as forward/dX/dW triples, three
  arrays read two times each, have it at `c_G = 1/3`: a triple's `xy + yz + zx ≤ (x + y + z)²/3`, and triples on
  disjoint arrays add. A set within one class has it at that class's `c_j` (P-share, `System.Shared`).
* **Apart** (`System.Apart`): `G`'s products have disjoint MACs.

`sumBound` is the bound on `ρ` that the cap summed over a set (`Nci.Products.Guarantees.SumCap`) gives at any `t > 0`,
`(N - h O / (2 t)) / (w k + h (t + (k / b_C) / t) / 2) · c / |C|` for a set of products with `N` MACs and `O` output
entries in all, in cells of at most `w` MACs, and `h = √c_G k (1 + E) / b`, `E` the whole-making factors of its classes
summed: `k (1 + E) / b` bounds the values a block has, paid at their width or made whole, and `k / b_C + s` the output
entries it touches through whole cells, `s` the chains it starts. It is largest at `t² = k / b_C + O / m`, `m` its own
number of blocks.
-/

namespace Nci.Products

open Finset
open Verity (Declaration)
open Nci.Formalism
open scoped ENNReal

namespace System

variable {d : Declaration} {ι κ : Type} (sys : System d ι κ)

/-- **P-share on a set** `G` at `c_G`, a fact of the given circuit at the policy's `(c, k)`: in every inference block,
`G`'s products have `Σ_{p ∈ G} |hadA_p| |hadB_p| ≤ c_G |hadValues G|²`. -/
def SharedOn (c k : ℕ) (cG : ℝ) (G : Finset ι) : Prop :=
  ∀ I, InferenceBlock d c k I →
    ((∑ p ∈ G, ((sys.prod p).hadA I).card * ((sys.prod p).hadB I).card : ℕ) : ℝ) ≤
      cG * ((sys.hadValues G I).card : ℝ) ^ 2

/-- **Apart** on a set `G`, a fact of the given circuit: `G`'s products have disjoint MACs. -/
def Apart (G : Finset ι) : Prop :=
  ∀ p ∈ G, ∀ q ∈ G, p ≠ q → Disjoint (sys.prod p).macs (sys.prod q).macs

end System

/-- **The products kind's bound summed over a set**, at `t > 0`:
`(N - h O / (2 t)) / (w k + h (t + (k / b_C) / t) / 2) · c / |C|`, `0` where it is negative, for a set of products with
`N` MACs and `O` output entries in all, in cells of at most `w` MACs, `h = √c_G k (1 + E) / b` with P-share on the set
at `c_G` and `E` the whole-making factors of its classes summed. -/
noncomputable def sumBound (d : Declaration) (N O w c k b bC : ℕ) (cG E t : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (((N : ℝ) - √cG * ((k : ℝ) * (1 + E) / b) * O / (2 * t)) /
      (w * k + √cG * ((k : ℝ) * (1 + E) / b) * (t + (k : ℝ) / bC / t) / 2)) *
    (c : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)

end Nci.Products
