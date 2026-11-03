import Mathlib.Analysis.Real.Sqrt
import Definitions.Nci.Products.Sum

/-!
# Products made whole through chains, several in a block: the cap and its bound on `ρ` summed over a set

The theorems about a set `G` of a whole-making system's products (`Definitions.Nci.Products.Sum`), proved in
`Nci.Products.SecurityProofs`: the sum over products, several of them in one block (the note's `Λ`, `SHARE`), where the
cap with operands made whole (`Nci.Products.Guarantees.WholeCap`) bounds one product at a time. `k` is the bits an
inference block reads beyond `θ`, `b` an operand's width, `b_C` a partial sum's, `μ` a post-fixed point of the system,
`E = Σ_j e_j` the whole-making factors (`wholeFactor`) of `G`'s classes summed, and `h = √c_G k (1 + E) / b`.

* **The cap summed over a set** (`SumCap`): a set `G` of the system's products with P-share on `G` at `c_G`
  (`System.SharedOn`) and disjoint MACs (`System.Apart`), each in cells of at most `w` MACs with its partial sums
  received at their width in them, has, in every partition into `m` inference blocks,
  `N ≤ m w k + h √(m (m k / b_C + O))`, with `N` and `O` the MACs and output entries of `G`'s products in all. In each
  block, Loomis–Whitney and Cauchy–Schwarz bound the MACs of `G`'s whole cells by
  `√(Σ_q |π₁₂| |π₂₃|) √(Σ_q |π₁₃|)`; P-share on `G` bounds the first factor by `√c_G H`, `H` the values the block has,
  at most `k (1 + E) / b`, paid at their width from its `k` bits or made whole, at most `E k / b`, by the class bound;
  and the entries it touches, at most `k / b_C + s_u`, have their partial sums read at `b_C` bits, from at most `k`
  bits more, or start their chains there, `s_u` of them. Cauchy–Schwarz over the blocks gives
  `Σ_u √(k / b_C + s_u) ≤ √(m (m k / b_C + O))`, each chain started once, and the split cells add at most `w k` MACs a
  block.
* **The bound summed over a set** (`SumBound`): `ρ ≥ (N - h O / (2 t)) / (w k + h (t + (k / b_C) / t) / 2) · c / |C|`
  at every `t > 0` (`sumBound`), a partition-level bound (`Nci.Formalism.PartitionBound`), on any set `G` with its
  premises: the cap with `√(m (m k / b_C + O)) ≤ (t m + (m k / b_C + O) / t) / 2`, largest at `t² = k / b_C + O / m`.

Against io_bound's `Λ`: `√c_G / √(1/4)` at `c_G = 1/3` is its `SHARE`, `2/√3`, a lone product's `Σ |hadA| |hadB|` being
at most `H²/4`; the partial sums' reads charged apart from the operands', from at most `k` bits each, take the place of
its `feed`, since a partial sum finished into another product's operand may be read for both; and `1 + E` is its
whole-making, with `E` summed over `G`'s classes. The premises are the class bound's (`Nci.Products.Guarantees.WholeCap`
names them), P-share on `G`, Apart, and each product's cells (`Standard.Cells`) and partial sums received at their width
(`Standard.Received`).

Each has an honest and a nontrivial lemma (`Definitions.Nci.Guarantees.Products.SumWitnesses`).
-/

namespace Nci.Products.Guarantees

open Finset
open Verity (Declaration Partition)
open Nci.Formalism Nci.Products

/-- **The cap summed over a set**: a set `G` of a whole-making system's products, with the class bound's premises,
P-share on `G` at `c_G` and disjoint MACs, each in cells of at most `w` MACs with its partial sums received at their
width in them, has in every partition into `m` inference blocks `N ≤ m w k + h √(m (m k / b_C + O))`, `N` and `O` the
MACs and output entries of `G`'s products in all, `h = √c_G k (1 + E) / b` and `E` the whole-making factors of their
classes summed. -/
def SumCap : Prop :=
  ∀ (d : Declaration) (ι κ : Type) [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ] (sys : System d ι κ)
    (c k b bC : ℕ) (cs : κ → ℝ) (A : κ → κ → ℝ) (μ : κ → ℝ) (G : Finset ι) (cG : ℝ)
    (cell : (q : ι) → Fin (sys.ni q) → Fin (sys.K (sys.cls q)) → Fin (sys.nj q) → ℕ) (w m : ℕ)
    (P : Partition d.D m),
    (∀ p, (sys.prod p).Wiring) → sys.Values → sys.Shared c k cs → sys.WidthExtended c k b → sys.Matched A →
    sys.SharedOn c k cG G → sys.Apart G → (∀ q ∈ G, (sys.prod q).Cells (cell q) w) →
    (∀ q ∈ G, (sys.prod q).Received (cell q) c k bC) → P.IsCK d c k → 0 < b → 0 < bC → (∀ j, 0 ≤ cs j) →
    (∀ i j, 0 ≤ A i j) → sys.PostFixed k b cs A μ →
    ((∑ q ∈ G, sys.ni q * sys.K (sys.cls q) * sys.nj q : ℕ) : ℝ) ≤
      m * (w * k) + √cG * ((k : ℝ) * (1 + ∑ j ∈ G.image sys.cls, wholeFactor A μ j) / b) *
        √((m : ℝ) * (m * ((k : ℝ) / bC) + (∑ q ∈ G, sys.ni q * sys.nj q : ℕ)))

/-- **The bound summed over a set**: for a set `G` of a whole-making system's products with the cap summed over a set's
premises, every partition `𝓘` into inference blocks has, at every `t > 0`,
`|𝓘| c / |C| ≥ (N - h O / (2 t)) / (w k + h (t + (k / b_C) / t) / 2) · c / |C|`, a partition-level bound. -/
def SumBound : Prop :=
  ∀ (d : Declaration) (ι κ : Type) [Fintype ι] [DecidableEq ι] [Fintype κ] [DecidableEq κ] (sys : System d ι κ)
    (c k b bC : ℕ) (cs : κ → ℝ) (A : κ → κ → ℝ) (μ : κ → ℝ) (G : Finset ι) (cG : ℝ)
    (cell : (q : ι) → Fin (sys.ni q) → Fin (sys.K (sys.cls q)) → Fin (sys.nj q) → ℕ) (w : ℕ) (t : ℝ),
    (∀ p, (sys.prod p).Wiring) → sys.Values → sys.Shared c k cs → sys.WidthExtended c k b → sys.Matched A →
    sys.SharedOn c k cG G → sys.Apart G → (∀ q ∈ G, (sys.prod q).Cells (cell q) w) →
    (∀ q ∈ G, (sys.prod q).Received (cell q) c k bC) → 0 < b → 0 < bC → (∀ j, 0 ≤ cs j) → (∀ i j, 0 ≤ A i j) →
    sys.PostFixed k b cs A μ → 0 < t →
    PartitionBound d c k (sumBound d (∑ q ∈ G, sys.ni q * sys.K (sys.cls q) * sys.nj q)
      (∑ q ∈ G, sys.ni q * sys.nj q) w c k b bC cG (∑ j ∈ G.image sys.cls, wholeFactor A μ j) t)

end Nci.Products.Guarantees
