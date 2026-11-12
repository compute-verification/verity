module

public import Definitions.Assumptions
public import Definitions.Nci
public import Definitions.Nci.Cut.Toy

/-!
# The min-cut bound and its slowdown

The training bound of record, over the vocabulary of `Definitions.Nci.Cut`. `Bound` holds for every circuit, every
charged work and every partition into blocks that each read at most `k` gates outside `θ`; `Slowdown` is its form for
`ρ = m c / |C|`. `HE` and `HEShared` put (H) and (E) (`Definitions.Assumptions`) into both, with two arrays
`W` and `X` whose matmuls share no pair of values, and with shared arrays. No statement reads a fan-in bound, a width
or a padding: `W` and `X` may share values, outputs may repeat, and each slowdown holds for every `c`, as it must for a
partition into blocks of at most `c` gates, each priced at `c`. `CountHE` is `HE` on a count verdict: its only premise
on the circuit is the verdict itself, `CompliesAtMost c k n`, at most `n` proof blocks of a `(c, k)`-partition, and
both bounds are at `n`. `HECharged` is `HE`'s count over a set of the blocks, each of them within the budget and the
rest charged whole, with what all the blocks read as a total and what the set's blocks read squared and summed.

The witnesses that each says something are at the toy (`Nci.Cut.Toy`): `Cut_honest`, `CountHE_honest` and
`HECharged_honest`, their premises holding together, and each statement's `_nontrivial`, the toy beating its bound
once the block cap `ReadsAtMost`, the verdict, or `ReadsOn` is dropped.
-/

@[expose] public section

namespace Nci.Cut.Guarantees

open Verity Finset

/-- **The min-cut bound.** Every partition of the gates into `m` blocks, each reading at most `k` gates outside `θ`
(`ReadsAtMost`), has `N ≤ m (k + α) + √(Π(k) m O)`. -/
def Bound : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : RegisteredCircuit) (W : Work d.D ι) (k m : ℕ) (P : Partition d.D m),
    ReadsAtMost d P k → (W.N : ℝ) ≤ m * (k + alpha d W k) + √(piMax d W k * m * W.O)

/-- **The slowdown.** With each of the `m` blocks priced at `c`, `ρ = m c / |C|` is at least
`(c / |C|) (N / (k + α) - √(Π(k) O N) / (k + α)^{3/2})`; the inequality holds for every `c`, so for blocks of at most
`c` gates. -/
def Slowdown : Prop :=
  ∀ (ι : Type) [Fintype ι] [DecidableEq ι] (d : RegisteredCircuit) (W : Work d.D ι) (k m c : ℕ) (P : Partition d.D m),
    ReadsAtMost d P k →
      (c / d.D.size d.D.computed : ℝ) *
          (W.N / (k + alpha d W k) - √(piMax d W k * W.O * W.N) / (k + alpha d W k) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

/-- **The bound under (H) and (E).** `val` names the value each word carries (`ValuesInjective`), and no two matmuls
share both a `w`- and an `x`-value (`NoSharedPair`). (H) at the arrays `W` and `X`, a value of each costing `β_w` and
`β_x` (`Assumptions.Hardness`), and (E) at `b > 0` (`Assumptions.Exports`) give, for every partition into `m` blocks
each reading at most `k` gates outside `θ`, `N ≤ m (k + α) + (k/2) √(m O / (β_w β_x))` with
`α = ½ √(k³ / (β_w β_x b))` (`alphaHE`), and for every `c`,
`m c / |C| ≥ (c / |C|) (N / (k + α) - (k/2) √(O N / (β_w β_x)) / (k + α)^{3/2})`. -/
def HE : Prop :=
  ∀ (ι V : Type) [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (k m c : ℕ) (P : Partition d.D m) (βw βx b : ℝ), 0 < βw → 0 < βx → 0 < b →
    ValuesInjective W val → NoSharedPair W val → Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k →
    Assumptions.Exports W b → ReadsAtMost d P k →
      (W.N : ℝ) ≤ m * (k + alphaHE k (βw * βx) b) + k / 2 * √(m * W.O / (βw * βx)) ∧
      (c / d.D.size d.D.computed : ℝ) * (W.N / (k + alphaHE k (βw * βx) b) -
          k / 2 * √(W.O * W.N / (βw * βx)) / (k + alphaHE k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

/-- **The bound under (H) and (E) on a count verdict.** A declaration that complies with `CompliesAtMost c k n`, at
most `n` proof blocks of a `(c, k)`-partition (the verdict a published `complies` of the count of `(c, k)`-inference
blocks gives, `NciCheck.Properties.DescCountsCK`), has, under `HE`'s premises on its work and no other on its circuit,
`N ≤ n (k + α) + (k/2) √(n O / (β_w β_x))` (`workHE`, `α = alphaHE k (β_w β_x) b`), and its size ratio at `n` blocks
each priced at `c` is at least `(c / |C|) (N / (k + α) - (k/2) √(O N / (β_w β_x)) / (k + α)^{3/2})`. The verdict's
blocks of at most `c` gates each read at most `k` gates outside `θ` (`ReadsAtMost`), and both bounds grow with the
number of blocks. -/
def CountHE : Prop :=
  ∀ (ι V : Type) [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (c k n : ℕ) (βw βx b : ℝ), 0 < βw → 0 < βx → 0 < b →
    ValuesInjective W val → NoSharedPair W val → Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k →
    Assumptions.Exports W b → CompliesAtMost c k n d →
      (W.N : ℝ) ≤ workHE n k (βw * βx) b W.O ∧
      (c / d.D.size d.D.computed : ℝ) * (W.N / (k + alphaHE k (βw * βx) b) -
          k / 2 * √(W.O * W.N / (βw * βx)) / (k + alphaHE k (βw * βx) b) ^ (3 / 2 : ℝ)) ≤
        n * c / d.D.size d.D.computed

/-- **The bound under (H) and (E), with shared arrays.** Matmul `a` reads arrays `(e a).1` and `(e a).2`, an edge of a
graph with no clique of `ω + 1` (`Arrays`), as when the arrays fall into `ω` classes and no matmul reads two of one
class: `ω = 3` with every layer's forward, `dX` and `dW` products on its weights, activations and output gradients,
however many layers share an array, and `ω = 2` with products of at most two of those three kinds. Array `r` costs
`β_r` a value, each matmul's pair at least `q` (`CostsAtLeast`). With (H) at those arrays and (E) at `b`, both terms of
`HE` grow by `F = √(2 (1 - 1/ω))` (`shareF`), with `q` for `β_w β_x`: `N ≤ m (k + F α) + F (k/2) √(m O / q)` with
`α = ½ √(k³ / (q b))`, and the slowdown likewise. -/
def HEShared : Prop :=
  ∀ (ι ρ V : Type) [Fintype ι] [DecidableEq ι] [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (k m c ω : ℕ) (P : Partition d.D m) (G : SimpleGraph ρ)
    (A : ρ → Finset (Finset (Fin d.D.N))) (e : ι → ρ × ρ) (β : ρ → ℝ) (q b : ℝ), 0 < ω → (∀ r, 0 ≤ β r) → 0 < q →
    0 < b → Arrays W G A e ω → ValuesInjective W val → CostsAtLeast e β q → Assumptions.Hardness d A val β k →
    Assumptions.Exports W b → ReadsAtMost d P k →
      (W.N : ℝ) ≤ m * (k + shareF ω * alphaHE k q b) + shareF ω * (k / 2 * √(m * W.O / q)) ∧
      (c / d.D.size d.D.computed : ℝ) * (W.N / (k + shareF ω * alphaHE k q b) -
          shareF ω * (k / 2 * √(W.O * W.N / q)) / (k + shareF ω * alphaHE k q b) ^ (3 / 2 : ℝ)) ≤
        m * c / d.D.size d.D.computed

/-- **The bound under (H) and (E), with blocks charged whole.** `G` is a set of a partition's blocks, each reading at
most `k` gates outside `θ`; the blocks outside `G` are charged whole, with no budget on what each reads. `T` is at least
what all the blocks read outside `θ` between them, and `S` at least the sum over `G` of the squares of what each of its
blocks reads (`ReadsOn`, the three together). Under `HE`'s premises, with (H) at `k`, `N` is at most the MACs the blocks
outside `G` hold whole (`Work.inside`) and `T + ½ √(S O / (β_w β_x)) + ½ √(S T / (β_w β_x b))` (`workHEOn`). A MAC split
between blocks is counted in `T`, whichever blocks hold its gates, and `O` is the whole work's, the charged blocks'
MACs' among it. MACs are disjoint, so where each has at least `γ` gates and each block outside `G` at most `c`, those
blocks hold at most `c / γ` MACs whole each: `γ N ≤ |Gᶜ| c + γ (T + …)`. At `G` every block, `T = m k` and `S = m k²`
the first bound is `HE`'s.

It is no second statement of the count. The count over a set of blocks, each under its own budget and the rest's MACs
held whole, is stated once, and this is its corollary at one (H) for every block of `G`, with the charged blocks' MACs
priced in gates. -/
def HECharged : Prop :=
  ∀ (ι V : Type) [Fintype ι] [DecidableEq ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) (k m c γ T S : ℕ) (P : Partition d.D m) (G : Finset (Fin m)) (βw βx b : ℝ),
    0 < βw → 0 < βx → 0 < b → ValuesInjective W val → NoSharedPair W val →
    Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k → Assumptions.Exports W b → ReadsOn d P G k T S →
      (W.N : ℝ) ≤ ((∑ u ∈ Gᶜ, (W.inside (P.gates u)).card : ℕ) : ℝ) + workHEOn T S (βw * βx) b W.O ∧
      ((∀ s, γ ≤ (W.mac s).card) → (∀ u ∈ Gᶜ, (P.gates u).card ≤ c) →
        (γ : ℝ) * W.N ≤ (Gᶜ.card : ℝ) * c + γ * workHEOn T S (βw * βx) b W.O)

/-! ### Witnesses at the toy -/

/-- **The premises hold together** at the toy: two blocks of one MAC each, reading two gates each; each word its own
value; the arrays `W` and `X` on the edge of the complete graph on two vertices (`ω = 2`), a value of each costing one
gate, so `q = 1`; (H) at those costs and the read cap `k = 2`; and (E) at `b = 1`, since no set of gates finishes an
output it holds no wire of. -/
def Cut_honest : Prop :=
  ReadsAtMost Toy.d Toy.P2 2 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Arrays Toy.W ⊤ Toy.arrays Toy.edge 2 ∧ CostsAtLeast Toy.edge ![1, 1] 1 ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 2 ∧ Assumptions.Exports Toy.W 1

/-- **`Bound` without the block cap**: one block holding both MACs reads three gates, more than `k = 0`, and
`N = 2` exceeds `m (k + α) + √(Π(0) m O) = 0`. -/
def Bound_nontrivial : Prop :=
  ¬ ReadsAtMost Toy.d Toy.P1 0 ∧
    ¬ ((Toy.W.N : ℝ) ≤ ((1 : ℕ) : ℝ) * (((0 : ℕ) : ℝ) + alpha Toy.d Toy.W 0) +
        √(piMax Toy.d Toy.W 0 * ((1 : ℕ) : ℝ) * Toy.W.O))

/-- **`Slowdown` without the block cap**: one block holding both MACs reads three gates, more than `k = 1`, and at
`c = 1` the bound `(c / |C|) (N / (k + α) - √(Π(1) O N) / (k + α)^{3/2}) = 1` exceeds `m c / |C| = 1/2`. -/
def Slowdown_nontrivial : Prop :=
  ¬ ReadsAtMost Toy.d Toy.P1 1 ∧
    ¬ ((((1 : ℕ) : ℝ) / Toy.d.D.size Toy.d.D.computed : ℝ) * (Toy.W.N / (((1 : ℕ) : ℝ) + alpha Toy.d Toy.W 1) -
          √(piMax Toy.d Toy.W 1 * Toy.W.O * Toy.W.N) / (((1 : ℕ) : ℝ) + alpha Toy.d Toy.W 1) ^ (3 / 2 : ℝ)) ≤
        ((1 : ℕ) : ℝ) * ((1 : ℕ) : ℝ) / Toy.d.D.size Toy.d.D.computed)

/-- **`HE` without the block cap**: `Cut_honest`'s values, costs and (E), with (H) at `k = 0`; one block holding both
MACs reads three gates, more than `k = 0`, and `N = 2` exceeds the bound, `0`. -/
def HE_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 0 ∧ Assumptions.Exports Toy.W 1 ∧
    ¬ ReadsAtMost Toy.d Toy.P1 0 ∧
    ¬ ((Toy.W.N : ℝ) ≤ ((1 : ℕ) : ℝ) * (((0 : ℕ) : ℝ) + alphaHE 0 (1 * 1) 1) +
        ((0 : ℕ) : ℝ) / 2 * √(((1 : ℕ) : ℝ) * Toy.W.O / (1 * 1)))

/-- **`CountHE`'s premises hold together** at the toy: a block for each MAC, one gate reading two, so the toy complies
with `CompliesAtMost 1 2 2`; each word its own value; (H) at unit costs and `k = 2`; and (E) at `b = 1`. -/
def CountHE_honest : Prop :=
  CompliesAtMost 1 2 2 Toy.d ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 2 ∧ Assumptions.Exports Toy.W 1

/-- **`CountHE` without the verdict**: `HE_nontrivial`'s values, costs, (H) at `k = 0` and (E); every block of a
partition of the toy reads an unregistered input, so the toy fails `CompliesAtMost 1 0 1`, and `N = 2` exceeds the bound
at one block, `0`. -/
def CountHE_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 0 ∧ Assumptions.Exports Toy.W 1 ∧
    ¬ CompliesAtMost 1 0 1 Toy.d ∧ ¬ ((Toy.W.N : ℝ) ≤ workHE 1 0 (1 * 1) 1 Toy.W.O)

/-- **`HEShared` without the block cap**: `Cut_honest`'s arrays, costs and (E), with (H) at `k = 0`; one block
holding both MACs reads three gates, more than `k = 0`, and `N = 2` exceeds the bound, `0`. -/
def HEShared_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ Arrays Toy.W ⊤ Toy.arrays Toy.edge 2 ∧ ValuesInjective Toy.W Toy.val ∧
    CostsAtLeast Toy.edge ![1, 1] 1 ∧ Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 0 ∧
    Assumptions.Exports Toy.W 1 ∧ ¬ ReadsAtMost Toy.d Toy.P1 0 ∧
    ¬ ((Toy.W.N : ℝ) ≤ ((1 : ℕ) : ℝ) * (((0 : ℕ) : ℝ) + shareF 2 * alphaHE 0 1 1) +
        shareF 2 * (((0 : ℕ) : ℝ) / 2 * √(((1 : ℕ) : ℝ) * Toy.W.O / 1)))

/-- **`HECharged`'s premises hold together** at the toy: a block for each MAC, the first in `G` and the second
charged whole; each word its own value; (H) at unit costs and `k = 2`, which the first block reads; (E) at `b = 1`;
the two blocks read at most `T = 4` gates between them, and the first's square is at most `S = 4`; and each MAC is one
gate, as is the charged block. -/
def HECharged_honest : Prop :=
  ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 2 ∧ Assumptions.Exports Toy.W 1 ∧
    ReadsOn Toy.d Toy.P2 {0} 2 4 4 ∧
    (∀ s, 1 ≤ (Toy.W.mac s).card) ∧ ∀ u ∈ ({0} : Finset (Fin 2))ᶜ, (Toy.P2.gates u).card ≤ 1

/-- **`HECharged` without `ReadsOn`**: `HE_nontrivial`'s values, costs, (H) at `k = 0` and (E); one block holding
both MACs, no block charged, reads three gates, more than `k = 0` and than the total `T = 0`, and `N = 2` exceeds the
bound at `T = 0` and `S = 0`, `0`. -/
def HECharged_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
    Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 0 ∧ Assumptions.Exports Toy.W 1 ∧
    ¬ ReadsOn Toy.d Toy.P1 univ 0 0 0 ∧
    ¬ ((Toy.W.N : ℝ) ≤ ((∑ u ∈ (univ : Finset (Fin 1))ᶜ, (Toy.W.inside (Toy.P1.gates u)).card : ℕ) : ℝ) +
        workHEOn 0 0 (1 * 1) 1 Toy.W.O)

end Nci.Cut.Guarantees
