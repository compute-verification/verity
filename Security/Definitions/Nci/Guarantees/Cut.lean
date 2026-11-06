module

public import Definitions.Nci.Assumptions.Cut
public import Definitions.Nci.Cut.Toy

/-!
# The min-cut bound and its slowdown

The training bound of record, over the vocabulary of `Definitions.Nci.Cut`. `Bound` holds for every circuit, every
charged work and every partition into blocks that each read at most `k` gates outside `θ`; `Slowdown` is its form for
`ρ = m c / |C|`. `HE` and `HEShared` put (H) and (E) (`Definitions.Nci.Assumptions.Cut`) into both, with two arrays
`W` and `X` whose matmuls share no pair of values, and with shared arrays. No statement reads a fan-in bound, a width
or a padding: `W` and `X` may share values, outputs may repeat, and each slowdown holds for every `c`, as it must for a
partition into blocks of at most `c` gates, each priced at `c`.

The witnesses that each says something are at the toy (`Nci.Cut.Toy`): `Cut_honest`, their premises holding together,
and each statement's `_nontrivial`, the toy beating its bound once the block cap `ReadsAtMost` is dropped.
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

/-- **`HEShared` without the block cap**: `Cut_honest`'s arrays, costs and (E), with (H) at `k = 0`; one block
holding both MACs reads three gates, more than `k = 0`, and `N = 2` exceeds the bound, `0`. -/
def HEShared_nontrivial : Prop :=
  (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ Arrays Toy.W ⊤ Toy.arrays Toy.edge 2 ∧ ValuesInjective Toy.W Toy.val ∧
    CostsAtLeast Toy.edge ![1, 1] 1 ∧ Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] 0 ∧
    Assumptions.Exports Toy.W 1 ∧ ¬ ReadsAtMost Toy.d Toy.P1 0 ∧
    ¬ ((Toy.W.N : ℝ) ≤ ((1 : ℕ) : ℝ) * (((0 : ℕ) : ℝ) + shareF 2 * alphaHE 0 1 1) +
        shareF 2 * (((0 : ℕ) : ℝ) / 2 * √(((1 : ℕ) : ℝ) * Toy.W.O / 1)))

end Nci.Cut.Guarantees
