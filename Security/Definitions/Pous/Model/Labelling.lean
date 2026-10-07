import Mathlib.Data.Fintype.Option
import Definitions.Pous.Game.Oracle

/-!
# One-way data-carrying labels and greedy pebbling (B5)

The definitions of the B5 draft (`lean/submissions/pebbling/B5ExPostFactoDraft.lean`: Pietrzak 2018/194 §6.1, §7.1,
§8), verbatim; its header lists the modelling choices. The labels are one-way: nothing here decodes a label.
-/

namespace Pous.Labelling

/-- A DAG on `Fin N` in topological order: the parents of `i` come before `i`. -/
structure TopoDAG (N : ℕ) where
  par : Fin N → Finset (Fin N)
  par_lt : ∀ i, ∀ j ∈ par i, j < i

variable {N w : ℕ}

/-- Label-shaped oracle inputs: a node and a partial assignment of labels. -/
abbrev LabelQry (N w : ℕ) : Type := Fin N × (Fin N → Option (Bits w))

/-- Bitwise XOR. -/
def bxor (a b : Bits w) : Bits w := fun j => xor (a j) (b j)

/-- Data-carrying labels (Pietrzak §8, PoR): `ℓ_i = H(i, ℓ_par(i)) ⊕ d_i`. With `d = 0`, the labels of §7. -/
def label (G : TopoDAG N) (d : Fin N → Bits w) (H : LabelQry N w → Bits w) (i : Fin N) : Bits w :=
  bxor (H (i, fun j => if _h : j ∈ G.par i then some (label G d H j) else none)) (d i)
termination_by i.val
decreasing_by exact G.par_lt i j _h

/-- Greedy parallel black pebbling (Pietrzak §6.1): the nodes carrying a pebble after `r` rounds from `P₀`, when
every node whose parents all carry pebbles receives one. -/
def reach (G : TopoDAG N) (P₀ : Finset (Fin N)) : ℕ → Finset (Fin N)
  | 0 => P₀
  | r + 1 => reach G P₀ r ∪ Finset.univ.filter fun v => G.par v ⊆ reach G P₀ r

/-- `Φ(G, V_C)` is `(s, D, k)`-hard: from any `s` initial pebbles, fewer than `k` challenge nodes carry a pebble
within `D` rounds. -/
def PebblingHard (G : TopoDAG N) (VC : Finset (Fin N)) (s D k : ℕ) : Prop :=
  ∀ P₀ : Finset (Fin N), P₀.card ≤ s → (VC.filter fun c => c ∈ reach G P₀ D).card < k

end Pous.Labelling
