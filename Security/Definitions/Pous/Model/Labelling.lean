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

end Pous.Labelling
