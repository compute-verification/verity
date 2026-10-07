import Definitions.Core.Circuit

/-!
# Partitions, committed wires, and the correctness of a unit

A partition puts every computed gate of a circuit in exactly one of `n` units; input gates are in none. Proof units
(PUs), replay units (RUs) and isolation units (IUs) are partitions in three roles, and `Refines` nests them (PUs inside
RUs). The committed wires are derived, never stored: the inputs, the outputs, and every gate that a gate of another unit
reads (`committed`). Unit `u`'s committed gates are `out(R)` for its gates `R` (`gates u`).

Unit `u` is **correct** in a transcript `X` when its committed gates hold what its gates compute from `X` on the gates
it reads from outside, in(R) (`Correct`): recompute `u`'s gates in order, reading every other gate from `X`
(`localEval`), and compare with `X` on `u`'s committed gates. Only committed values are bound, and `wrong X` is the set
of units that are not correct.
-/

namespace Verity

open Finset

/-- A partition of a circuit's computed gates into units `0, …, n - 1`; input gates are in no unit (`none`). -/
structure Partition (C : Circuit) (n : ℕ) where
  unit : Fin C.N → Option (Fin n)
  input_iff : ∀ g, unit g = none ↔ C.isInput g = true

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

/-- The gates of unit `u`. -/
def gates (u : Fin n) : Finset (Fin C.N) := univ.filter fun g => P.unit g = some u

/-- The committed wires: inputs, outputs, and every gate that a gate of another unit reads. -/
def committed : Finset (Fin C.N) :=
  univ.filter fun g => C.isInput g = true ∨ g ∈ C.outputs ∨ ∃ h, g ∈ C.reads h ∧ P.unit h ≠ P.unit g

/-- `u`'s gates recomputed in order, every other gate read from `X`. -/
def localEval (X : C.Transcript) (u : Fin n) : C.Transcript :=
  C.run fun g => if P.unit g = some u then none else some (X g)

/-- **Correctness of a unit**: its committed gates hold what its gates compute from its committed inputs. -/
def Correct (X : C.Transcript) (u : Fin n) : Prop :=
  ∀ g ∈ P.committed, P.unit g = some u → X g = P.localEval X u g

open Classical in
/-- The wrong units of a transcript. -/
noncomputable def wrong (X : C.Transcript) : Finset (Fin n) :=
  univ.filter fun u => ¬ P.Correct X u

open Classical in
/-- The units holding a gate that feeds some gate of `T`. -/
noncomputable def cone (T : Finset (Fin C.N)) : Finset (Fin n) :=
  univ.filter fun u => ∃ t ∈ T, ∃ g, C.Feeds g t ∧ P.unit g = some u

open Classical in
/-- `u`'s committed inputs and outputs: what an opening of the unit shows. -/
noncomputable def io (u : Fin n) : Finset (Fin C.N) :=
  P.committed.filter fun g => P.unit g = some u ∨ ∃ h, P.unit h = some u ∧ g ∈ C.reads h

/-- `Pf` refines `Pc`: every fine unit lies inside one coarse unit, its `parent`. -/
structure Refines {nf nc : ℕ} (Pf : Partition C nf) (Pc : Partition C nc) where
  parent : Fin nf → Fin nc
  sub : ∀ g, Pc.unit g = (Pf.unit g).map parent

end Partition

end Verity
