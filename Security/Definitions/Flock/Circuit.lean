import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Definitions.Core.Boolean

/-!
# Partitions of a circuit, and the correctness of a unit

The circuit is the Boolean one (`Definitions.Core.Boolean`: `Op`, `Circuit`, `Circuit.eval`). A partition puts
every computed gate in exactly one of `n` units; the input gates are the input unit (`none`). The
committed wires are derived, never stored: the inputs, the outputs, and every gate that a gate of
another unit reads (`Partition.committed`).

A transcript assigns a bit to every gate; only its committed entries are ever read. Unit `u` is
**correct** in a transcript `X` when its committed outputs are its gates applied to its committed
inputs (`Partition.Correct`): evaluate `u`'s gates in order, reading every other gate from `X`
(`Partition.localEval`), and compare with `X` on `u`'s committed gates.

The lemmas about them (composition, congruence, refinement) are in `Proofs.Flock.Soundness.Audit.Circuit`.
-/

namespace FlockSoundness.Audit

namespace Circuit

variable (C : Circuit)

/-- The input unit: the committed inputs are the anchors' values. -/
def InputsAgree (X a : Fin C.N → Bool) : Prop :=
  ∀ g, (C.op g).isInput = true → X g = a g

/-- `g` feeds `t`: `g = t`, or `g` is an argument of a gate that feeds `t`. -/
inductive Feeds : Fin C.N → Fin C.N → Prop
  | refl (t : Fin C.N) : Feeds t t
  | step {g h t : Fin C.N} : g ∈ (C.op h).args → Feeds h t → Feeds g t

end Circuit

/-- A partition of a circuit's computed gates into `n` units; the input gates are the input unit
(`none`). -/
structure Partition (C : Circuit) (n : ℕ) where
  unit : Fin C.N → Option (Fin n)
  input_iff : ∀ g, unit g = none ↔ (C.op g).isInput = true

namespace Partition

variable {C : Circuit} {n : ℕ} (P : Partition C n)

open Classical in
/-- The committed wires: inputs, outputs, and every gate that a gate of another unit reads. -/
noncomputable def committed : Finset (Fin C.N) :=
  Finset.univ.filter fun g => (C.op g).isInput = true ∨ g ∈ C.outputs ∨
    ∃ h, g ∈ (C.op h).args ∧ P.unit h ≠ P.unit g

/-- `u`'s gates computed in order, every other gate read from `X`. -/
def localEval (X : Fin C.N → Bool) (u : Fin n) : Fin C.N → Bool :=
  C.run fun g => if P.unit g = some u then none else some (X g)

/-- **Correctness of a unit**: its committed outputs are its gates applied to its committed inputs. -/
def Correct (X : Fin C.N → Bool) (u : Fin n) : Prop :=
  ∀ g ∈ P.committed, P.unit g = some u → X g = P.localEval X u g

open Classical in
/-- The wrong units of a transcript. -/
noncomputable def wrong (X : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ P.Correct X u

open Classical in
/-- The units holding an ancestor of some wire in `T`. -/
noncomputable def cone (T : Finset (Fin C.N)) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ t ∈ T, ∃ g, C.Feeds g t ∧ P.unit g = some u

open Classical in
/-- `u`'s committed inputs and outputs. -/
noncomputable def io (u : Fin n) : Finset (Fin C.N) :=
  P.committed.filter fun g => P.unit g = some u ∨ ∃ h, P.unit h = some u ∧ g ∈ (C.op h).args

/-! ## Refinement -/

/-- `Pf` refines `Pc`: every fine unit lies inside one coarse unit, its `parent`. -/
structure Refines {nf nc : ℕ} (Pf : Partition C nf) (Pc : Partition C nc) where
  parent : Fin nf → Fin nc
  sub : ∀ g, Pc.unit g = (Pf.unit g).map parent

end Partition

end FlockSoundness.Audit
