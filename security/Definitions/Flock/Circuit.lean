module

public import Definitions.Core.Boolean
public import Mathlib.Data.Fintype.Card

/-! Partitions of the Boolean circuit, their committed wires, and when a unit is correct in a transcript. -/

@[expose] public section

namespace Verity.Boolean.Circuit

variable (C : Circuit)

/-- The input unit: the committed inputs are the anchors' values. -/
def InputsAgree (X a : Fin C.N → Bool) : Prop :=
  ∀ g, (C.op g).isInput = true → X g = a g

inductive Feeds : Fin C.N → Fin C.N → Prop
  | refl (t : Fin C.N) : Feeds t t
  | step {g h t : Fin C.N} : g ∈ (C.op h).args → Feeds h t → Feeds g t

end Verity.Boolean.Circuit

namespace FlockSoundness.Audit

/-- A partition of a circuit's computed gates into `n` units; the input gates are the input unit
(`none`). -/
structure Partition (C : Verity.Boolean.Circuit) (n : ℕ) where
  unit : Fin C.N → Option (Fin n)
  input_iff : ∀ g, unit g = none ↔ (C.op g).isInput = true

namespace Partition

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n)

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
/-- The incorrect proof blocks of a transcript. -/
noncomputable def incorrect (X : Fin C.N → Bool) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ P.Correct X u

open Classical in
noncomputable def cone (T : Finset (Fin C.N)) : Finset (Fin n) :=
  Finset.univ.filter fun u => ∃ t ∈ T, ∃ g, C.Feeds g t ∧ P.unit g = some u

open Classical in
/-- `u`'s committed inputs and outputs. -/
noncomputable def io (u : Fin n) : Finset (Fin C.N) :=
  P.committed.filter fun g => P.unit g = some u ∨ ∃ h, P.unit h = some u ∧ g ∈ (C.op h).args

structure Refines {nf nc : ℕ} (Pf : Partition C nf) (Pc : Partition C nc) where
  parent : Fin nf → Fin nc
  sub : ∀ g, Pc.unit g = (Pf.unit g).map parent

end Partition

end FlockSoundness.Audit
