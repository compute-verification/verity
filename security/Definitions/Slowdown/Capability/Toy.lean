module

public import Definitions.Core.Partition
public import Definitions.Core.RegisteredCircuit

/-! The toy circuit the slowdown witnesses are stated at. -/

@[expose] public section

namespace Slowdown.Capability.Toy

open Verity Finset

/-- The toy circuit: two inputs, the weight `0` and the token `1`, and two gates `2` and `3` computing `0 ∧ 1`. -/
@[reducible] def C : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := fun g => if g.val < 2 then ∅ else {0, 1}
  reads_lt := by decide
  input_reads := by decide
  op := fun g v => if g.val < 2 then 0 else BitVec.ofBool ((v 0).getLsbD 0 && (v 1).getLsbD 0)
  op_congr := by
    intro g v v' h
    by_cases hg : g.val < 2
    · simp only [hg, ↓reduceIte]
    · simp only [hg, ↓reduceIte]
      rw [h 0 (by simp [hg]), h 1 (by simp [hg])]
  outputs := {2, 3}

/-- The toy registered circuit: the weight is registered. -/
@[reducible] def D : RegisteredCircuit := .ofCircuit C {0} (by decide)

/-- One proof block holding both computed gates. -/
def P1 : Partition C 1 where
  unit := fun g => if g.val < 2 then none else some 0
  input_iff := by decide

end Slowdown.Capability.Toy
