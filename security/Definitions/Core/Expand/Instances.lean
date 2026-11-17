module

public import Definitions.Core.Expand
public import Mathlib.Data.Fin.VecNotation

/-! The instance `AndPair` the bit-level expansion's honest and nontrivial lemmas are stated at. -/

@[expose] public section

namespace Verity.Expand.AndPair

/-- Gates 0 and 1 are two-bit inputs, and gate 2 is their bitwise AND. -/
abbrev C : Circuit where
  N := 3
  w _ := 2
  isInput := ![true, true, false]
  reads := ![∅, ∅, {0, 1}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 2 then v 0 &&& v 1 else 0
  op_congr g v v' hv := by
    by_cases hg : g = 2
    · subst hg
      simp only [hv 0 (by decide), hv 1 (by decide)]
    · simp [hg]
  outputs := {2}

/-- Gate 0 is registered. -/
abbrev d : RegisteredCircuit := .ofCircuit C {0} (by decide)

/-- Gate 2 is the one unit. -/
abbrev P : Partition d.D 1 := ⟨![none, none, some 0], by decide⟩

end Verity.Expand.AndPair
