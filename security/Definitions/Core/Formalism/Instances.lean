module

public import Definitions.Core.Formalism
public import Mathlib.Data.Fin.VecNotation

/-! The instances `Honest`, `Counter` and `counted` that the lemmas of `CKSize` and `Counting` are stated at. -/

@[expose] public section

namespace Verity.Formalism

/-- Gate 0 is an input, and gates 1, 2 and 3 each read the gate before. -/
abbrev C : Circuit where
  N := 4
  w _ := 1
  isInput := ![true, false, false, false]
  reads := ![∅, {0}, {1}, {2}]
  reads_lt := by decide
  input_reads := by decide
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3}

/-- Gate 0 is registered. -/
abbrev d : RegisteredCircuit := .ofCircuit C {0} (by decide)

/-- The proof blocks `{1, 2}` and `{3}`. -/
abbrev Honest : Partition d.D 2 := ⟨![none, some 0, some 0, some 1], by decide⟩

/-- The three operation gates in one proof block. -/
abbrev Counter : Partition d.D 1 := ⟨![none, some 0, some 0, some 0], by decide⟩

/-- The counted gates: the three operation gates. -/
abbrev counted : Finset (Fin d.D.N) := {1, 2, 3}

/-- Gates 1 and 2 weigh 1 and gate 3 weighs 2, so each proof block of `Honest` weighs 2. -/
abbrev weight : Fin d.D.N → ℕ := ![0, 1, 1, 2]

/-- `weight` with gate 3 at 3, so the second proof block of `Honest` weighs 3. -/
abbrev heavy : Fin d.D.N → ℕ := ![0, 1, 1, 3]

end Verity.Formalism
