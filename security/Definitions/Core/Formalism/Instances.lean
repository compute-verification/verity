module

public import Definitions.Core.Formalism
public import Mathlib.Data.Fin.VecNotation

/-!
# A `(c, k)`-partition and a partition that is none (trusted)

The instances the honest and nontrivial lemmas of `Verity.Guarantees.CKSize` and `Verity.Guarantees.Counting` are
stated at (`Security/Definitions/Core/Guarantees/Formalism/Witnesses.lean`), on one small Boolean declaration `d`: gate
0 is its one registered input, and gates 1, 2 and 3 each read the gate before.

* `Honest`: the proof blocks `{1, 2}` and `{3}`, a `(2, 1)`-partition whose second proof block has one gate, under
  `c = 2`. The first reads only the registered gate 0, and gate 3 reads gate 2, the second's one unregistered input bit.
* `Counter`: the three operation gates in one proof block, above `c = 2`, so `|C| = 3` is above `c · B = 2`.
* `counted`, the three operation gates, at the weights `weight` (gates 1 and 2 at 1, gate 3 at 2), so each proof block
  of `Honest` weighs 2, or `heavy` (gate 3 at 3), so its second weighs 3.
-/

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
