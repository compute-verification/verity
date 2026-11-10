module

public import Definitions.Core.Expand
public import Mathlib.Data.Fin.VecNotation

/-!
# The bit-level expansion: an AND of two two-bit gates (trusted)

The instance the honest and nontrivial lemmas of `Verity.Guarantees.ExpandSize` and `Verity.Guarantees.ExpandAndSize`
are stated at (`Security/Definitions/Core/Guarantees/Expand/Witnesses.lean`).

`AndPair`: gates 0 and 1 are two-bit input gates, gate 0 registered, and gate 2 is their bitwise AND, the one unit of
`P`. Expanded, each of gate 2's two bits reads the four input bits and is the AND of one bit of gate 0 and one of gate 1,
so the expansion has two AND gates where the declaration has two bits of computed gates, and its unit reads gate 1's
two unregistered bits where the declaration's unit reads gate 1, of width 2: both bounds hold with equality.
-/

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
