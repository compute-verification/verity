module

public import Definitions.Slowdown.Formalism.Runs
public import Mathlib.Data.Fin.VecNotation
public import Mathlib.Tactic.FinCases

/-! The small Boolean instances at which the training formalism v2's honest and nontrivial lemmas are stated. -/

@[expose] public section

namespace Slowdown.Formalism

open Verity

/-- The decoder that outputs nothing. -/
def silent : Decoder (List Bool) where
  run _ _ _ := none
  mono _ _ _ _ _ _ h := nomatch h
  prefixFree _ _ _ _ _ _ h _ _ := nomatch h

namespace Honest

/-- Gate 0 is the weight `θ` and gate 1 a data bit `x`, both inputs; gate 2 computes `θ' = θ ⊕ x` and gate 3, the token,
`θ' ∧ x`. -/
abbrev C : Circuit where
  N := 4
  w _ := 1
  isInput := ![true, true, false, false]
  reads := ![∅, ∅, {0, 1}, {1, 2}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 2 then v 0 ^^^ v 1 else if g = 3 then v 2 &&& v 1 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · rfl
    · simp [hv 0 (by decide), hv 1 (by decide)]
    · simp [hv 1 (by decide), hv 2 (by decide)]
  outputs := {3}

/-- Gate 0 is registered. -/
abbrev d : RegisteredCircuit := .ofCircuit C {0} (by decide)

/-- Two blocks, gate 2 and gate 3. -/
abbrev P : Partition d.D 2 := ⟨![none, none, some 0, some 1], by decide⟩

end Honest

namespace Chain

/-- Gate 0 is the weight `θ`, an input; gate 1 computes `θ'`, a copy of it, and gates 2, 3 and 4, the token, each a copy
of the one before. -/
abbrev C : Circuit where
  N := 5
  w _ := 1
  isInput := ![true, false, false, false, false]
  reads := ![∅, {0}, {1}, {2}, {3}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else if g = 2 then v 1 else if g = 3 then v 2 else if g = 4 then v 3 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using hv 1 (by decide)
    · simpa using hv 2 (by decide)
    · simpa using hv 3 (by decide)
  outputs := {4}

/-- Gate 0 is registered. -/
abbrev d : RegisteredCircuit := .ofCircuit C {0} (by decide)

/-- The new weight is gate 1, and the one token holds gates 2, 3 and 4. -/
abbrev A : Annotation d (Fin 1) := ⟨{1}, ⟨fun _ => {2, 3, 4}⟩⟩

/-- Four blocks, one gate each. -/
abbrev blocks : Partition d.D 4 := ⟨![none, some 0, some 1, some 2, some 3], by decide⟩

/-- One block of all four gates. -/
abbrev whole : Partition d.D 1 := ⟨![none, some 0, some 0, some 0, some 0], by decide⟩

/-- Where a block's work on the token begins: one entry for every block. -/
abbrev entry (_ : Finset (Fin d.D.N)) (_ : Fin 1) : Unit := ()

/-- The witness blocks: the token's three gates, one block, which reads the new weight alone. -/
abbrev witnesses : Finset (Finset (Fin d.D.N)) := { {2, 3, 4} }

/-- A witness block that skips gate 3: gates 2 and 4, which read the new weight and gate 3's value. -/
abbrev skipping : Finset (Finset (Fin d.D.N)) := { {2, 4} }

/-- The run on the all-zero inputs. -/
abbrev r : Run := ⟨d, 1, A, fun _ => 0⟩

/-- The methods considered: the one run. -/
abbrev methods : Set Run := {r}

end Chain

end Slowdown.Formalism
