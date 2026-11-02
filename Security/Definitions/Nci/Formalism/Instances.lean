import Definitions.Nci.Formalism.Runs
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# The training formalism v1: the honest instance and the counter-example (trusted)

The instances the honest and nontrivial lemmas of `Nci.Formalism.Guarantees` are stated at
(`Security/Definitions/Nci/Guarantees/Formalism/Witnesses.lean`). Each is a small Boolean declaration whose gate 0 is its
one registered input, the weight `θ`, with an annotation of its training that has one token, a partition into blocks,
and a run on the all-zero inputs that is the only one of the methods considered.

* `Honest`, a training step and a token on its result: gate 1 is a data bit `x`, gate 2 the new weight `θ' = θ ⊕ x` and
  gate 3 the token, `θ' ∧ x`, so the token's one gate is its dependent work. Each gate is a block: gate 2's reads the
  data bit beyond `θ` and gate 3's the data bit and `θ'`, a `(1, 2)`-partition, and A1 holds at `c = 1` with
  `u(x) = x`, as the block that does the token's dependent work reads `θ'`, one computed value.
* `Counter`: gate 1 is the new weight `θ'`, a copy of `θ`, and gate 2 the token, a copy of `θ'`. The block holds gates 1
  and 2 and reads nothing beyond `θ`, so it does all of the token's dependent work reading no computed value: the
  slowdown at `c = 2` is at most `1` for every `k`, and A1 fails at that block.

`silent` is the decoder that outputs nothing, so every total update is `⊤`, within `D = ⊤` bits.
-/

namespace Nci.Formalism

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
abbrev d : Declaration := ⟨C, {0}, by decide⟩

/-- The new weight is gate 2, and the one token holds gate 3. -/
abbrev A : Annotation d (Fin 1) := ⟨{2}, ⟨fun _ => {3}⟩⟩

/-- Two blocks, gate 2 and gate 3. -/
abbrev P : Partition d.D 2 := ⟨![none, none, some 0, some 1], by decide⟩

/-- The run on the all-zero inputs. -/
abbrev r : Run := ⟨d, 1, A, fun _ => 0⟩

/-- The methods considered: the one run. -/
abbrev methods : Set Run := {r}

end Honest

namespace Counter

/-- Gate 0 is the weight `θ`, an input; gate 1 computes `θ'`, a copy of it, and gate 2, the token, a copy of `θ'`. -/
abbrev C : Circuit where
  N := 3
  w _ := 1
  isInput := ![true, false, false]
  reads := ![∅, {0}, {1}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else if g = 2 then v 1 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using hv 1 (by decide)
  outputs := {2}

/-- Gate 0 is registered. -/
abbrev d : Declaration := ⟨C, {0}, by decide⟩

/-- The new weight is gate 1, and the one token holds gate 2. -/
abbrev A : Annotation d (Fin 1) := ⟨{1}, ⟨fun _ => {2}⟩⟩

/-- One block, gates 1 and 2. -/
abbrev P : Partition d.D 1 := ⟨![none, some 0, some 0], by decide⟩

/-- The run on the all-zero inputs. -/
abbrev r : Run := ⟨d, 1, A, fun _ => 0⟩

/-- The methods considered: the one run. -/
abbrev methods : Set Run := {r}

end Counter

end Nci.Formalism
