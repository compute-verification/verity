import Definitions.Nci.Formalism.Runs
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# The training formalism v2: the instances its witnesses are stated at (trusted)

The instances the honest and nontrivial lemmas of `Nci.Formalism.Guarantees` are stated at
(`Security/Definitions/Nci/Guarantees/Formalism/Witnesses.lean`). Each is a small Boolean declaration whose gate 0 is its
one registered input, the weight `θ`.

* `Honest`, serving one token: gate 1 is a data bit `x`, gate 2 the new weight `θ' = θ ⊕ x` and gate 3 the token,
  `θ' ∧ x`. Each gate is a block of one gate: gate 2's reads the data bit beyond `θ`, and gate 3's the data bit and the
  new weight, its description. Fit's lemmas are stated here.
* `Chain`, a training step and a token of three gates: gate 1 is the new weight `θ'`, a copy of `θ`, and gates 2, 3 and 4
  the token, each a copy of the one before, a chain. `blocks` puts each gate in a block of its own, a partition into
  inference blocks at `(c, k) = (1, 1)`; only gate 2's block computes the token from its inputs, reading the new weight,
  and gates 3 and 4's each read the token's gate before. `whole` is one block of all four gates, which is no partition
  into inference blocks at `c = 1` and prices them at `1 / 4`. A1's values here are `entry`, where a block's work on the
  token begins, the token's gates it holds; `state`, the token's state there, the token's values that work reads; and
  `update`, the update bits a block reads, the new weight when it reads it. The lemmas of the counting lemma, Slowdown
  and Security are stated here, Security's with a run on the all-zero inputs that is the only one of the methods
  considered.

`silent` is the decoder that outputs nothing, so every total update is `⊤`, within `D = ⊤` bits.
-/

namespace Nci.Formalism

open Verity
open scoped NNReal

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
abbrev d : Declaration := ⟨C, {0}, by decide⟩

/-- The new weight is gate 1, and the one token holds gates 2, 3 and 4. -/
abbrev A : Annotation d (Fin 1) := ⟨{1}, ⟨fun _ => {2, 3, 4}⟩⟩

/-- Four blocks, one gate each. -/
abbrev blocks : Partition d.D 4 := ⟨![none, some 0, some 1, some 2, some 3], by decide⟩

/-- One block of all four gates. -/
abbrev whole : Partition d.D 1 := ⟨![none, some 0, some 0, some 0, some 0], by decide⟩

/-- Where a block's work on the token begins: the token's gates it holds. -/
abbrev entry (I : Finset (Fin d.D.N)) (T : Fin 1) : Finset (Fin d.D.N) := I ∩ A.tokens.holds T

/-- The token's state at an entry: the token's values that the work from it reads. -/
noncomputable abbrev state (T : Fin 1) (S : Finset (Fin d.D.N)) : ℝ≥0 := ((d.D.into S ∩ A.tokens.holds T).card : ℝ≥0)

/-- The update bits a block reads: the new weight, when it reads it. -/
noncomputable abbrev update (I : Finset (Fin d.D.N)) : ℝ≥0 := ((d.D.into I ∩ A.θ').card : ℝ≥0)

/-- The run on the all-zero inputs. -/
abbrev r : Run := ⟨d, 1, A, fun _ => 0⟩

/-- The methods considered: the one run. -/
abbrev methods : Set Run := {r}

end Chain

end Nci.Formalism
