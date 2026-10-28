import Definitions.Accounting.ServingBlocks
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# Serving-block layouts: the honest instance and the counter-examples (trusted)

The instances the honest and nontrivial lemmas of `Accounting.Guarantees.ServingBlocks` are stated at
(`Security/Definitions/Accounting/Guarantees/ServingBlocks/Witnesses.lean`). Each is a small Boolean declaration with its
registered input gates, a partition into serving blocks and the dark remainder, its inference blocks, counted operations
and caps. In each, the inference blocks are the serving blocks themselves (`refl`), one inference block per serving block,
and a constant is a registered input gate.

`Honest` is one server with one wipe, and one dark leaf. Gate 0 is a registered input. Gate 1 copies it before the wipe,
gate 2 negates gate 1 after it, and gate 3 copies gate 0 at the dark leaf. Its two counted operations, each of weight 1,
are gate 2's and gate 3's. Gate 2's is computed by gates 1 and 2 and so split across the wipe: the operation
`OperationBodies` allows and the boundary term pays for. The serving blocks are `{1}` and `{2}`, and the dark remainder
is `{3}`. `q(c, k) = c` is a per-block cap at every size, as each operation's body holds its representative and weighs 1.
It is a small faithful instance of a serving-block layout, which this structural bound accepts as its honest instance:
it rests on no run and no `art:`, and the deployed premises are Layer 2's envelopes, each with its probability.

The counter-examples, each on one server:
* `Counter`: one gate reading only a registered input gate, computing one operation of weight 1, with the cap
  `q(c, 0) = 0` and `q(c, k) = c` for `k > 0`. A cap at `k = 0` says a block reading no unregistered bit computes
  nothing, which this gate refutes. `P₂` puts its gate in the first of two units, and `away` sends its one inference
  block to the second: a parent map that is no refinement.
* `Straddle`: gate 1 reads the registered gate 0 before the wipe and gate 2 reads gate 1 after it, and one operation is
  computed by gates 1 and 2. No block of one gate computes it, so the cap is `0`, and only the boundary term
  `ω_max μ k_S` covers it.
* `Detached`: `Straddle` with gate 2 reading gate 0, so the operation takes in gate 1, which no gate of it reads.
-/

namespace Accounting.ServingBlocks

open Verity

/-- A partition refines itself: each unit is its own one sub-unit. -/
def refl {C : Circuit} {n : ℕ} (P : Partition C n) : P.Refines P := ⟨id, fun g => by cases P.unit g <;> rfl⟩

namespace Honest

/-- Gate 0 is an input, gate 1 copies it, gate 2 negates gate 1 and gate 3 copies gate 0. -/
abbrev C : Circuit where
  N := 4
  w _ := 1
  isInput := ![true, false, false, false]
  reads := ![∅, {0}, {1}, {0}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 0 then 0 else if g = 2 then ~~~ v 1 else v 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using congrArg (~~~ ·) (hv 1 (by decide))
    · simpa using hv 0 (by decide)
  outputs := {2, 3}

/-- Gate 0 is registered. -/
abbrev d : Registered := .ofCircuit C {0} (by decide)

/-- The serving blocks `{1}` (before the wipe) and `{2}` (after it), and the dark remainder `{3}`. -/
abbrev P : Partition d.D 3 := ⟨![none, some 0, some 1, some 2], by decide⟩

abbrev S : Finset (Fin 3) := {0, 1}

abbrev r : P.Refines P := refl P

/-- Gate 2's operation, computed by gates 1 and 2, and gate 3's, each of weight 1. -/
abbrev J : Counted d.D where
  V := {2, 3}
  body := ![∅, ∅, {1, 2}, {3}]
  ω _ := 1
  V_computed := by decide
  rep_mem := by decide
  ω_nonneg _ _ := zero_le_one

noncomputable abbrev q : ℕ → ℕ → ℝ := fun c _ => c

end Honest

namespace Counter

/-- Gate 0 is an input, and gate 1 reads it. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput := ![true, false]
  reads := ![∅, {0}]
  reads_lt := by decide
  input_reads := by decide
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {1}

/-- Gate 0 is registered. -/
abbrev d : Registered := .ofCircuit C {0} (by decide)

/-- One serving block, `{1}`. -/
abbrev P : Partition d.D 1 := ⟨![none, some 0], by decide⟩

abbrev S : Finset (Fin 1) := {0}

abbrev r : P.Refines P := refl P

/-- Gate 1's operation, of weight 1. -/
abbrev J : Counted d.D where
  V := {1}
  body v := {v}
  ω _ := 1
  V_computed := by decide
  rep_mem _ _ := Finset.mem_singleton_self _
  ω_nonneg _ _ := zero_le_one

/-- A cap at `k = 0` says a block reading no unregistered bit computes nothing. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun c k => if k = 0 then 0 else c

/-- Gate 1 in the first of two units. -/
abbrev P₂ : Partition d.D 2 := ⟨![none, some 0], by decide⟩

/-- A parent map sending `P`'s one unit to `P₂`'s second unit, which holds no gate. -/
abbrev away : Fin 1 → Fin 2 := fun _ => 1

end Counter

namespace Straddle

/-- Gate 0 is an input, gate 1 reads it and gate 2 reads gate 1. -/
abbrev C : Circuit where
  N := 3
  w _ := 1
  isInput := ![true, false, false]
  reads := ![∅, {0}, {1}]
  reads_lt := by decide
  input_reads := by decide
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {2}

/-- Gate 0 is registered. -/
abbrev d : Registered := .ofCircuit C {0} (by decide)

/-- The serving blocks `{1}` (before the wipe) and `{2}` (after it). -/
abbrev P : Partition d.D 2 := ⟨![none, some 0, some 1], by decide⟩

abbrev S : Finset (Fin 2) := {0, 1}

abbrev r : P.Refines P := refl P

/-- Gate 2's operation, computed by gates 1 and 2 and so split across the wipe, of weight 1. -/
abbrev J : Counted d.D where
  V := {2}
  body := ![∅, ∅, {1, 2}]
  ω _ := 1
  V_computed := by decide
  rep_mem := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

end Straddle

namespace Detached

/-- Gate 0 is an input, and gates 1 and 2 read it. -/
abbrev C : Circuit where
  N := 3
  w _ := 1
  isInput := ![true, false, false]
  reads := ![∅, {0}, {0}]
  reads_lt := by decide
  input_reads := by decide
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {2}

/-- Gate 0 is registered. -/
abbrev d : Registered := .ofCircuit C {0} (by decide)

/-- The serving blocks `{1}` (before the wipe) and `{2}` (after it). -/
abbrev P : Partition d.D 2 := ⟨![none, some 0, some 1], by decide⟩

abbrev S : Finset (Fin 2) := {0, 1}

abbrev r : P.Refines P := refl P

/-- An operation represented by gate 2 whose body takes in gate 1, which no gate of it reads, of weight 1. -/
abbrev J : Counted d.D where
  V := {2}
  body := ![∅, ∅, {1, 2}]
  ω _ := 1
  V_computed := by decide
  rep_mem := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

end Detached

end Accounting.ServingBlocks
