import Definitions.Accounting.ServingBlocks
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# Serving-block layouts: the honest instance and the counter-examples (trusted)

The instances the honest and nontrivial lemmas of `Accounting.Guarantees.ServingBlocks` are stated at
(`Security/Definitions/Accounting/Guarantees/ServingBlocks/Witnesses.lean`), each a small actual circuit
(`Accounting.Placed`) with its registered inputs, counted operations, servers, wipes, dark leaves and caps.

`Honest` is one server, leaf `0`, wiped at tick `1` of `[0, 2)`, and one dark leaf, leaf `1`. Gate 0 is a registered
input on the server, gate 1 copies it at tick 0, gate 2 negates gate 1 at tick 1 and gate 3 is a constant at the dark
leaf. Its two counted operations, each of weight 1, are gate 2's, computed by gates 1 and 2 and so split across the wipe
(the operation `OperationBodies` allows and the boundary term pays for), and gate 3's. The serving blocks are `{1}` and
`{2}`, each counted with itself as its one inference block, and the dark leaves hold `{3}`. `q(c, k) = c` is a per-block
cap at every size, as each operation's body holds its representative and weighs 1. It rests on no run and no `art:`. It
is not the deployed partition at the onsite reference numbers, whose premises are Layer 2's envelopes with their
probabilities.

The counter-examples, each on one server:
* `Counter`: one constant gate computing one operation of weight 1, with the cap `q(c, 0) = 0` and `q(c, k) = c` for
  `k > 0`. A cap at `k = 0` says a block reading nothing computes nothing, which the constant refutes.
* `Straddle`: two constants, gate 1 reading gate 0 after the wipe, and one operation computed by both. No block of one
  gate computes it, so the cap is `0`, and only the boundary term `ω_max μ k_S` covers it.
* `Detached`: `Straddle` with gate 1 reading nothing, so the operation takes in a gate it never reads.
-/

namespace Accounting.ServingBlocks

open Verity

/-- The cuts of `[0, J)` at every tick: the windows `[j, j + 1)`. -/
abbrev ticks (J : ℕ) : Cuts J := ⟨J, fun i => i.val, fun _ _ h => h, rfl, rfl⟩

namespace Honest

/-- Gate 0 is an input, gate 1 copies it, gate 2 negates gate 1 and gate 3 is a constant. -/
abbrev C : Circuit where
  N := 4
  w _ := 1
  isInput := ![true, false, false, false]
  reads := ![∅, {0}, {1}, ∅]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else if g = 2 then ~~~ v 1 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using congrArg (~~~ ·) (hv 1 (by decide))
    · rfl
  outputs := {2, 3}

/-- Gates 0 to 2 on the server, leaf 0, gate 2 after its wipe at tick 1; gate 3 at the dark leaf, leaf 1. -/
abbrev A : Placed (Fin 2) where
  toCircuit := C
  x := ![0, 0, 0, 1]
  t := ![0, 0, 1, 0]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := {0}

abbrev dark : Finset (Fin 2) := {1}

abbrev X : Unit → Finset (Fin 2) := fun _ => {0}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- Gate 2's operation, computed by gates 1 and 2, and gate 3's, each of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {2, 3}
  body := ![∅, ∅, {1, 2}, {3}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

noncomputable abbrev q : ℕ → ℕ → ℝ := fun c _ => c

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

end Honest

namespace Counter

/-- One computed gate, a constant. -/
abbrev C : Circuit where
  N := 1
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := absurd h (Finset.notMem_empty _)
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {0}

abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t _ := 0
  cell _ := 0
  cell_inj a b _ := Subsingleton.elim a b

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 1 := fun _ => ticks 1

/-- The gate's operation, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {0}
  body v := {v}
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- A cap at `k = 0` says a block reading nothing computes nothing. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun c k => if k = 0 then 0 else c

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

end Counter

namespace Straddle

/-- Two constants, gate 1 reading gate 0. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput _ := false
  reads := ![∅, {0}]
  reads_lt := by decide
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {1}

/-- Both gates on one server, gate 1 after its wipe at tick 1. -/
abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t := ![0, 1]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- Gate 1's operation, computed by both gates and so split across the wipe, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {1}
  body := ![∅, {0, 1}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

end Straddle

namespace Detached

/-- Two constants, reading nothing. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput _ := false
  reads _ := ∅
  reads_lt _ _ h := absurd h (Finset.notMem_empty _)
  input_reads _ h := absurd h (by decide)
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {1}

/-- Both gates on one server, gate 1 after its wipe at tick 1. -/
abbrev A : Placed Unit where
  toCircuit := C
  x _ := ()
  t := ![0, 1]
  cell g := g.val
  cell_inj _ _ h := Fin.ext (congrArg (fun p => p.2.1) h)

abbrev θ : Finset (Fin A.N) := ∅

abbrev dark : Finset Unit := ∅

abbrev X : Unit → Finset Unit := fun _ => {()}

abbrev κ : Unit → Cuts 2 := fun _ => ticks 2

/-- An operation represented by gate 1 whose body takes in gate 0, which no gate of it reads, of weight 1. -/
abbrev J : Counted A.toCircuit where
  V := {1}
  body := ![∅, {0, 1}]
  ω _ := 1
  V_computed := by decide
  ω_nonneg _ _ := zero_le_one

/-- No block of one gate computes the operation wholly. -/
noncomputable abbrev q : ℕ → ℕ → ℝ := fun _ _ => 0

noncomputable abbrev inf : (S : Block κ) → Fin 1 → Finset (Fin A.N) := fun S _ => block A X κ S

end Detached

end Accounting.ServingBlocks
