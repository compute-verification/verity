module

public import Spot.Wires
public import Definitions.Spot.Dataflow

/-! The dataflow a wiring check accepts is acyclic (`WiresAcyclic`), and what a statement about committed rows reads
of a wiring. -/

@[expose] public section

open Spot.Wires

namespace Properties.Spot

/-- A wiring's dataflow: units `(template, unit)` over the output rows `(template, unit, output)`; each unit writes its
template's outputs and reads the writer's rows of its wired rows. -/
def wiresDataflow (c : Wiring) : Population.Dataflow (String × Int) (String × Int × String) where
  input := ∅
  reads y := ((c.wires.flatMap Wire.entries).filterMap fun e =>
    if (e.1.1, (e.1.2.1 : Int)) = y then some e.2 else none).toFinset
  writes x := ((((c.templates.lookup x.1).map (·.outs)).getD []).map fun o => (x.1, x.2, o.1)).toFinset

/-- The dataflow of a wiring `check` accepts is acyclic. -/
def WiresAcyclic : Prop :=
  ∀ (c : Wiring) (S : Summary), check c = .ok S → (wiresDataflow c).Acyclic

/-- **The mis-wired units** of a wiring under a comparison of rows: the reader units `(template, unit)` with a row
`(template, unit, port, row)` that a wire covers (`Wire.entries`) and that is not its writer's output row
`(template, unit, output)`, as `same` compares the two. -/
def miswired (c : Wiring) (same : String × Nat × String × Nat → String × Int × String → Prop) :
    Set (String × Nat) :=
  {y | ∃ e ∈ c.wires.flatMap Wire.entries, (e.1.1, e.1.2.1) = y ∧ ¬ same e.1 e.2}

/-- **A unit is correct in the wired population**: it is correct on its own rows (`correct`: its output rows hold
its function of its input rows), and each of its rows that a wire covers is its writer's row (it is not in
`miswired`). Its output rows then hold its function of its writers' output rows, which is what
`Population.Dataflow.Correct` asks of a unit of `wiresDataflow`. -/
def WiredCorrect (c : Wiring) (same : String × Nat × String × Nat → String × Int × String → Prop)
    (correct : String × Nat → Prop) (y : String × Nat) : Prop :=
  correct y ∧ y ∉ miswired c same

namespace Wires

/-- A template whose second unit reads its first unit's output, with its order stated: `check` accepts it. -/
def honest : Wiring :=
  { templates := [("g", ⟨2, [("x", ⟨1, 1⟩)], [("o", 1)]⟩)], cuts := [],
    wires := [{ reader := "g", port := "x", writer := "g", out := "o", start := 1, radix := [1], base := 0, step := [0],
                cut0 := 0, cuts := 1 }],
    order := some [("g", 0, 1), ("g", 1, 1)] }

/-- A unit wired to read its own output: its dataflow has a cycle. -/
def selfRead : Wiring :=
  { templates := [("g", ⟨1, [("x", ⟨1, 1⟩)], [("o", 1)]⟩)], cuts := [],
    wires := [{ reader := "g", port := "x", writer := "g", out := "o", start := 0, radix := [1], base := 0, step := [0],
                cut0 := 0, cuts := 1 }] }

end Wires

end Properties.Spot
