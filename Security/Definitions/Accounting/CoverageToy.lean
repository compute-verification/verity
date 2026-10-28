import Definitions.Accounting.Slowdown

/-!
# The floor's toy instance

The task and computations at which `Nci.Guarantees.CapabilityFloor`'s witnesses
(`Definitions.Accounting.Guarantees.Coverage`) are stated: one input, no output positions (so every run reaches the
target, all of whose outputs are acceptable), and the run of the empty actual circuit.
-/

namespace Accounting.Slowdown.Toy

open Accounting

/-- The empty actual circuit, at one leaf. -/
def empty : Placed Unit where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅
  x := fun g => g.elim0
  t := fun g => g.elim0
  cell := fun g => g.elim0
  cell_inj := fun g => g.elim0

/-- The run of the empty circuit. -/
def run : Actual.Run Unit where
  A := empty
  val := fun g => g.elim0
  faithful := fun g => g.elim0
  held := fun _ _ => ∅
  holding := ⟨fun _ _ _ h => absurd h (Finset.notMem_empty _), fun g => g.elim0, fun g => g.elim0⟩

/-- The task: one input, and every output acceptable. -/
def task : Task Unit (Fin 0 → List Bool) where
  Ω := Unit
  draw := fun _ => ((), Set.univ)

/-- The computation that runs the empty circuit on its input. -/
def comp : Unit → Actual.Run Unit := fun _ => run

/-- No output positions. -/
def pos : Fin 0 → Unit × ℕ × ℕ := fun i => i.elim0

/-- Every computation has the covered structure. -/
def covered : (Unit → Actual.Run Unit) → Unit → Prop := fun _ _ => True

/-- A cost of `5` for every run. -/
def cost : Actual.Run Unit → ℕ := fun _ => 5

/-- A cost of `0` for every run. -/
def free : Actual.Run Unit → ℕ := fun _ => 0

end Accounting.Slowdown.Toy
