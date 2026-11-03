import Definitions.Accounting.Slowdown

/-!
# The floor's toy instance

The task and computations at which `Nci.Guarantees.CapabilityFloor`'s witnesses
(`Definitions.Accounting.Guarantees.Coverage`) are stated. A hidden coin `ω` sets the target: the outputs whose one
position holds `[ω]` are acceptable, and the input says nothing. The covered structure is at least two gates, the cost
a run's gate count. A guess, the one-gate run writing a fixed bit at the output position, lacks the structure and
reaches the target on half the coins; no computation does better, since a position holds one gate.
-/

namespace Accounting.Slowdown.Toy

open Accounting

/-- The one-gate actual circuit at one leaf: an input of width one at cell `0`, tick `0`. -/
def oneGate : Placed Unit where
  N := 1
  w := fun _ => 1
  isInput := fun _ => true
  reads := fun _ => ∅
  reads_lt := fun _ _ h => absurd h (Finset.notMem_empty _)
  input_reads := fun _ _ => rfl
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := ∅
  x := fun _ => ()
  t := fun _ => 0
  cell := fun _ => 0
  cell_inj := fun g g' _ => Subsingleton.elim g g'

/-- The guess `b`: the one-gate circuit's run, its gate holding `b`. -/
def guess (b : Bool) : Actual.Run Unit where
  A := oneGate
  val := fun _ => BitVec.ofBool b
  faithful := fun _ h => by cases h
  held := fun _ _ => Finset.univ
  holding := ⟨fun _ _ _ _ => ⟨rfl, Or.inl rfl⟩, fun _ _ => Finset.mem_univ _,
    fun _ _ h => absurd h (Finset.notMem_empty _)⟩

/-- The task: a hidden coin `ω`, one input, and the outputs whose one position holds `[ω]` acceptable. -/
def task : Task Unit (Fin 1 → List Bool) where
  Ω := Bool
  draw := fun ω => ((), {y | y 0 = [ω]})

/-- The computation that guesses `1` on its input. -/
def comp : Unit → Actual.Run Unit := fun _ => guess true

/-- The output position: the leaf's cell `0` at tick `0`. -/
def pos : Fin 1 → Unit × ℕ × ℕ := fun _ => ((), 0, 0)

/-- The covered structure: at least two gates. -/
def covered : (Unit → Actual.Run Unit) → Unit → Prop := fun C x => 2 ≤ (C x).A.N

/-- The cost: a run's gate count. -/
def cost : Actual.Run Unit → ℕ := fun R => R.A.N

end Accounting.Slowdown.Toy
