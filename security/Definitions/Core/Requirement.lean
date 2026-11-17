module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-! The execution, placement and store a registered circuit's requirement is read against, a step always an engine
step. -/

@[expose] public section

namespace ResourceAccounting.Requirement

/-- A scheduled execution of a circuit over values `G`. -/
structure Execution (G : Type) where
  /-- the values it produces, the registered ones among them -/
  ran : Finset G
  /-- the engine step at which it produces each value (an engine step, never a device tick) -/
  step : G → ℕ
  /-- each value, as its bits -/
  value : G → ℕ

/-- A scheduled execution placed on devices `L`: the device of each value, and the tick at which each device begins each
engine step. -/
structure Placement (L G : Type) extends Execution G where
  /-- the device at which each value is produced or, an input, stored: a leaf of the actual circuit's tree of locations
  (`ResourceAccounting.Placed.x`), the device, never an SM, an SMSP or a unit -/
  leaf : G → L
  ticks : L → ℕ → ℕ

/-- The tick at which a value is placed: the one at which its device begins its engine step (`ResourceAccounting.Placed.t`). -/
def Placement.tick {L G : Type} (p : Placement L G) (v : G) : ℕ := p.ticks (p.leaf v) (p.step v)

def Placement.Paced {L G : Type} (p : Placement L G) (X : Finset L) : Prop := ∀ x ∈ X, StrictMono (p.ticks x)

/-- The audit's store: the registered store and the memory map beside it (I3). -/
structure Store where
  /-- the registered store's blocks `N`, store indices `0 … N - 1` -/
  registered : ℕ
  /-- the fresh store indices the record of engine step `s` wrote (memory-map/v1's `written`) -/
  written : ℕ → Finset ℕ
  /-- the store indices the record of engine step `s` freed (memory-map/v1's `freed`, by the indices their slots held,
  and the blocks its writes overwrote) -/
  freed : ℕ → Finset ℕ
  /-- each block's payload, as its bits -/
  payload : ℕ → ℕ

/-- **The blocks resident when step `s` begins**: the registered ones and those written in the records of steps before
`s`, less those freed in them. -/
def Store.resident (σ : Store) (s : ℕ) : Finset ℕ :=
  (Finset.range σ.registered ∪ (Finset.range s).biUnion σ.written) \ (Finset.range s).biUnion σ.freed

end ResourceAccounting.Requirement
