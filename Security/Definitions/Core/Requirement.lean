import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# A requirement's execution, placement and store (trusted)

The objects a declared circuit's requirement (`Definitions/Accounting/Requirement.lean`) is read against, which every
resource audit shares:

* **Execution.** A scheduled execution is the values it produces (`ran`: the executed prefix, a request's rows up to
  its end and the engine's lag), the engine step at which it produces each (`step`), and the values (`value`, as bits).
  Registered values are there before step 0.
* **Placement.** A placed execution also says at which device each value is produced or, an input, stored (`leaf`: a
  leaf of the actual circuit's tree of locations, `Accounting.Placed`, the device, never an SM, an SMSP or a unit), and
  at which tick each device begins each engine step (`ticks`). A value is placed at the tick its device begins its step
  (`Placement.tick`).
* **Store.** The registered store with the memory map beside it (I3, `memory-map/v1` of
  `verity/core/protocols/remote_bandwidth/memory_map.py`): `N` registered blocks, the fresh indices each serving step
  wrote and those it freed, and each block's payload. A block is written once, complete and encoded, so its payload is
  a function of its index. Step `s`'s record takes effect after step `s`'s reads: a block is written in the record of
  the step before its first encoded read, readable from the step after, and freed in the record of a step no earlier
  than its values' last structural reader.

A step is always an **engine step**: the engine's `s`-th step, never a device tick.

`verity/ml/circuits/requirement.py` (`Placement`, `Store`) and its code-Lean port
`verity/core/primitives/requirement/CircuitRequirement.lean` are their executable forms.
-/

namespace Accounting.Requirement

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
  (`Accounting.Placed.x`), the device, never an SM, an SMSP or a unit -/
  leaf : G → L
  /-- the tick at which device `x` begins engine step `s` -/
  ticks : L → ℕ → ℕ

/-- The tick at which a value is placed: the one at which its device begins its engine step (`Accounting.Placed.t`). -/
def Placement.tick {L G : Type} (p : Placement L G) (v : G) : ℕ := p.ticks (p.leaf v) (p.step v)

/-- Each device of `X` begins its engine steps at strictly increasing ticks. -/
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

end Accounting.Requirement
