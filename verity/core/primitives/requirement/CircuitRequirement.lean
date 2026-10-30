import Std.Data.HashSet

/-!
# A declared circuit's requirement, computed

`Accounting.Requirement` (`Security/Definitions/Accounting/Requirement.lean`) on one instance: a declared circuit's
wiring, one scheduled execution of it and a layout of its values (`Instance`), the audit's store (`Store`), and a
placement of the execution on devices (`Placement`), whose regions, ingress and credits are PoUW's over the placed
execution (`Definitions/Accounting/Guarantees/RequirementPouw.lean`). A step is always an engine step, never a device tick.
This is code, so it requires nothing of `Security/`: `Properties.Accounting.Requirement` states that each function here
is the definition it computes, on the wiring, execution, layout, store and placement an instance denotes, and
`Proofs.Properties.Accounting.Requirement` proves it. `requirement-difftest` serves these functions to the difftest, which
holds them to `verity.ml.circuits.requirement` (`tests/test_requirement_difftest.py`).

An instance's values are `0 … n - 1`, `n` the length of `width`, and the operands a gate uses are those at this
execution's values (`Wiring.uses` at `Execution.value`). An array read past its end reads as empty, or 0. A set is a list,
ascending where a function returns one, and membership goes through a hash set built once per call, so that a step costs
time linear in the wiring, and `packs` sorts the values by block and offset and compares neighbours: a registry row of
tens of thousands of gates runs in seconds.
-/

namespace CircuitRequirement

/-- The members of `l`, each once, ascending. -/
def distinct (l : List Nat) : List Nat := (Std.HashSet.ofList l).toList.mergeSort

/-- Every member of `a` is one of `b`. -/
def subsetOf (a b : List Nat) : Bool :=
  let hb := Std.HashSet.ofList b
  a.all hb.contains

/-- A declared circuit's wiring, one scheduled execution of it and a layout of its values, value by value. -/
structure Instance where
  /-- each value's width in bits (`Wiring.width`); the values are `0 … n - 1`, `n` this array's length -/
  width : Array Nat
  /-- the registered values (`Wiring.registered`) -/
  registered : List Nat
  /-- the run's inputs (`Wiring.inputs`): a request's tokens, an input gate wherever it is placed -/
  inputs : List Nat
  /-- each gate's operands (`Wiring.ops`) -/
  ops : Array (List Nat)
  /-- the operands each gate uses at this execution's values (`Wiring.uses` at `Execution.value`) -/
  uses : Array (List Nat)
  /-- the engine step (never a device tick) at which the execution produces each value (`Execution.step`), `none` for one
  it produces only by registration or not at all -/
  step : Array (Option Nat)
  /-- each value, as its bits (`Execution.value`) -/
  value : Array Nat
  /-- each value's block, a store index (`Store.Placement.block`) -/
  block : Array Nat
  /-- each value's first bit in its block (`Store.Placement.offset`) -/
  offset : Array Nat
  /-- each block's first encoded read, by store index (`Store.Placement.encoded`): 0, as past its end, for a block
  encoded once complete -/
  encoded : Array Nat := #[]

/-- The audit's store (`Accounting.Requirement.Store`): the registered blocks, the fresh indices each step wrote and those
it freed, and each block's payload. -/
structure Store where
  registered : Nat
  written : Array (List Nat)
  freed : Array (List Nat)
  payload : Array Nat

namespace Store

variable (s : Store)

def writtenAt (t : Nat) : List Nat := s.written.getD t []
def freedAt (t : Nat) : List Nat := s.freed.getD t []
def payloadOf (j : Nat) : Nat := s.payload.getD j 0

/-- `Store.resident`: the blocks resident when step `t` begins. -/
def resident (t : Nat) : List Nat :=
  let freed := Std.HashSet.ofList ((List.range t).flatMap s.freedAt)
  distinct ((List.range s.registered ++ (List.range t).flatMap s.writtenAt).filter fun j => !freed.contains j)

end Store

/-- An instance's execution placed on devices (`Accounting.Requirement.Placement`): each value's device, and the tick at
which each device begins each engine step, device by device. -/
structure Placement where
  leaf : Array Nat
  ticks : Array (Array Nat)

namespace Placement

variable (p : Placement)

def leafOf (v : Nat) : Nat := p.leaf.getD v 0
def ticksOf (x s : Nat) : Nat := (p.ticks.getD x #[]).getD s 0

end Placement

namespace Instance

variable (i : Instance)

def widthOf (v : Nat) : Nat := i.width.getD v 0
def opsOf (g : Nat) : List Nat := i.ops.getD g []
def usesOf (g : Nat) : List Nat := i.uses.getD g []
def stepOf (v : Nat) : Nat := (i.step.getD v none).getD 0
def valueOf (v : Nat) : Nat := i.value.getD v 0
def blockOf (v : Nat) : Nat := i.block.getD v 0
def offsetOf (v : Nat) : Nat := i.offset.getD v 0
def encodedOf (j : Nat) : Nat := i.encoded.getD j 0

/-- `Execution.ran`: the values the execution produces, the registered ones and those with a step. -/
def ran : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  (List.range i.width.size).filter fun v => reg.contains v || (i.step.getD v none).isSome

/-- `Wiring.Topological`: each gate's operands come before it. -/
def topological : Bool := (List.range i.ops.size).all fun g => (i.opsOf g).all (· < g)

/-- `Execution.Follows`. -/
def follows : Bool :=
  let reg := Std.HashSet.ofList i.registered
  let r := i.ran
  let ran := Std.HashSet.ofList r
  i.registered.all ran.contains && r.all fun g =>
    (i.usesOf g).all (i.opsOf g).contains &&
      (i.opsOf g).all fun v => ran.contains v && (reg.contains v || decide (i.stepOf v ≤ i.stepOf g))

/-- `live t`: produced before `t` (or registered), and an operand of a gate run at step `t` or later. -/
def live (t : Nat) : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  let r := i.ran
  let later := Std.HashSet.ofList ((r.filter fun g => decide (t ≤ i.stepOf g)).flatMap i.opsOf)
  r.filter fun v => (reg.contains v || decide (i.stepOf v < t)) && later.contains v

/-- `reads t`: produced before `t` (or registered), and used by a gate run at step `t`. -/
def reads (t : Nat) : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  let r := i.ran
  let used := Std.HashSet.ofList ((r.filter fun g => decide (i.stepOf g = t)).flatMap i.usesOf)
  r.filter fun v => (reg.contains v || decide (i.stepOf v < t)) && used.contains v

def bits (S : List Nat) : Nat := (S.map i.widthOf).sum
def liveBits (t : Nat) : Nat := i.bits (i.live t)
def readBits (t : Nat) : Nat := i.bits (i.reads t)

/-- `held` over the window of `len` steps from `start`. -/
def held (start len : Nat) : Nat := ((List.range' start len).map i.liveBits).sum

/-- `read` over the window of `len` steps from `start`. -/
def read (start len : Nat) : Nat := ((List.range' start len).map i.readBits).sum

def blocks (S : List Nat) : List Nat := distinct (S.map i.blockOf)
def bitsIn (S : List Nat) (j : Nat) : Nat := i.bits (S.filter fun v => decide (i.blockOf v = j))
def liveBlocks (t : Nat) : List Nat := i.blocks (i.live t)
def liveIn (t j : Nat) : Nat := i.bitsIn (i.live t) j
def readBlocks (t : Nat) : List Nat := i.blocks (i.reads t)
def readIn (t j : Nat) : Nat := i.bitsIn (i.reads t) j

/-- One past a value's last bit in its block. -/
def endOf (v : Nat) : Nat := i.offsetOf v + i.widthOf v

/-- `u` sorts no later than `v`: by block, then first bit, then end. -/
def before (u v : Nat) : Bool :=
  decide (i.blockOf u < i.blockOf v) || decide (i.blockOf u = i.blockOf v) &&
    (decide (i.offsetOf u < i.offsetOf v) || decide (i.offsetOf u = i.offsetOf v) && decide (i.endOf u ≤ i.endOf v))

/-- Each value in a sorted list ends before the next begins when the two share a block. -/
def apart : List Nat → Bool
  | u :: tl@(v :: _) => (decide (i.blockOf u ≠ i.blockOf v) || decide (i.endOf u ≤ i.offsetOf v)) && apart tl
  | _ => true

/-- `Packs b S`: each value lies within its block of `b` bits, and two in one block don't overlap. -/
def packs (b : Nat) (S : List Nat) : Bool :=
  S.all (fun v => decide (i.endOf v ≤ b)) && i.apart (S.mergeSort i.before)

/-- `Carries s S`: each value of `S` is its block's payload from its offset. -/
def carries (s : Store) (S : List Nat) : Bool :=
  S.all fun v => decide (s.payloadOf (i.blockOf v) / 2 ^ i.offsetOf v % 2 ^ i.widthOf v = i.valueOf v)

/-- `incomplete t`: the blocks of the values the execution produces, but the registered ones, at step `t` or later or
with their block's first encoded read after `t`. -/
def incomplete (t : Nat) : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  distinct ((i.ran.filter fun v =>
    !reg.contains v && (decide (t ≤ i.stepOf v) || decide (t < i.encodedOf (i.blockOf v)))).map i.blockOf)

/-- The values of `S` in blocks complete and encoded when step `t` begins. -/
def complete (S : List Nat) (t : Nat) : List Nat :=
  let inc := Std.HashSet.ofList (i.incomplete t)
  S.filter fun v => !inc.contains (i.blockOf v)

/-- `credited t`: the live values in complete blocks. -/
def credited (t : Nat) : List Nat := i.complete (i.live t) t

/-- `creditedReads t`: the values step `t` reads from complete blocks. -/
def creditedReads (t : Nat) : List Nat := i.complete (i.reads t) t

def creditedBits (t : Nat) : Nat := i.bits (i.credited t)
def creditedReadBits (t : Nat) : Nat := i.bits (i.creditedReads t)

/-- `StoreMatches s b t`: the credited blocks are resident, and other blocks may be. -/
def storeMatches (s : Store) (b t : Nat) : Bool :=
  let C := i.credited t
  subsetOf (i.blocks C) (s.resident t) && i.packs b (i.live t) && i.carries s C

/-- `PlanMatches plan b t`: the plan holds the complete blocks step `t` reads, and may hold others. -/
def planMatches (plan : List Nat) (b t : Nat) : Bool :=
  subsetOf (i.blocks (i.creditedReads t)) plan && i.packs b (i.reads t)

/-- `Placement.tick`: the tick at which a value's device begins its engine step. -/
def tick (p : Placement) (v : Nat) : Nat := p.ticksOf (p.leafOf v) (i.stepOf v)

/-- The gates the execution computes: the values it produces but the registered ones and the run's inputs. -/
def computed : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  let inp := Std.HashSet.ofList i.inputs
  i.ran.filter fun g => !reg.contains g && !inp.contains g

/-- Whether `p` places the execution: a device for every value it produces, and a tick at its step on its device for
every gate it computes. A placement that doesn't would read device 0 or tick 0 for it, and credit it silently. -/
def places (p : Placement) : Bool :=
  i.ran.all (· < p.leaf.size) && i.computed.all fun g => decide (i.stepOf g < (p.ticks.getD (p.leafOf g) #[]).size)

/-- `region X [lo, hi)`: the gates the execution computes on the devices `X` at a tick of `[lo, hi)`. -/
def region (p : Placement) (X : List Nat) (lo hi : Nat) : List Nat :=
  let xs := Std.HashSet.ofList X
  i.computed.filter fun g => xs.contains (p.leafOf g) && decide (lo ≤ i.tick p g) && decide (i.tick p g < hi)

/-- `stepRegion X w`: the same at the engine steps `[start, start + len)`. -/
def stepRegion (p : Placement) (X : List Nat) (start len : Nat) : List Nat :=
  let xs := Std.HashSet.ofList X
  i.computed.filter fun g => xs.contains (p.leafOf g) && decide (start ≤ i.stepOf g) && decide (i.stepOf g < start + len)

/-- `into S`: the values outside `S` that a gate of `S` uses. -/
def into (S : List Nat) : List Nat :=
  let s := Std.HashSet.ofList S
  distinct ((S.flatMap i.usesOf).filter fun a => !s.contains a)

/-- `ingress X [lo, hi)`: the bits of the values placed outside `X` that the region uses. -/
def ingress (p : Placement) (X : List Nat) (lo hi : Nat) : Nat :=
  let xs := Std.HashSet.ofList X
  i.bits ((i.into (i.region p X lo hi)).filter fun a => !xs.contains (p.leafOf a))

/-- `work π R` at the pricing `price`, a value's price read as 0 past its end. -/
def work (price : Array Nat) (R : List Nat) : Nat := (R.map fun g => price.getD g 0).sum

/-- The work of the region less the marked values: PoUW's credits over the placed execution, by `RequirementCredits`,
for the wiring over `Fin n` with these values. -/
def credits (p : Placement) (price : Array Nat) (X marked : List Nat) (lo hi : Nat) : Nat :=
  let m := Std.HashSet.ofList marked
  work price ((i.region p X lo hi).filter fun g => !m.contains g)

end Instance

end CircuitRequirement
