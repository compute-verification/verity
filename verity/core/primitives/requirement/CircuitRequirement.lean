import Std.Data.HashSet

/-!
# A declared circuit's requirement, computed

`Accounting.Requirement` (`Security/Definitions/Accounting/Requirement.lean`) on one instance: a declared circuit's
wiring, one scheduled execution of it and a layout of its values (`Instance`), and the audit's store (`Store`). This is
code, so it requires nothing of `Security/`: `Properties.Accounting.Requirement` states that each function here is the
definition it computes, on the wiring, execution, layout and store an instance denotes, and
`Properties.Proofs.Accounting.Requirement` proves it. `requirement-difftest` serves these functions to the difftest, which
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

/-- The same members, either way round. -/
def sameSet (a b : List Nat) : Bool :=
  let ha := Std.HashSet.ofList a
  let hb := Std.HashSet.ofList b
  a.all hb.contains && b.all ha.contains

/-- A declared circuit's wiring, one scheduled execution of it and a layout of its values, value by value. -/
structure Instance where
  /-- each value's width in bits (`Wiring.width`); the values are `0 … n - 1`, `n` this array's length -/
  width : Array Nat
  /-- the registered values (`Wiring.registered`) -/
  registered : List Nat
  /-- each gate's operands (`Wiring.ops`) -/
  ops : Array (List Nat)
  /-- the operands each gate uses at this execution's values (`Wiring.uses` at `Execution.value`) -/
  uses : Array (List Nat)
  /-- the step at which the execution produces each value (`Execution.step`), `none` for one it produces only by
  registration or not at all -/
  step : Array (Option Nat)
  /-- each value, as its bits (`Execution.value`) -/
  value : Array Nat
  /-- each value's block, a store index (`Layout.block`) -/
  block : Array Nat
  /-- each value's first bit in its block (`Layout.offset`) -/
  offset : Array Nat

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

namespace Instance

variable (i : Instance)

def widthOf (v : Nat) : Nat := i.width.getD v 0
def opsOf (g : Nat) : List Nat := i.ops.getD g []
def usesOf (g : Nat) : List Nat := i.uses.getD g []
def stepOf (v : Nat) : Nat := (i.step.getD v none).getD 0
def valueOf (v : Nat) : Nat := i.value.getD v 0
def blockOf (v : Nat) : Nat := i.block.getD v 0
def offsetOf (v : Nat) : Nat := i.offset.getD v 0

/-- `Execution.ran`: the values the execution produces, the registered ones and those with a step. -/
def ran : List Nat :=
  let reg := Std.HashSet.ofList i.registered
  (List.range i.width.size).filter fun v => reg.contains v || (i.step.getD v none).isSome

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

/-- `Packs B S`: each value lies within its block of `B` bits, and two in one block don't overlap. -/
def packs (B : Nat) (S : List Nat) : Bool :=
  S.all (fun v => decide (i.endOf v ≤ B)) && i.apart (S.mergeSort i.before)

/-- `Carries s S`: each value of `S` is its block's payload from its offset. -/
def carries (s : Store) (S : List Nat) : Bool :=
  S.all fun v => decide (s.payloadOf (i.blockOf v) / 2 ^ i.offsetOf v % 2 ^ i.widthOf v = i.valueOf v)

/-- `StoreMatches s B t`. -/
def storeMatches (s : Store) (B t : Nat) : Bool :=
  let S := i.live t
  sameSet (s.resident t) (i.blocks S) && i.packs B S && i.carries s S

/-- `PlanMatches plan B t`. -/
def planMatches (plan : List Nat) (B t : Nat) : Bool :=
  let S := i.reads t
  sameSet plan (i.blocks S) && i.packs B S

end Instance

end CircuitRequirement
