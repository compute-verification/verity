import Lean.Data.Json
import Flock.Canon
import Difftest
import CircuitRequirement

/-!
# The requirement difftest: `CircuitRequirement` behind `Difftest.serve`

`requirement-difftest` serves `table` to the core difftest harness (`verity/core/difftest`), whose Python half
(`tests/test_requirement_difftest.py`) holds it to `verity.ml.circuits.requirement`. Its one function, `requirement`,
takes an instance, value by value, with the store, a plan per step, the block size and the number of steps:

    {"width": [w], "registered": [v], "ops": [[v]], "uses": [[v]], "step": [s | null], "value": [x],
     "block": [j], "offset": [o], "store": {"registered": N, "written": [[j]], "freed": [[j]], "payload": [hex]},
     "plans": [[j]], "block_bits": B, "steps": T}

A payload is lowercase hex (Python writes no integer of more than 4,300 digits, a block's payload among them). It refuses
an instance that isn't one (a field missing or of the wrong kind, a per-value list not of one entry per value, a
registered value that isn't one, a plan count that isn't `T`) and an execution that doesn't follow its wiring. For each
step `t < T` it returns the live and read values, their bits, the blocks that hold them with the bits in each, the
resident blocks, and whether the store and the plan are `W`'s (`storeMatches`, `planMatches`), with `held` and `read`
over the `T` steps.
-/

namespace CircuitRequirement.Difftest

open Lean (Json toJson)

def nats (j : Json) (ctx : String) : Flock.V (List Nat) := do
  (← Flock.J.asArr j ctx).toList.mapM (Flock.J.asNat · ctx)

def optNat (j : Json) (ctx : String) : Flock.V (Option Nat) :=
  match j with
  | .null => pure none
  | _ => some <$> Flock.J.asNat j ctx

def hexNat (j : Json) (ctx : String) : Flock.V Nat := do
  (← Flock.J.asStr j ctx).toList.foldlM (init := 0) fun n c =>
    if '0' ≤ c ∧ c ≤ '9' then pure (16 * n + (c.toNat - '0'.toNat))
    else if 'a' ≤ c ∧ c ≤ 'f' then pure (16 * n + (c.toNat - 'a'.toNat + 10))
    else throw s!"{ctx}: not lowercase hex"

/-- A list with one entry per value, each decoded by `f`. -/
def perValue {α : Type} (j : Json) (k : String) (n : Nat) (f : Json → String → Flock.V α) : Flock.V (Array α) := do
  let a ← (← Flock.J.arr j k "input").mapM (f · k)
  if a.size != n then throw s!"{k} has {a.size} entries, not one per value ({n})"
  return a

def store (j : Json) : Flock.V Store := do
  let s ← Flock.J.get j "store" "input"
  return {
    registered := ← Flock.J.nat s "registered" "store"
    written := ← (← Flock.J.arr s "written" "store").mapM (nats · "store.written")
    freed := ← (← Flock.J.arr s "freed" "store").mapM (nats · "store.freed")
    payload := ← (← Flock.J.arr s "payload" "store").mapM (hexNat · "store.payload") }

def decode (j : Json) : Flock.V Instance := do
  let width ← (← Flock.J.arr j "width" "input").mapM (Flock.J.asNat · "width")
  let n := width.size
  let registered ← nats (← Flock.J.get j "registered" "input") "registered"
  if registered.any (n ≤ ·) then throw "registered: a value the circuit doesn't have"
  return {
    width, registered
    ops := ← perValue j "ops" n nats
    uses := ← perValue j "uses" n nats
    step := ← perValue j "step" n optNat
    value := ← perValue j "value" n Flock.J.asNat
    block := ← perValue j "block" n Flock.J.asNat
    offset := ← perValue j "offset" n Flock.J.asNat }

def natsJson (l : List Nat) : Json := .arr (l.map toJson).toArray

def byBlock (i : Instance) (S : List Nat) : Json :=
  .arr ((i.blocks S).map fun j => Json.arr #[toJson j, toJson (i.bitsIn S j)]).toArray

def stepJson (i : Instance) (s : Store) (plan : List Nat) (B t : Nat) : Json :=
  let live := i.live t
  let reads := i.reads t
  Json.mkObj [("live", natsJson live), ("live_bits", toJson (i.bits live)), ("live_blocks", byBlock i live),
    ("reads", natsJson reads), ("read_bits", toJson (i.bits reads)), ("read_blocks", byBlock i reads),
    ("resident", natsJson (s.resident t)), ("store_matches", toJson (i.storeMatches s B t)),
    ("plan_matches", toJson (i.planMatches plan B t))]

def requirement (j : Json) : Flock.V Json := do
  let i ← decode j
  let s ← store j
  let plans ← (← Flock.J.arr j "plans" "input").mapM (nats · "plans")
  let B ← Flock.J.nat j "block_bits" "input"
  let T ← Flock.J.nat j "steps" "input"
  if plans.size != T then throw s!"plans has {plans.size} entries, not one per step ({T})"
  if !i.follows then throw "the execution does not follow its wiring (`Execution.Follows`)"
  return Json.mkObj [("held", toJson (i.held 0 T)), ("read", toJson (i.read 0 T)),
    ("steps", .arr ((List.range T).map fun t => stepJson i s (plans.getD t []) B t).toArray)]

def table := _root_.Difftest.table [("requirement", requirement)]

end CircuitRequirement.Difftest
