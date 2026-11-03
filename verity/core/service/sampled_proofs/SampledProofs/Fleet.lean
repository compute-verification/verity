/-
# The proof system's fleet: each stream's operator, and the operator's halt (`service.Fleet`'s rules)

A fleet is the challenger's streams across protocols and levels, each attached to its operator (the identity it
registers for) with the failure budget it runs under. Its rule is R5-all: at the failure that brings a stream of an
operator to its budget (its b-th), the operator halts. `run` decides one op on what the fleet's rules hold (a `State`),
as `Stream.run` decides a stream's:

* at that failure the fleet halts the operator and names every stream attached to it, which `Stream.halt` then halts:
  each refuses every op from then on, and every outcome of it not yet final ends as `halted`, never `accept`; outcomes
  already final stay;
* the fleet refuses a registration to any stream of a halted operator, and refuses to attach another stream to it;
* each registration the fleet admits has a fleet position, the number it admitted before, and a halt's cause names the
  stream whose failure halted the operator and the fleet position then.

A state, as JSON: `size`, the registrations admitted; `streams`, each `{operator, budget}` in the order attached, a stream
known by its index there; `halted`, each `{operator, cause}`, the cause `{operator, stream, at}`.

An op is an object with `op` and the fields it reads:

* `attach` `{operator, budget}`: a stream joins the fleet, and the decision is its index;
* `register` `{stream}`: the fleet admits a registration to the stream, and the decision is its fleet position;
* `failures` `{stream, failures}`: the stream's failures after an outcome. At its budget the decision is
  `{cause, streams}`, the halt's cause and the index of every stream of the operator, in order; below it, null.

A decision is `{"refused": codes}` or `{"ok": value}`: `unattached` (no stream of that index) or `halted` (the operator
halted: an op naming it or one of its streams is refused and changes nothing). An error is an op or a state this port
doesn't read.
-/
import SampledProofs.Stream

namespace SampledProofs.Fleet

open Lean (Json)
open Flock (V)
open SampledProofs.Stream (Decision)

/-- A stream of the fleet: its operator, and the failure budget it runs under. -/
structure Attached where
  operator : String
  budget : Nat

structure State where
  /-- the registrations the fleet admitted: the next one's fleet position -/
  size : Nat := 0
  /-- the fleet's streams, by index, in the order attached -/
  streams : Array Attached := #[]
  /-- the halted operators, each with its halt's cause -/
  halted : Array (String × Json) := #[]

/-- The operator's halt's cause, once it halted. -/
def haltOf (s : State) (o : String) : Option Json := (s.halted.find? (·.1 == o)).map (·.2)

/-- The index of every stream attached to the operator, in order. -/
def streamsOf (s : State) (o : String) : Array Nat :=
  (Array.range s.streams.size).filter fun i => s.streams[i]?.map (·.operator) == some o

/-- A halt's cause: the operator, the stream whose failure halted it, and the fleet position then. -/
def causeOf (o : String) (i at_ : Nat) : Json :=
  Json.mkObj [("operator", .str o), ("stream", Lean.toJson i), ("at", Lean.toJson at_)]

/-- A halt's decision: its cause, and the streams it halts. -/
def haltValue (cause : Json) (streams : Array Nat) : Json :=
  Json.mkObj [("cause", cause), ("streams", natsJson streams)]

private def str? : Json → Option String
  | .str s => some s
  | _ => none

def Attached.ofJson (j : Json) : V Attached := do
  if keys? j != some #["budget", "operator"] then throw "a stream is exactly {operator, budget}"
  let some operator := str? (getD j "operator") | throw "an operator is a string"
  let some budget := nat? (getD j "budget") | throw "a failure budget is a positive integer"
  if budget == 0 then throw "a failure budget is a positive integer"
  return { operator, budget }

def State.ofJson (j : Json) : V State := do
  if keys? j != some #["halted", "size", "streams"] then throw "a fleet's state is exactly {size, streams, halted}"
  let some size := nat? (getD j "size") | throw "size is a count"
  let .arr ss := getD j "streams" | throw "streams is a list"
  let streams ← ss.mapM Attached.ofJson
  let .arr hs := getD j "halted" | throw "halted is a list"
  let halted ← hs.mapM fun h => do
    if keys? h != some #["cause", "operator"] then throw "a halt is exactly {operator, cause}"
    let some o := str? (getD h "operator") | throw "an operator is a string"
    let .obj c := getD h "cause" | throw "a halt's cause is an object"
    pure (o, Json.obj c)
  return { size, streams, halted }

def State.toJson (s : State) : Json := Json.mkObj [
  ("size", Lean.toJson s.size),
  ("streams", Json.arr (s.streams.map fun a => Json.mkObj [("operator", .str a.operator), ("budget", Lean.toJson a.budget)])),
  ("halted", Json.arr (s.halted.map fun h => Json.mkObj [("operator", .str h.1), ("cause", h.2)]))]

/-- A step: it ends early with a decision that changes nothing, or errs on an op it doesn't read. -/
abbrev M := StateT State (ExceptT Decision V)

def no {α} (code : String) : M α := throwThe Decision (.refused #[code])
def bad {α} (msg : String) : M α := throwThe String msg

/-- The stream the op names, and what it is attached as, unless its operator halted. -/
def attached (op : Json) : M (Nat × Attached) := do
  let s ← get
  match nat? (getD op "stream") with
  | none => no "unattached"
  | some i => match s.streams[i]? with
    | none => no "unattached"
    | some a => if (haltOf s a.operator).isSome then no "halted" else pure (i, a)

def attach (op : Json) : M Decision := do
  let s ← get
  match getD op "operator", nat? (getD op "budget") with
  | .str operator, some budget =>
    if budget == 0 then bad "a failure budget is a positive integer"
    else if (haltOf s operator).isSome then no "halted"
    else do
      set { s with streams := s.streams.push { operator, budget } }
      return .ok (Lean.toJson s.streams.size)
  | .str _, none => bad "a failure budget is a positive integer"
  | _, _ => bad "an operator is a string"

def register (op : Json) : M Decision := do
  let _ ← attached op
  let s ← get
  set { s with size := s.size + 1 }
  return .ok (Lean.toJson s.size)

def failures (op : Json) : M Decision := do
  let (i, a) ← attached op
  let s ← get
  match nat? (getD op "failures") with
  | none => bad "failures is a count"
  | some f =>
    if f < a.budget then return .ok Json.null
    else do
      let cause := causeOf a.operator i s.size
      set { s with halted := s.halted.push (a.operator, cause) }
      return .ok (haltValue cause (streamsOf s a.operator))

def step (op : Json) : M Decision :=
  match getD op "op" with
  | .str "attach" => attach op
  | .str "register" => register op
  | .str "failures" => failures op
  | _ => bad "an op is one of attach, register, failures"

/-- The op's decision on `s`, and what the fleet's rules hold after it: `s` itself where the op ends early. -/
def run (s : State) (op : Json) : V (Decision × State) := do
  match ← ((step op).run s).run with
  | .ok (d, s') => pure (d, s')
  | .error d => pure (d, s)

end SampledProofs.Fleet
