/-
The proof service's run of one registration, from its public check through its verdict, as `service.Stream` makes it on a Spec
whose selection rule `Select.Rule` reads: what the challenger holds once a delivered proof's outcome is decided, and every
check the stream ran on it, in the stream's order.

* `register`: the public-items rule on the boundary values (`Service.checkPublic`), the population (a positive integer),
  then R1–R7 (`Registration.check`) against the challenger's copies under the law the rule names over that population
  (`Select.Rule.lawOver`, whatever law the copies carry), and the receipt (`registration.receipt`);
* `select`: the draw of record on the bytes of the challenger's coins (`Select.select`), held to the registration and
  the expanded law (`Draw.checkDraw`);
* `deliver`: one or more session records' SHA-512 (`service._DIGEST`);
* `outcome`: the verdicts' hold (`Service.checkVerdicts`), the public outputs of the first verifier of record, and the
  audit record (`Record.record`) over the draw's proved units, whose acceptance is the symbol.

A step's refusal is `{"step", "codes"}`; a decided run is `{"receipt", "draw", "read", "symbol", "public", "audit"}`,
the audit record as the text of Python's `canonical_json`.
`Stream.outcome` holds its own outcome to this run (`verifier.run`). A run whose proof came after the proof deadline is
the clock's (`late`), and a two-stage run with registered interiors is not read here.
-/
import SampledProofs.OneStage
import SampledProofs.Select
import SampledProofs.Service

namespace SampledProofs.Run

open Lean (Json)
open Flock (V)

def RECEIPT : String := "verity/registration-receipt/v1"

structure Input where
  params : OneStage.Params
  record : Json
  rule : Select.Rule
  values : Array Service.Value
  declared : Array String
  challenger : String
  received : String
  source : ByteArray
  closure : Option (Array (Int × Array Int))
  sessions : Array Json
  verdicts : Array (String × Json)
  outputs : Array String
  statement : Json
  delta : Json
  provedUnits : Option (Array Int)
  instances : Option Int
  served : Json
  sessionRoots : Json

private def refused (step : String) (codes : Array String) : Json :=
  Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]

private def hexDigit (c : Char) : Bool := ('0' ≤ c && c ≤ '9') || ('a' ≤ c && c ≤ 'f')

/-- `service._DIGEST.match`: 128 lowercase hex digits, which `$` lets a final newline follow. -/
def digest? : Json → Bool
  | .str s => (s.length == 128 || (s.length == 129 && s.endsWith "\n")) && (s.toList.take 128).all hexDigit
  | _ => false

/-- The receipt the stream writes for an accepted registration: its digest, the challenger and its place in the log. -/
def receipt (record : Json) (challenger received : String) : V Json := do
  return Json.mkObj [("format", Json.str RECEIPT), ("registration", Json.str (← Registration.digestHex record)),
    ("verifier", Json.str challenger), ("received", Json.str received)]

/-- The run, step by step; an error where the Python raises. -/
def run (i : Input) : V Json := do
  let v := i.params.verifier
  let pub ← Service.checkPublic i.record (Json.str v.scheme) (← Registration.queryDigest v.query) i.values i.declared
  if !pub.codes.isEmpty then return refused "register" pub.codes
  let some n := nat? (getD i.record "population") | return refused "register" #["population"]
  if n == 0 then return refused "register" #["population"]
  let law ← i.rule.lawOver n
  let v := { v with law }
  let codes := Registration.check i.record v
  if !codes.isEmpty then return refused "register" codes
  let receipt ← receipt i.record i.challenger i.received
  let derived ← Draw.derive v.program v.query law v.workTable
  let (draw, _, read) ← Select.select i.rule v.program v.query v.workTable i.closure i.source
  let codes := Draw.checkDraw draw (getD i.record "population") derived
  if !codes.isEmpty then return refused "select" codes
  if i.sessions.isEmpty || !i.sessions.all digest? then return refused "deliver" #["sessions"]
  let held ← Service.checkVerdicts i.verdicts i.sessions i.outputs
  if !held.codes.isEmpty then return refused "outcome" held.codes
  let ofRecord := (i.verdicts.qsort (·.1 < ·.1)).filter (truthy <| getD ·.2 "of_record")
  let shown := if i.outputs.isEmpty then Json.mkObj [] else getD ((ofRecord[0]?.map (·.2)).getD Json.null) "public"
  let proved := i.provedUnits.getD (Draw.proved draw)
  let instances := i.instances.getD proved.size
  let rec_ ← Record.record (← Record.inputOf (Json.mkObj [("registration", i.record), ("receipt", receipt),
    ("draw", draw), ("law", derived), ("proved_units", Json.arr (proved.map Lean.toJson)),
    ("instances", Lean.toJson instances), ("statement", i.statement), ("verdicts", Json.mkObj i.verdicts.toList),
    ("delta", i.delta), ("sessions", Json.arr i.sessions), ("served", i.served),
    ("session_roots", i.sessionRoots)]))
  let accepted := jeq (getD rec_ "accepted") (.bool true)
  return Json.mkObj [("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
    ("symbol", Json.str (if accepted then "accept" else "reject")), ("public", if accepted then shown else Json.mkObj []),
    ("audit", Json.str (← PyJson.canonical rec_))]

end SampledProofs.Run
