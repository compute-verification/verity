/-
The proof service's run of one registration, from its public check through its verdict, as `service.Stream` makes it on a Spec
whose selection rule `Select.Rule` reads: what the challenger holds once a delivered proof's outcome is decided, and every
check the stream ran on it, in the stream's order.

* `register`: the public-items rule on the boundary values (`Service.checkPublic`), the population (a positive integer),
  then R1–R7 (`Registration.check`) against the challenger's copies under the law the rule names over that population
  (`Select.Rule.lawOver`, whatever law the copies carry), and the receipt (`registration.receipt`);
* `select`: the draw of record on the bytes of the challenger's coins (`Select.select`), held to the registration and
  the expanded law (`Draw.checkDraw`);
* `interior` (`Stream.register_interiors`, two-stage only): the drawn RUs' interiors, after the draw and before any proof:
  a two-stage rule that drew some RU, the public-items rule on the interior values, a population of the K proved RUs (an
  integer), R1–R7 against the interiors' own copies under the run's query and `subset:K` over them, and the receipt;
* `deliver`: one or more session records' SHA-512 (`service._DIGEST`), and under a two-stage rule that drew some RU, the
  interiors registered first (`interior`);
* `outcome`: the verdicts' hold (`Service.checkVerdicts`), the public outputs of the first verifier of record, and the
  audit record (`Record.record`) over the draw's proved units, with the interiors' over all of theirs, the sessions' roots
  split between the two by the interior values' names under a two-stage rule; the symbol accepts when every audit record
  does.

A step's refusal is `{"step", "codes"}`; a decided run is `{"receipt", "draw", "read", "interior", "symbol", "public",
"audit"}`, each audit record as the text of Python's `canonical_json`, `interior` its receipt and audit record or null.
`Stream.outcome` holds its own outcome to this run (`verifier.run`). A run whose proof came after the proof deadline is
the clock's (`late`).
-/
import SampledProofs.OneStage
import SampledProofs.Select
import SampledProofs.Service

namespace SampledProofs.Run

open Lean (Json)
open Flock (V)

def RECEIPT : String := "verity/registration-receipt/v1"

/-- The interiors' registration as `Stream.register_interiors` takes it: their copies (the population program over the
proved RUs, the ports, leaf layer and anchors, and the log its R6 and R7 read; the run sets the query and the law), the
record, and the log's length at its receipt. -/
structure Interior where
  params : OneStage.Params
  record : Json
  received : String

/-- `values` are the boundary values (every value under a one-stage rule), `innerValues` a two-stage Spec's interior
ones. -/
structure Input where
  params : OneStage.Params
  record : Json
  rule : Select.Rule
  values : Array Service.Value
  innerValues : Array Service.Value
  declared : Array String
  challenger : String
  received : String
  source : ByteArray
  closure : Option (Array (Int × Array Int))
  interior : Option Interior
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

/-- `subset:K` over `[0, K)`, every unit drawn: the interiors' law and draw. -/
private def every (k : Nat) : Json × Json :=
  let law := Json.mkObj [("law", "subset"), ("k", Lean.toJson k)]
  (law, Json.mkObj [("law", "subset"), ("k", Lean.toJson k), ("population", Lean.toJson k),
    ("units", natsJson (Array.range k))])

/-- The run, step by step; an error where the Python raises. -/
def run (i : Input) : V Json := do
  let v := i.params.verifier
  let qd ← Registration.queryDigest v.query
  let pub ← Service.checkPublic i.record (Json.str v.scheme) qd i.values i.declared
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
  let drawn := Draw.proved draw
  let staged := i.rule matches .twoStage _
  let inner ← match i.interior with
    | none => pure none
    | some ir => do
      if !staged then return refused "interior" #["rule"]
      if drawn.isEmpty then return refused "interior" #["interior-empty"]
      let pub ← Service.checkPublic ir.record (Json.str v.scheme) qd i.innerValues i.declared
      if !pub.codes.isEmpty then return refused "interior" pub.codes
      if int? (getD ir.record "population") != some drawn.size then return refused "interior" #["interior-population"]
      let (ilaw, idraw) := every drawn.size
      let codes := Registration.check ir.record { ir.params.verifier with law := ilaw, query := v.query }
      if !codes.isEmpty then return refused "interior" codes
      pure (some (ir, ← SampledProofs.Run.receipt ir.record i.challenger ir.received, ilaw, idraw))
  if i.sessions.isEmpty || !i.sessions.all digest? then return refused "deliver" #["sessions"]
  if staged && !drawn.isEmpty && inner.isNone then return refused "deliver" #["interior"]
  let held ← Service.checkVerdicts i.verdicts i.sessions i.outputs
  if !held.codes.isEmpty then return refused "outcome" held.codes
  let ofRecord := (i.verdicts.qsort (·.1 < ·.1)).filter (truthy <| getD ·.2 "of_record")
  let shown := if i.outputs.isEmpty then Json.mkObj [] else getD ((ofRecord[0]?.map (·.2)).getD Json.null) "public"
  let proved := i.provedUnits.getD drawn
  let instances := i.instances.getD proved.size
  let audit (reg rcpt d l : Json) (units : Json) (served roots : Json) : V Json := do
    Record.record (← Record.inputOf (Json.mkObj [("registration", reg), ("receipt", rcpt), ("draw", d), ("law", l),
      ("proved_units", units), ("instances", Lean.toJson instances), ("statement", i.statement),
      ("verdicts", Json.mkObj i.verdicts.toList), ("delta", i.delta), ("sessions", Json.arr i.sessions),
      ("served", served), ("session_roots", roots)]))
  let (roots, innerRoots) ← match staged, i.sessionRoots with
    | false, r | _, r@Json.null => pure (r, Json.null)
    | true, .obj _ =>
      let inner (p : String × Json) := i.innerValues.any (·.name == p.1)
      let kvs := entries i.sessionRoots
      pure (Json.mkObj (kvs.filter (!inner ·)).toList, Json.mkObj (kvs.filter inner).toList)
    | true, _ => throw "session_roots is not an object"
  let rec_ ← audit i.record receipt draw derived (Json.arr (proved.map Lean.toJson)) i.served roots
  let innerRec ← match inner with
    | none => pure none
    | some (ir, r, ilaw, idraw) => do
      let a ← audit ir.record r idraw ilaw (natsJson (Array.range drawn.size)) Json.null innerRoots
      pure (some (r, a))
  let ok (a : Json) := jeq (getD a "accepted") (.bool true)
  let accepted := ok rec_ && innerRec.all (ok ·.2)
  let interior ← match innerRec with
    | none => pure Json.null
    | some (r, a) => do pure (Json.mkObj [("receipt", r), ("audit", Json.str (← PyJson.canonical a))])
  return Json.mkObj [("receipt", receipt), ("draw", draw), ("read", Lean.toJson read), ("interior", interior),
    ("symbol", Json.str (if accepted then "accept" else "reject")), ("public", if accepted then shown else Json.mkObj []),
    ("audit", Json.str (← PyJson.canonical rec_))]

end SampledProofs.Run
