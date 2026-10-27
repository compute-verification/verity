/-
The proof service's run of one registration, from its public check through its verdict, as `service.Stream` makes it on a Spec
whose selection rule `Select.Rule` reads: what the challenger holds once a delivered proof's outcome is decided, and every
check the stream ran on it, in the stream's order.

* `register`: the public-items rule on the Spec's values (`Service.checkPublic`), the population (a positive integer),
  the rule's rate law over it (`Select.Rule.over`: `rate`, a count of none or more than the population), then R1–R7
  (`Registration.check`) against the challenger's copies under the law the rule names over that population
  (`Select.Rule.lawOver`, whatever law the copies carry), a law whose drawn count varies (`countVaries`: `shapes`), a
  program and query whose population is not the registration's units in order (`Slots.popOk`: `population-order`), and
  the receipt (`registration.receipt`);
* `select`: the draw of record on the bytes of the challenger's coins (`Select.select`, under the rule over the program's
  population, `Select.resolve`); the coins are what the run's source says (`coinsHold`, code `coins`): from the
  verifier's own coins, no evidence; from the auditor's challenger, evidence (`coins`) that the bytes it read are
  exactly the coins the session the ledger opens for the registration's window (the verifier's `windowRoot`) released
  after the registration's digest, every one read and each opening checked against the challenger's commitment
  (`Challenger.coinsReason`), missing evidence or no window refused; then the draw is held to the registration and the
  expanded law (`Draw.checkDraw`);
* `deliver`: one or more session records' SHA-512 (`service._DIGEST`);
* `outcome`: the verdicts' hold (`Service.checkVerdicts`), the public outputs of the first verifier of record, and the
  audit record (`Record.record`) over the draw's proved units; the symbol accepts when the audit record does.

A step's refusal is `{"step", "codes"}`; a decided run is `{"receipt", "draw", "read", "symbol", "public", "audit"}`, the
audit record as the text of Python's `canonical_json`.
`Stream.outcome` holds its own outcome to this run (`verifier.run`). A run whose proof came after the proof deadline is
the clock's (`late`).
-/
import SampledProofs.OneStage
import SampledProofs.Select
import SampledProofs.Slots
import SampledProofs.Service
import SampledProofs.ChallengerCoins

namespace SampledProofs.Run

open Lean (Json)
open Flock (V)

def RECEIPT : String := "verity/registration-receipt/v1"

/-- `values` are the Spec's values. -/
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
  /-- the draw's coins are the auditor's challenger's (`coin_source` `challenger`), not the verifier's own -/
  fromChallenger : Bool
  coins : Json := Json.null

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

/-- `service.count_varies`: the law draws a count that varies from draw to draw, so the sessions of one registration set
up tables of different shapes: `bernoulli` at `0 < p < 1`. A work law's closure varies too, but the work draw with a
producer closure is a planned design, so it is not refused. -/
def countVaries (law : Json) : Bool :=
  match getD law "law", getD law "p" with
  | .str "bernoulli", .str p => match Draw.probability? p with
    | some (num, den) => 0 < num && num < den
    | none => false
  | _, _ => false

/-- The draw's coins are what its source says: a draw from the verifier's own coins carries no evidence (`null`), and
one from the auditor's challenger carries evidence that holds of the bytes it read (`Challenger.coinsReason`, after the
registration's digest, for the window whose open root `window` the verifier holds), so missing evidence, or a verifier
that holds no window, is a refusal and never a skip. -/
def coinsHold (fromChallenger : Bool) (coins record : Json) (window : Option String) (read : ByteArray) : Bool :=
  if !fromChallenger then coins == Json.null else
  match window, Registration.digest record with
  | some w, .ok after => (Challenger.coinsReason coins w after read).isNone
  | _, _ => false

/-- The run, step by step; an error where the Python raises. -/
def run (i : Input) : V Json := do
  let v := i.params.verifier
  let qd ← Registration.queryDigest v.query
  let pub ← Service.checkPublic i.record (Json.str v.scheme) qd i.values i.declared
  if !pub.codes.isEmpty then return refused "register" pub.codes
  let some n := nat? (getD i.record "population") | return refused "register" #["population"]
  if n == 0 then return refused "register" #["population"]
  let some rule := i.rule.over n | return refused "register" #["rate"]
  let law ← rule.lawOver n
  let v := { v with law }
  let codes := Registration.check i.record v
  if !codes.isEmpty then return refused "register" codes
  if countVaries law then return refused "register" #["shapes"]
  if !Slots.popOk v.program v.query n then return refused "register" #["population-order"]
  let receipt ← receipt i.record i.challenger i.received
  let derived ← Draw.derive v.program v.query law v.workTable
  let (draw, _, read) ← Select.select (← Select.resolve i.rule v.program v.query) v.program v.query v.workTable
    i.closure i.source
  if !coinsHold i.fromChallenger i.coins i.record v.windowRoot (i.source.extract 0 read) then
    return refused "select" #["coins"]
  let codes := Draw.checkDraw draw (getD i.record "population") derived
  if !codes.isEmpty then return refused "select" codes
  let drawn := Draw.proved draw
  if i.sessions.isEmpty || !i.sessions.all digest? then return refused "deliver" #["sessions"]
  let held ← Service.checkVerdicts i.verdicts i.sessions i.outputs
  if !held.codes.isEmpty then return refused "outcome" held.codes
  let ofRecord := (i.verdicts.qsort (·.1 < ·.1)).filter (truthy <| getD ·.2 "of_record")
  let shown := if i.outputs.isEmpty then Json.mkObj [] else getD ((ofRecord[0]?.map (·.2)).getD Json.null) "public"
  let proved := i.provedUnits.getD drawn
  let instances := i.instances.getD proved.size
  let rec_ ← Record.record (← Record.inputOf (Json.mkObj [("registration", i.record), ("receipt", receipt),
    ("draw", draw), ("law", derived), ("proved_units", Json.arr (proved.map Lean.toJson)),
    ("instances", Lean.toJson instances), ("statement", i.statement), ("verdicts", Json.mkObj i.verdicts.toList),
    ("delta", i.delta), ("sessions", Json.arr i.sessions), ("served", i.served), ("session_roots", i.sessionRoots)]))
  let accepted := jeq (getD rec_ "accepted") (.bool true)
  return Json.mkObj [("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
    ("symbol", Json.str (if accepted then "accept" else "reject")), ("public", if accepted then shown else Json.mkObj []),
    ("audit", Json.str (← PyJson.canonical rec_))]

end SampledProofs.Run
