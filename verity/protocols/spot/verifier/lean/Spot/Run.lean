/-
The proof system's run of one registration, from its public check through its verdict, as `service.Stream` makes it on a Spec
whose selection rule `Select.Rule` reads: what the challenger holds once a delivered proof's outcome is decided, and every
check the stream ran on it, in the stream's order.

* `register`: the public-items rule on the Spec's values (`Service.checkPublic`), the population (a positive integer),
  the rule's rate over it and the program's strata (`Select.Rule.over`, `Select.strataSizes`: `rate`, a count of none or
  more than the population) and the work table it is expanded over (`Select.Rule.tableOver`: `rate`, a work rate law
  whose strata or table don't apply), then R1–R8 (`Registration.check`) against the challenger's copies under the sampling design the rule names over
  that population (`Select.Rule.lawOver`, whatever law the copies carry) and that work table, a `bernoulli` sampling
  design at any `p` (`Draw.bernoulli`: `bernoulli`), a
  program and query whose population is not the registration's units in order (`Slots.popOk`: `population-order`), and
  the receipt (`registration.receipt`);
* `select`: a closure map, refused before any coin is read (`closure`: a work sample widened by one is outside the
  sampled audit's soundness statement); the sample of record on the bytes of the challenger's coins (`Select.select`,
  under the rule over the program's population, `Select.resolve`); the verifier has no coins of its own, so a sample that
  read any byte carries evidence (`coins`) that the bytes are exactly the coins the session the ledger opens for the
  registration's window (the verifier's `windowRoot`) released after the registration's digest, every one read and each
  opening checked against the challenger's commitment (`Challenger.coinsReason`), and one that read none carries none
  (`coinsHold`, code `coins`), missing evidence or no window refused; then the sample is held to the registration and the
  expanded law (`Draw.checkDraw`);
* under a two-stage rule (`Select.Rule.twoStage grain replay k`), the registration is the boundary commitment over the
  replay blocks, the population of the challenger's program under its query, which is the rule's `grain` (else `grain`),
  and `select` is the first stage, its replay sample. Then `interiors`: the interior registration (`Interior`) over the
  sample's `Q · n` proof blocks, `n = Σ n_t` those of one replay block, `n_t` its instances of template `t`
  (`Select.replayCells` of the challenger's replay block Program under the check's query), template-major, replay block
  `sample[i]`'s instances of template `t` being `[off_t + i·n_t, off_t + (i+1)·n_t)`, `off_t = Q · Σ_{t'<t} n_t'`: the
  public-items rule on the interior values, the population (else `interior-population`), its window (else
  `interior-window`: the boundary registration's, or the one the verifier computes from the boundary registration's
  digest and the sample, `interiorWindowOk`), R1–R8 against its copies under `subset:Q·n` (all of them committed),
  `population-order`, and its receipt; and `second`: the second stage on its own
  coins (`Select.selectSecond`, the rule's counts over the cells, `Select.Counts.over`), read after the interior
  registration's digest and carrying their evidence as the first stage's do (`coins`);
* `deliver`: one or more session records' SHA-512 (`service._DIGEST`);
* `outcome`: the verdicts' hold (`Service.checkVerdicts`), the public outputs of the first verifier, and the
  audit record (`Record.record`) over the sample's proved units, and under a two-stage rule the interior registration's
  over the second stage's sample, with round 2's statement check of the second-stage sessions' statements against the
  interior registration, the second stage's cells and the design's links and wires, the rows their units read and the
  names of what the block returns, which the verifier derives from the replay block's Definition in the challenger's
  program (`SecondStage.replayLinks`, `SecondStage.replayWires`, `SecondStage.replayRows`,
  `SecondStage.replayReturns`), as its `statements` check
  (`SecondStage.check`, the two-stage form allowed only when `SecondStage.FORM` is); the symbol accepts when every audit
  record does. Under a two-stage rule each audit record takes the sessions' reads of its registration's own ports at
  their own roots (`own_reads`), whose rows round 2 holds.

A step's refusal is `{"step", "codes"}`; a decided run is `{"receipt", "draw", "read", "symbol", "public", "audit"}`, the
audit record as the text of Python's `canonical_json`, and under a two-stage rule `interior`, the interior registration's
`{"receipt", "draw", "read", "audit"}` in the same forms.
`Stream.outcome` holds its own outcome to this run (`verifier.run`). A run whose proof came after the proof deadline is
the clock's (`late`).
-/
module

public import Spot.OneStage
public import Spot.Select
public import Spot.Slots
public import Spot.Service
public import Spot.ChallengerCoins
public import Spot.SecondStage

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Run

open Lean (Json)
open Flock (V)

def RECEIPT : String := "verity/registration-receipt/v1"

/-- A two-stage run's interior registration and second stage: the challenger's copies for the interior registration
(its program, the population of the replay sample's proof blocks, under the check's query), one replay block's Program,
the interior values, the record and its place in the log, the second stage's coins and their evidence, its audit
record's proved units, instances, served units and session roots, the second-stage sessions' statements
(`SecondStage.Stmt`), and the circuit file each template's sessions verify against, by SHA-512, with the reference
header and registered reads round 2 holds their statements to (`SecondStage.headerOk`, `SecondStage.valuesOk`), as the
auditor's params name them (`templateCircuits`). The design's links and wires, their rows and the names of what the
block returns are the verifier's own, from the replay block's Definition in the boundary program
(`SecondStage.replayLinks`, `SecondStage.replayWires`, `SecondStage.replayRows`, `SecondStage.replayReturns`), and
never part of it. -/
structure Interior where
  params : OneStage.Params
  replay : Program
  values : Array Service.Value
  record : Json
  received : String
  source : ByteArray
  coins : Json := Json.null
  provedUnits : Option (Array Int)
  instances : Option Int
  served : Json
  sessionRoots : Json
  statements : Option (Array Json) := none
  circuits : Array (String × String × Json × Json) := #[]

/-- `values` are the Spec's values, a two-stage rule's boundary values. -/
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
  coins : Json := Json.null
  interior : Option Interior := none

private def refused (step : String) (codes : Array String) : Json :=
  Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]

private def hexDigit (c : Char) : Bool := ('0' ≤ c && c ≤ '9') || ('a' ≤ c && c ≤ 'f')

/-- `service._DIGEST.match`: 128 lowercase hex digits, which `$` lets a final newline follow. -/
def digest? : Json → Bool
  | .str s => (s.length == 128 || (s.length == 129 && s.endsWith "\n")) && (s.toList.take 128).all hexDigit
  | _ => false

/-- `service.template_circuits`: the auditor's params naming each template's circuit file and its registered reference
statement's header and registered reads, a two-stage run's `interior.params.proofs`, `{"templates": [{"template":
<descriptor id>, "circuit": <SHA-512, lowercase hex>, "header": <the registered reference statement's header>,
"registered": <the registered reads it reads, null for none>}, ...]}`, as (template, circuit, header,
registered) in the order the check's template query `query` names the templates (`Select.templateIds`), which is
REPLAY_c's first-instance order (`Select.replayCells`); `null` names no file and gives none. An error for any other
form, a circuit that isn't 128 lowercase hex digits, a header that isn't an object, registered reads neither an object
nor null, and a list that doesn't name each of the check's templates once, in its order. -/
def templateCircuits (proofs query : Json) : V (Array (String × String × Json × Json)) := do
  if proofs matches .null then return #[]
  let ids ← Select.templateIds query
  let form := "the params' proofs are {\"templates\": [{\"template\": <descriptor id>, \"circuit\": <SHA-512>, " ++
    "\"header\": <object>, \"registered\": <object or null>}, ...]}"
  let (some #["templates"], .arr es) := (keys? proofs, getD proofs "templates") | throw form
  let named ← es.mapM fun e =>
    let bad := s!"{form}, each circuit a SHA-512 in lowercase hex, each header an object and each registered " ++
      s!"an object or null: {e.compress}"
    match keys? e, getD e "template", getD e "circuit", getD e "header", getD e "registered" with
    | some #["circuit", "header", "registered", "template"], .str t, .str c, .obj h, r =>
      if c.length == 128 && c.all hexDigit && ((r matches .obj _) || (r matches .null)) then pure (t, c, Json.obj h, r)
      else throw bad
    | _, _, _, _, _ => throw bad
  if named.map (·.1) != ids then
    throw s!"the params name the circuits of {named.map (·.1)}, not the check's templates {ids}, each once, in its order"
  return named

/-- The receipt the stream writes for an accepted registration: its digest, the challenger and its place in the log. -/
def receipt (record : Json) (challenger received : String) : V Json := do
  return Json.mkObj [("format", Json.str RECEIPT), ("registration", Json.str (← Registration.digestHex record)),
    ("verifier", Json.str challenger), ("received", Json.str received)]

/-- The sample's coins are the challenger's: a sample that read any byte carries evidence that holds of the bytes it read
(`Challenger.coinsReason`, after the registration's digest, for the window whose open root `window` the verifier holds),
so missing evidence, or a verifier that holds no window, is a refusal and never a skip; one that read none (`All`)
carries none (`null`). -/
def coinsHold (coins record : Json) (window : Option String) (read : ByteArray) : Bool :=
  if read.isEmpty then coins == Json.null else
  match window, Registration.digest record with
  | some w, .ok after => (Challenger.coinsReason coins w after read).isNone
  | _, _ => false

/-- The audit record over `draw` (`Record.record`) with the run's statement, verdicts, δ and sessions, taking reads of
the registration's own ports at their own roots under a two-stage rule. -/
private def audit (i : Input) (record receipt draw law : Json) (proved : Array Int) (instances : Int)
    (served sessionRoots : Json) : V Json :=
  do Record.record (← Record.inputOf (Json.mkObj [("registration", record), ("receipt", receipt),
    ("draw", draw), ("law", law), ("proved_units", Json.arr (proved.map Lean.toJson)),
    ("instances", Lean.toJson instances), ("statement", i.statement), ("verdicts", Json.mkObj i.verdicts.toList),
    ("delta", i.delta), ("sessions", Json.arr i.sessions), ("served", served), ("session_roots", sessionRoots),
    ("own_reads", Json.bool (i.rule matches .twoStage ..))]))

/-- Under a two-stage rule, the check's query is the rule's `grain`. -/
private def grainHolds (rule : Select.Rule) (query : Json) : Bool :=
  match rule with
  | .twoStage grain _ _ => jeq query grain
  | _ => true

/-- **The interior registration's window** is the boundary registration's, or the one the verifier computes from the
boundary registration's digest `digest` (lowercase hex) and the replay sample (`SecondStage.interiorRoot`); `boundary`
and `interior` are the two registrations' window roots as the verifier holds them. So one session samples both stages,
or the second stage's is the session on the one root the boundary registration and its sample name: the second stage
is sampled on no other window. -/
def interiorWindowOk (boundary interior : Option String) (digest : String) (sample : Array Nat) : Bool :=
  interior == boundary || (interior.isSome && interior == SecondStage.interiorRoot digest sample)

/-- The interior registration and the second stage over the first stage's replay sample `sample`, the rule's counts `k`
in each cell: the interior, its receipt, the second stage's sample and law, and the bytes it read; or the refusal's step
and codes. -/
private def interiorSteps (i : Input) (it : Interior) (k : Select.Counts) (sample : Array Nat) :
    V (Except (String × Array String) (Interior × Json × Json × Json × Nat)) := do
  let v := it.params.verifier
  let pub ← Service.checkPublic it.record (Json.str v.scheme) (← Registration.queryDigest v.query) it.values i.declared
  if !pub.codes.isEmpty then return .error ("interiors", pub.codes)
  let ns ← Select.replayCells it.replay v.query
  let ks ← k.over (← Select.templateIds v.query) ns
  let m := sample.size * ns.toList.sum
  if nat? (getD it.record "population") != some m then return .error ("interiors", #["interior-population"])
  let boundary ← Registration.digestHex i.record
  if !interiorWindowOk i.params.verifier.windowRoot v.windowRoot boundary sample then
    return .error ("interiors", #["interior-window"])
  let v := { v with law := Json.mkObj [("law", "subset"), ("k", Lean.toJson m)] }
  let codes := Registration.check it.record v
  if !codes.isEmpty then return .error ("interiors", codes)
  if !Slots.popOk v.program v.query m then return .error ("interiors", #["population-order"])
  let receipt ← receipt it.record i.challenger it.received
  let (draw, read) ← Select.selectSecond sample ns ks it.source
  if !coinsHold it.coins it.record v.windowRoot (it.source.extract 0 read) then return .error ("second", #["coins"])
  return .ok (it, receipt, draw, Select.secondLaw sample ns ks, read)

/-- The second stage, under a two-stage rule, over the first stage's replay sample `sample`; none under a one-stage
rule. -/
private def secondStage (i : Input) (sample : Array Nat) :
    V (Except (String × Array String) (Option (Interior × Json × Json × Json × Nat))) :=
  match i.rule, i.interior with
  | .twoStage _ _ k, some it => return (← interiorSteps i it k sample).map some
  | .twoStage _ _ _, none => return .error ("interiors", #["interior"])
  | _, some _ => throw "a one-stage run has no interior registration"
  | _, none => return .ok none

/-- The interior registration's audit record over the second stage's proved units, with round 2's statement check
(`SecondStage.check`) over the first stage's replay sample `sample`: whether it accepts, and the run's `interior` entry;
nothing under a one-stage rule. -/
private def interiorAudit (i : Input) (sample : Array Nat) :
    Option (Interior × Json × Json × Json × Nat) → V (Bool × List (String × Json))
  | none => return (true, [])
  | some (it, receipt, draw, law, read) => do
    let proved := it.provedUnits.getD (Draw.proved draw)
    let rec_ ← audit i it.record receipt draw law proved (it.instances.getD proved.size) it.served it.sessionRoots
    let strata ← Partition.strata it.params.verifier.program it.params.verifier.query
    let own : SecondStage.Own :=
      { sample, parts := SecondStage.parts sample.size strata, interior := it.record, boundary := i.record,
        drawn := (Draw.proved draw).map Int.toNat,
        links := SecondStage.replayLinks i.params.verifier.program.prog it.replay.prog.programFn i.record,
        reported := Record.sessionRegistered i.verdicts, verified := SecondStage.verifiedOf i.verdicts,
        refs := it.circuits.map fun (t, _, h, _) => (t, h), refReads := it.circuits.map fun (t, _, _, r) => (t, r),
        wires := SecondStage.replayWires i.params.verifier.program.prog it.replay.prog.programFn,
        rows := SecondStage.replayRows i.params.verifier.program.prog it.replay.prog.programFn,
        returns := SecondStage.replayReturns i.params.verifier.program.prog it.replay.prog.programFn i.record }
    let rec_ := SecondStage.attach rec_ (SecondStage.check SecondStage.FORM own it.statements)
      (SecondStage.interiorReads own ((it.statements.getD #[]).toList))
    return (jeq (getD rec_ "accepted") (.bool true), [("interior", Json.mkObj [("receipt", receipt), ("draw", draw),
      ("read", Lean.toJson read), ("audit", Json.str (← PyJson.canonical rec_))])])

/-- The run, step by step; an error where the Python raises. -/
def run (i : Input) : V Json := do
  let v := i.params.verifier
  if !grainHolds i.rule v.query then return refused "register" #["grain"]
  let qd ← Registration.queryDigest v.query
  let pub ← Service.checkPublic i.record (Json.str v.scheme) qd i.values i.declared
  if !pub.codes.isEmpty then return refused "register" pub.codes
  let some n := nat? (getD i.record "population") | return refused "register" #["population"]
  if n == 0 then return refused "register" #["population"]
  let some rule := i.rule.over n (Select.strataSizes v.program v.query) | return refused "register" #["rate"]
  let law ← rule.lawOver n
  let .ok workTable := i.rule.tableOver v.program v.query v.workTable | return refused "register" #["rate"]
  let v := { v with law, workTable }
  let codes := Registration.check i.record v
  if !codes.isEmpty then return refused "register" codes
  if Draw.bernoulli law then return refused "register" #["bernoulli"]
  if !Slots.popOk v.program v.query n then return refused "register" #["population-order"]
  let receipt ← receipt i.record i.challenger i.received
  let derived ← Draw.derive v.program v.query law v.workTable
  if i.closure.isSome then return refused "select" #["closure"]
  let (draw, _, read) ← Select.select (← Select.resolve i.rule v.program v.query) v.program v.query v.workTable
    i.closure i.source
  if !coinsHold i.coins i.record v.windowRoot (i.source.extract 0 read) then
    return refused "select" #["coins"]
  let codes := Draw.checkDraw draw (getD i.record "population") derived
  if !codes.isEmpty then return refused "select" codes
  let drawn := Draw.proved draw
  match ← secondStage i (drawn.map Int.toNat) with
  | .error (step, codes) => return refused step codes
  | .ok inner =>
    if i.sessions.isEmpty || !i.sessions.all digest? then return refused "deliver" #["sessions"]
    let held ← Service.checkVerdicts i.verdicts i.sessions i.outputs
    if !held.codes.isEmpty then return refused "outcome" held.codes
    let ofRecord := (i.verdicts.qsort (·.1 < ·.1)).filter (truthy <| getD ·.2 "of_record")
    let shown := if i.outputs.isEmpty then Json.mkObj [] else getD ((ofRecord[0]?.map (·.2)).getD Json.null) "public"
    let proved := i.provedUnits.getD drawn
    let instances := i.instances.getD proved.size
    let rec_ ← audit i i.record receipt draw derived proved instances i.served i.sessionRoots
    let (accepted₂, staged) ← interiorAudit i (drawn.map Int.toNat) inner
    let accepted := jeq (getD rec_ "accepted") (.bool true) && accepted₂
    return Json.mkObj ([("receipt", receipt), ("draw", draw), ("read", Lean.toJson read),
      ("symbol", Json.str (if accepted then "accept" else "reject")),
      ("public", if accepted then shown else Json.mkObj []), ("audit", Json.str (← PyJson.canonical rec_))] ++ staged)

end Spot.Run
