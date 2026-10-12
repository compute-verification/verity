/-
The one-stage audit's pieces as JSON functions, which `sampled-proofs-difftest` serves (`Difftest.serve`) for the
difftest against `one_stage`'s Python, beside the whole protocol (`sampled-proofs-one-stage serve`):

* `decode`: a case's three decoders run independently (`answer` stops at the first refusal);
* `partition`: the partition's digest, population, strata and expanded law from the verifier's parameters;
* `registration`: R1–R7's failure codes for a record against the parameters;
* `registered`: the verifier's own entries for the values its parameters expect registered;
* `domain`: a registered value's commitment domain;
* `check_draw`, `check_proved`: the draw's checks, each `{"ok", "codes", "detail"}`;
* `draw`: the verifier's draw from given bytes of its own randomness;
* `bound`: the exact bounds on wrong units at δ;
* `verdict`: the verdicts' share of the outcome once the record's checks hold;
* `matches_public`: the verifier's own public file against the record (`Record.matchesPublic`);
* `session_registered`: the registered roots the verifiers of record report (`Record.sessionRegistered`), and with a
  registration only its own values' (`Record.ownRegistered`);
* `record`: the whole audit record (`Record.record`), as the text of Python's `canonical_json`;
* `check_public`: the proof service's public-items rule on a registration record (`Service.checkPublic`);
* `check_verdicts`: the verdicts' share of a proof service outcome (`Service.checkVerdicts`).

`sampled-proofs-verifier` serves those the verifier of record runs (`verifier`): `registration`, `matches_public`,
`check_draw`, `check_proved`, `record`, `check_public` and `check_verdicts`.
-/
import SampledProofs.OneStage
import SampledProofs.Service

namespace SampledProofs.Pieces

open Lean (Json)
open Flock (V)

private def params (j : Json) : V OneStage.Params := do
  OneStage.paramsOfJson (← Core.canonicalJson (← Flock.canon j).toUTF8)

private def orNull {α} (f : α → Json) : V α → Json
  | .ok a => f a
  | .error _ => Json.null

private def codes (cs : Array String) : Json := Json.mkObj [("codes", Json.arr (cs.map Json.str))]

private def ints (j : Json) (what : String) : V (Array Int) := do
  let .arr xs := j | throw s!"{what} is not a list of integers"
  xs.mapM fun x => match int? x with
    | some n => pure n
    | none => throw s!"{what} is not a list of integers"

/-- `{"commitment": its digest or null, "evidence": decodes, "params": decodes}`; a malformed case is refused. -/
def decode (j : Json) : V Json := do
  let p ← Flock.J.get j "params" "the case"
  let c ← Flock.unhex (← Flock.J.str j "commitment" "the case")
  let e ← Flock.unhex (← Flock.J.str j "evidence" "the case")
  return Json.mkObj [("params", Json.bool (params p).toBool),
    ("commitment", orNull (fun c => Json.str (Flock.hexOf c.digest)) (OneStage.commitmentOf c)),
    ("evidence", Json.bool (OneStage.evidenceOf e).toBool)]

/-- `{"params"}` to `{"digest", "population", "strata", "law"}`, each null where Python raises. -/
def partition (j : Json) : V Json := do
  let v := (← params (getD j "params")).verifier
  let part := v.partition.getD (Partition.build (Flock.hexOf v.program.sha512) v.query)
  let range := fun ((lo, hi) : Nat × Nat) => Json.arr #[Lean.toJson lo, Lean.toJson hi]
  return Json.mkObj [("digest", (Partition.digest part).elim Json.null Json.str),
    ("population", orNull Lean.toJson (Partition.units v.program v.query)),
    ("strata", orNull (fun ss => Json.arr (ss.map fun (t, rs) => Json.arr #[Json.str t, Json.arr (rs.map range)]))
      (Partition.strata v.program v.query)),
    ("law", orNull id (Draw.derive v.program v.query v.law v.workTable))]

/-- `{"params", "record"}` to R1–R7's codes. -/
def registration (j : Json) : V Json := do
  return codes (Registration.check (getD j "record") (← params (getD j "params")).verifier)

/-- `{"params"}` to `registered_entries`: `{value: {"leaves", "words", "word_bits", "domain"}}`, null where it raises. -/
def registered (j : Json) : V Json := do
  let v := (← params (getD j "params")).verifier
  return orNull (fun es => Json.mkObj (es.toList.map fun e => (e.value, e.toJson)))
    (Registered.entries v.program v.params v.registered)

/-- `{"program", "value", "words", "leaves", "word_bits"}`, two strings and three natural numbers, to
`registered.domain(...).domain_id` in hex; refused where it raises (a width but 16 or 32). -/
def domain (j : Json) : V Json := do
  let (.str program, .str value) := (getD j "program", getD j "value") | throw "program or value is not a string"
  let (some words, some leaves, some b) := (nat? (getD j "words"), nat? (getD j "leaves"), nat? (getD j "word_bits"))
    | throw "words, leaves or word_bits is not a natural number"
  return Json.str (Flock.hexOf (← Registered.domain program value words leaves b))

/-- `{"draw", "population", "law"}`, the population an integer and the law an object, to `check_draw`'s codes. -/
def checkDraw (j : Json) : V Json := do
  if (int? (getD j "population")).isNone then throw "population is not an integer"
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  return (Draw.checkDrawFull (getD j "draw") (getD j "population") (getD j "law")).toJson

/-- `{"law", "population", "source"}`, the law an object, the population a natural number and the source hex, to the draw
`two_stage.draw_first` samples from the source's bytes; refused where it raises. -/
def draw (j : Json) : V Json := do
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  let some n := nat? (getD j "population") | throw "population is not a natural number"
  Draw.drawFirst (getD j "law") n (← Flock.unhex (← Flock.J.str j "source" "the draw"))

/-- `{"draw", "proved_units", "instances"}`, a draw whose units (and closure) are integers, to `check_proved`'s codes. -/
def checkProved (j : Json) : V Json := do
  let draw := getD j "draw"
  if !(draw matches .obj _) then throw "draw is not an object"
  let _ ← ints (getD draw "units") "units"
  if let some c := field? draw "closure" then let _ ← ints c "closure"
  let us ← ints (getD j "proved_units") "proved_units"
  if (int? (getD j "instances")).isNone then throw "instances is not an integer"
  return (Draw.checkProvedFull draw (us.map Lean.toJson) (getD j "instances")).toJson

/-- `{"strata": [{"name", "size", "k"}], "delta"}` to `{"bound", "per_stratum"}` (`Stratified.count_bound` and
`stratum_bound`); refused where `Stratified` is: no strata, a name empty or twice, `k > size`, δ not in (0, 1). -/
def bound (j : Json) : V Json := do
  let .arr ss := getD j "strata" | throw "strata is not a list"
  let strata ← ss.mapM fun s => do
    if keys? s != some #["k", "name", "size"] then throw "a stratum is exactly {name, size, k}"
    let (.str name, some size, some k) := (getD s "name", nat? (getD s "size"), nat? (getD s "k"))
      | throw "a stratum's name is a string and its size and k natural numbers"
    if name.isEmpty || size < k then throw "a stratum needs a name and 0 <= k <= size"
    pure ({ name, size, k } : Core.Stratum)
  if strata.isEmpty then throw "no strata"
  if (sortStrings (strata.map (·.name))).toList.eraseDups.length != strata.size then throw "a name twice"
  let .str d := getD j "delta" | throw "delta is not a ratio"
  let δ ← Core.Ratio.ofString d "delta"
  if δ.num == 0 || δ.den ≤ δ.num then throw "delta is not in (0, 1)"
  return Json.mkObj [("bound", Lean.toJson (Bound.countBound strata δ)),
    ("per_stratum", Json.mkObj (strata.toList.map fun s => (s.name, Lean.toJson (Bound.stratumBound s δ))))]

/-- `{"verdicts": {name: {...}}}` to `{"verdict"}`. -/
def verdict (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  let name : Core.Verdict := match OneStage.verdictOf vs with
    | .accept () () => .accept
    | .reject _ => .reject
    | .undecided _ => .undecided
  return Json.mkObj [("verdict", Json.str name.name)]

/-- `{"record", "scheme", "roots", "domains", "population"}` to `matches_public`'s `{"ok", "codes", "detail"}`: a record
whose roots are objects with a string port, root and domain, which has a scheme and an integer population; a string
scheme, roots and domains objects of strings, an integer population. (Python's `==` and `jeq` agree there: no boolean
meets an integer.) -/
def matchesPublic (j : Json) : V Json := do
  let rec_ := getD j "record"
  let .arr roots := getD rec_ "roots" | throw "the record's roots are not a list"
  if !roots.all fun r => ["port", "root", "domain"].all (getD r · matches .str _) then
    throw "a registered root is not an object with a string port, root and domain"
  if (field? rec_ "scheme").isNone || (int? (getD rec_ "population")).isNone then
    throw "the record has no scheme or no integer population"
  let strings := fun (k : String) => match getD j k with
    | .obj _ => (entries (getD j k)).all (·.2 matches .str _)
    | _ => false
  if !(getD j "scheme" matches .str _) || !strings "roots" || !strings "domains" || (int? (getD j "population")).isNone then
    throw "scheme is not a string, roots or domains not an object of strings, or population not an integer"
  return (Record.matchesPublic rec_ (getD j "scheme") (getD j "roots") (getD j "domains") (getD j "population")).toJson

/-- `{"verdicts", "registration"?}` to `{"registered": {value: root} or null}` (`Record.sessionRegistered`, with a
registration `Record.ownRegistered`); refused where `inputOf` refuses the verdicts, and a registration whose roots are not
a list of objects with a string port or whose registered values, if any, are not a list of objects with a string value. -/
def sessionRegistered (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  if !vs.all fun (_, v) =>
      !(truthy (getD v "of_record") && jeq (getD v "accepted") (.bool true) && (getD v "registered" matches .obj _))
      || (entries (getD v "registered")).all fun (_, e) => (getD e "value" matches .str _) && (field? e "root").isSome then
    throw "a verifier of record reports registered reads that are not objects with a string value and a root"
  let opened := Record.sessionRegistered vs
  let opened ← match field? j "registration" with
    | none => pure opened
    | some reg => do
      let .arr roots := getD reg "roots" | throw "the registration's roots are not a list"
      if !roots.all (getD · "port" matches .str _) then throw "a registered root has no string port"
      if let some es := field? reg "registered" then
        let .arr es := es | throw "the registration's registered values are not a list"
        if !es.all (getD · "value" matches .str _) then throw "a registered value has no string value"
      pure (Record.ownRegistered reg opened)
  return Json.mkObj [("registered", match opened with
    | some kvs => Json.mkObj kvs.toList
    | none => Json.null)]

/-- `audit_record`'s arguments (`Record.inputOf`) to the record, as the text Python's `canonical_json` writes. -/
def record (j : Json) : V Json := do
  return Json.str (← PyJson.canonical (← Record.record (← Record.inputOf j)))

/-- `{"record", "scheme", "query", "values": [{"name", "registered"}], "public": [item]}` to the service's public-items
rule's `{"ok", "codes", "detail"}` (`Service.checkPublic`), the query's digest computed here: a string scheme, the check's query
an object, each value a string name with a boolean, each declared item a string. -/
def checkPublic (j : Json) : V Json := do
  if !(getD j "scheme" matches .str _) then throw "scheme is not a string"
  if !(getD j "query" matches .obj _) then throw "query is not an object"
  let .arr vs := getD j "values" | throw "values is not a list"
  let values ← vs.mapM fun v => match getD v "name", getD v "registered" with
    | .str name, .bool registered => pure ({ name, registered } : Service.Value)
    | _, _ => throw "a value is not a string name with a boolean"
  let .arr ps := getD j "public" | throw "public is not a list"
  let declared ← ps.mapM fun
    | .str s => pure s
    | _ => throw "a declared item is not a string"
  let query ← Registration.queryDigest (getD j "query")
  return (← Service.checkPublic (getD j "record") (getD j "scheme") query values declared).toJson

/-- `{"verdicts", "sessions", "outputs"}` to the outcome's verdict checks' `{"ok", "codes", "detail"}`
(`Service.checkVerdicts`): verdicts an object of objects, sessions a list of strings, outputs distinct strings. -/
def checkVerdicts (j : Json) : V Json := do
  let .obj _ := getD j "verdicts" | throw "verdicts is not an object"
  let vs := entries (getD j "verdicts")
  if !vs.all (·.2 matches .obj _) then throw "a verdict is not an object"
  let .arr ss := getD j "sessions" | throw "sessions is not a list"
  if !ss.all (· matches .str _) then throw "a session is not a string"
  let .arr os := getD j "outputs" | throw "outputs is not a list"
  let outputs ← os.mapM fun
    | .str o => pure o
    | _ => throw "an output is not a string"
  if (sortStrings outputs).toList.eraseDups.length != outputs.size then throw "an output twice"
  return (← Service.checkVerdicts vs ss outputs).toJson

def table : List (String × (Json → V Json)) :=
  [("decode", decode), ("partition", partition), ("registration", registration), ("registered", registered),
   ("domain", domain), ("check_draw", checkDraw), ("draw", draw), ("check_proved", checkProved), ("bound", bound),
   ("verdict", verdict), ("matches_public", matchesPublic), ("session_registered", sessionRegistered), ("record", record),
   ("check_public", checkPublic), ("check_verdicts", checkVerdicts)]

def verifier : List (String × (Json → V Json)) :=
  [("registration", registration), ("matches_public", matchesPublic), ("check_draw", checkDraw),
   ("check_proved", checkProved), ("record", record), ("check_public", checkPublic),
   ("check_verdicts", checkVerdicts)]

end SampledProofs.Pieces
