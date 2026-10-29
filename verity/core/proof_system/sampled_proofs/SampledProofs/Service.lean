/-
The proof service's own checks (`service.py`), each with Python's codes, order and details:

* `checkPublic`: the public-items rule (`service.check_public`) on a one-stage registration record: each observable item
  is the service's own, a salted root of one of the given values, or an item the Spec declares. Codes: `keys` alone for a
  record that is not an object; then `unsalted` (a scheme other than the service's), `query` (not the check's), `keys` (a
  field one-stage doesn't define, or an entry's), `roots` (an entry that is not one of the values' one salted root) and
  `undeclared` (the items the record shows that the Spec doesn't declare, first found first). An entry's fields are read
  in key order, as canonical JSON writes them.
* `checkVerdicts`: what an outcome holds the verifiers' verdicts to before the audit record (`service.check_verdicts`),
  the first failure only: `pending` (no verdict of record, or one that hasn't run), `sessions` (verdicts of record about
  other sessions than the delivered ones) and `public` (the check's public outputs not reported alike).
-/
import SampledProofs.Registration
import SampledProofs.PyJson

namespace SampledProofs.Service

open Lean (Json)
open Flock (V)

def RECORD_KEYS : List String := ["format", "protocol", "program", "query", "partition", "population", "law", "scheme",
  "roots", "leaf_layer", "anchors", "window", "previous", "registered"]

/-- A committed value as the rule reads it (`service.Value`): its name, and whether it is registered. -/
structure Value where
  name : String
  registered : Bool

/-- The layout fields an entry shows: a root's (`roots`) or a registered value's (`registered`). -/
def layout (registered : Bool) : List String :=
  if registered then ["leaves", "words", "domain", "word_bits"] else ["leaves", "domain"]

/-- Python's `str(x)` of a JSON scalar; refused for a list or an object, whose `repr` this port does not write. -/
def pyStr : Json → V String
  | .null => pure "None"
  | .bool b => pure (if b then "True" else "False")
  | .num x => PyJson.number x
  | .str s => pure s
  | _ => throw "str of a list or an object"

/-- `dict.fromkeys(xs)`: each once, first found first. -/
def firsts (xs : Array String) : Array String := xs.foldl (fun acc x => if acc.contains x then acc else acc.push x) #[]

/-- The public-items rule on `record` with the service's `scheme`, the check's query digest `query`, the values whose
roots it holds, and the Spec's declared items. Refused where Python raises: an entry named by a list or an object
(unhashable), or a stratum's template one. -/
def checkPublic (record scheme : Json) (query : String) (values : Array Value) (declared : Array String) : V Check := do
  let .obj _ := record | return ({} : Check).fail "keys" (Json.str "not a record")
  let present := fun k => (field? record k).isSome
  let mut extra := sortStrings ((entries record).map (·.1) |>.filter (!RECORD_KEYS.contains ·))
  let mut c : Check := {}
  if !jeq (getD record "scheme") scheme then
    c := c.fail "unsalted" (Json.mkObj [("scheme", getD record "scheme"), ("service", scheme)])
  if !jeq (getD record "query") (Json.str query) then
    c := c.fail "query" (Json.mkObj [("registered", getD record "query"), ("check", Json.str query)])
  let mut need : Array String := #[]
  if ["population", "program", "partition"].any present then need := need.push "population"
  if let .arr strata := getD (getD record "law") "strata" then
    for s in strata do
      need := need.push ("population:" ++ (← pyStr (if s matches .obj _ then getD s "template" else s)))
  if present "window" then need := need.push "window"
  if !(getD record "leaf_layer" matches .null) then need := need.push "leaf_layer"
  if truthy (getD record "anchors") then need := need.push "anchors"
  let mut wrong : Array Json := #[]
  for (fieldName, registered) in [("roots", false), ("registered", true)] do
    let some v := field? record fieldName | continue
    let es := match v with
      | .arr xs => xs
      | x => #[x]
    let names := (values.filter (·.registered == registered)).map (·.name)
    let key := if registered then "value" else "port"
    let mut seen : Array String := #[]
    for e in es do
      match getD e key with
      | .null => wrong := wrong.push e
      | .arr _ | .obj _ => throw "an entry named by a list or an object"
      | .str name =>
        if !names.contains name || seen.contains name then
          wrong := wrong.push (Json.str name)
        else
          seen := seen.push name
          for k in sortStrings ((entries e).map (·.1)) do
            if (layout registered).contains k then need := need.push s!"{k}:{name}"
            else if k != key && k != "root" then extra := extra.push s!"{fieldName}.{name}.{k}"
      | name => wrong := wrong.push name
  if !extra.isEmpty then c := c.fail "keys" (Json.arr (extra.map Json.str))
  if !wrong.isEmpty then c := c.fail "roots" (Json.arr wrong)
  let missing := (firsts need).filter (!declared.contains ·)
  if !missing.isEmpty then c := c.fail "undeclared" (Json.arr (missing.map Json.str))
  return c

/-- The verdicts' share of an outcome, on the verdicts in key order, the delivered `sessions` and the check's public
`outputs` (distinct). Refused where Python would read a verdict of record's `sessions` that is neither a list nor empty
(`list` of it), or compare reported outputs canonical JSON doesn't write (a non-integer number). -/
def checkVerdicts (verdicts : Array (String × Json)) (sessions : Array Json) (outputs : Array String) : V Check := do
  let ofRecord := (verdicts.qsort (·.1 < ·.1)).filter (truthy <| getD ·.2 "of_record")
  if ofRecord.isEmpty || ofRecord.any (getD ·.2 "accepted" matches .null) then
    return ({} : Check).fail "pending" (Json.str "the verifier of record hasn't run")
  let wrong ← ofRecord.filterMapM fun (k, v) => do
    let listed ← match getD v "sessions" with
      | .arr xs => pure xs
      | x => if truthy x then throw "a verdict of record's sessions are not a list" else pure #[]
    return if listed.size == sessions.size && (listed.zip sessions).all fun (a, b) => jeq a b then none else some k
  if !wrong.isEmpty then
    return ({} : Check).fail "sessions" (Json.mkObj [("verdicts", Json.arr (wrong.map Json.str)),
      ("delivered", Json.arr sessions)])
  if outputs.isEmpty then return {}
  let reported := ofRecord.map (getD ·.2 "public")
  let shaped := reported.all fun r => keys? r == some (sortStrings outputs)
  if shaped then
    for r in reported do let _ ← Flock.canon r
  if !shaped || !reported.all (jeq · (reported.getD 0 Json.null)) then
    return ({} : Check).fail "public" (Json.mkObj [("outputs", Json.arr (outputs.map Json.str)),
      ("reported", Json.arr reported)])
  return {}

end SampledProofs.Service
