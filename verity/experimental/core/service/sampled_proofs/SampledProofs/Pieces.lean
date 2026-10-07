/-
The one-stage audit's pieces as JSON functions, which `sampled-proofs-difftest` serves (`Difftest.serve`) for the
difftest against `one_stage`'s Python, beside the whole protocol (`sampled-proofs-one-stage serve`):

* `decode`: a case's three decoders run independently (`answer` stops at the first refusal);
* `partition`: the partition's digest, population, strata and expanded law from the verifier's parameters;
* `registration`: R1–R6's failure codes for a record against the parameters;
* `check_draw`, `check_proved`: the draw's checks;
* `bound`: the exact bounds on wrong units at δ;
* `verdict`: the verdicts' share of the outcome once the record's checks hold.
-/
import SampledProofs.OneStage

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

/-- `{"params", "record"}` to R1–R6's codes. -/
def registration (j : Json) : V Json := do
  return codes (Registration.check (getD j "record") (← params (getD j "params")).verifier)

/-- `{"draw", "population", "law"}`, the population an integer and the law an object, to `check_draw`'s codes. -/
def checkDraw (j : Json) : V Json := do
  if (int? (getD j "population")).isNone then throw "population is not an integer"
  if !(getD j "law" matches .obj _) then throw "law is not an object"
  return codes (Draw.checkDraw (getD j "draw") (getD j "population") (getD j "law"))

/-- `{"draw", "proved_units", "instances"}`, a draw whose units (and closure) are integers, to `check_proved`'s codes. -/
def checkProved (j : Json) : V Json := do
  let draw := getD j "draw"
  if !(draw matches .obj _) then throw "draw is not an object"
  let _ ← ints (getD draw "units") "units"
  if let some c := field? draw "closure" then let _ ← ints c "closure"
  let us ← ints (getD j "proved_units") "proved_units"
  if (int? (getD j "instances")).isNone then throw "instances is not an integer"
  return codes (Draw.checkProved draw (us.map Lean.toJson) (getD j "instances"))

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

def table : List (String × (Json → V Json)) :=
  [("decode", decode), ("partition", partition), ("registration", registration), ("check_draw", checkDraw),
   ("check_proved", checkProved), ("bound", bound), ("verdict", verdict)]

end SampledProofs.Pieces
