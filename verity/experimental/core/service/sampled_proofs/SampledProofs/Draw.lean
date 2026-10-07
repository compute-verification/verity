/-
The draw's checks (`one_stage/draw.py`): the verifier's law expanded over its own program and query (`derive`), a draw held
to the registration and to that law (`checkDraw`), and the proved statement's units held to the draw (`checkProved`). The
strata, their rules and the law object are C-Flock's (`Flock.Draw`). Nothing here samples: the draw comes from the verifier's
own randomness, outside the check.
-/
import Flock.Draw
import SampledProofs.Partition

namespace SampledProofs.Draw

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- `draw.derive`: `stratified` and `work` (the rule, or an expanded law, whose strata are ignored) expand over the strata of
the verifier's program and query (`Flock.Draw.derive`, and `Flock.Draw.deriveWork` with its work table); `subset` and
`bernoulli` are unchanged. An error when the query doesn't apply, `k` isn't in `[1, population]`, a work law has no table,
the table names no work for a template, or the population does none. -/
def derive (d : Desc) (query law : Json) (table : Option Json) : V Json := do
  let kind := getD law "law"
  if !(jeq kind "stratified" || jeq kind "work") then return law
  let strata ← Partition.strata d query
  let n := strata.foldl (fun acc (_, rs) => acc + rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) 0
  let some k := nat? (getD law "k") | throw s!"a law needs 1 <= k <= {n}"
  if jeq kind "stratified" then
    return (← Flock.Draw.derive k strata).toJson
  let some t := table | throw "a work law needs the verifier's work table"
  return (← Flock.Draw.deriveWork k strata (← Flock.Draw.workTableOfJson t)).toJson

/-- `draw.same_law`: the same canonical bytes. -/
def sameLaw (a b : Json) : Bool := jeq a b

/-- `draw.law_of`: the law a draw names. -/
def lawOf (draw : Json) : Json :=
  let kind := getD draw "law"
  if jeq kind "stratified" || jeq kind "work" then
    Json.mkObj [("law", kind), ("k", getD draw "k"), ("strata", getD draw "strata")]
  else if jeq kind "subset" then Json.mkObj [("law", "subset"), ("k", getD draw "k")]
  else Json.mkObj [("law", "bernoulli"), ("p", getD draw "p")]

private def STRATUM_KEYS : Array String := sortStrings #["template", "units", "k"]
private def WORK_STRATUM_KEYS : Array String := sortStrings #["template", "units", "work", "floor", "k"]

/-- `draw._ranges_ok`: a non-empty list of `[lo, hi]` integer pairs with `lo < hi`. -/
def ranges? : Json → Option (Array (Int × Int))
  | .arr rs =>
    if rs.isEmpty then none else rs.mapM fun r => match r with
      | .arr p => do
        if p.size != 2 then none
        let lo ← int? p[0]!
        let hi ← int? p[1]!
        if lo < hi then some (lo, hi) else none
      | _ => none
  | _ => none

private def natRanges (rs : Array (Int × Int)) : Option (Array (Nat × Nat)) :=
  rs.mapM fun (lo, hi) => if 0 ≤ lo then some (lo.toNat, hi.toNat) else none

/-- One stratum of a draw's law: exactly its keys, a string template, its units ranges, an integer `k` and, for a work
stratum, natural `work` and `floor`. A negative bound or `k` can't tile `[0, population)` or meet its rule, so it is a
problem here already. -/
private def stratum? (work : Bool) (s : Json) : Option Flock.Draw.WorkStratum := do
  if keys? s != some (if work then WORK_STRATUM_KEYS else STRATUM_KEYS) then none
  let .str t := getD s "template" | none
  let rs ← natRanges (← ranges? (getD s "units"))
  let k ← nat? (getD s "k")
  let w ← if work then nat? (getD s "work") else some 0
  let f ← if work then nat? (getD s "floor") else some 0
  return { template := t, units := rs, k, work := w, floor := f }

/-- `draw.strata_problem`: whether a stratified or work law is not well formed over `population` units. Each stratum is
exactly its keys with a distinct template, its ranges ascending, disjoint, not adjacent and tiling `[0, population)`, and
its `k` the rule's; the problems after parsing are C-Flock's (`Flock.Draw.strataProblem`, `Flock.Draw.workProblem`). -/
def strataProblem (law : Json) (population : Int) : Bool := Id.run do
  let work := jeq (getD law "law") "work"
  let .arr ss := getD law "strata" | return true
  let some k := int? (getD law "k") | return true
  if ss.isEmpty || k < 1 || population < k then return true
  let some strata := ss.mapM (stratum? work) | return true
  if strata.any fun s => (List.range (s.units.size - 1)).any fun i => s.units[i + 1]!.1 ≤ s.units[i]!.2 then return true
  return if work then (Flock.Draw.workProblem k.toNat strata population.toNat).isSome
    else (Flock.Draw.strataProblem k.toNat (strata.map (·.toStratum)) population.toNat).isSome

private def lawKeys (kind : String) : Array String :=
  sortStrings (#["law", "population", "units"] ++
    if kind == "subset" then #["k"] else if kind == "bernoulli" then #["p"] else #["k", "strata"])

private def ascending (xs : Array Int) : Bool := (List.range (xs.size - 1)).all fun i => xs[i]! < xs[i + 1]!

private def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- `draw.check_draw`'s failure codes: the draw is well formed, over the registered population, under the verifier's law
(for `stratified` and `work`, the expanded law of `derive`). -/
def checkDraw (draw population law : Json) : Array String := Id.run do
  let kind := match getD draw "law" with | .str s => s | _ => ""
  if !(draw matches .obj _) || !["subset", "bernoulli", "stratified", "work"].contains kind then return #["not-a-draw"]
  let closure := kind == "work" && (field? draw "closure").isSome
  let want := if closure then sortStrings ((lawKeys kind).push "closure") else lawKeys kind
  if keys? draw != some want then return #["keys"]
  let mut c : Array String := #[]
  let n := getD draw "population"
  if !sameLaw (lawOf draw) law then c := addCode c "law"
  if !jeq n population then c := addCode c "population"
  let .arr us := getD draw "units" | return addCode c "units"
  let some units := us.mapM int? | return addCode c "units"
  if !ascending units then c := addCode c "units-order"
  if h : 0 < units.size then
    let inRange := 0 ≤ units[0] && match int? n with
      | some n => units[units.size - 1]! < n
      | none => false
    if !inRange then c := addCode c "units-range"
  if kind == "subset" && int? (getD draw "k") != some (units.size : Int) then c := addCode c "units-count"
  if kind == "bernoulli" then
    let ok := match getD draw "p" with
      | .str p =>
        let num := (p.splitOn "/").headD ""
        let den := (p.drop (num.length + 1)).toString
        p.contains '/' && digits num && digits den && 0 < den.toNat! && num.toNat! ≤ den.toNat!
      | _ => false
    if !ok then c := addCode c "p"
  if kind == "stratified" || kind == "work" then
    let problem := match int? n with
      | some n => strataProblem draw n
      | none => true
    if problem then c := addCode c "strata"
    else
      let wrong := (match getD draw "strata" with | .arr ss => ss | _ => #[]).any fun s =>
        let rs := (ranges? (getD s "units")).getD #[]
        let drawn := units.foldl (fun acc u => if rs.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0
        int? (getD s "k") != some (drawn : Int)
      if wrong then c := addCode c "strata-count"
  if closure then
    let cs := match getD draw "closure" with
      | .arr cs => cs.mapM int?
      | _ => none
    match cs with
    | some cs =>
      let inRange := match cs[0]?, cs.back?, int? n with
        | some lo, some hi, some n => 0 ≤ lo && hi < n
        | _, _, _ => false
      if cs.isEmpty || !ascending cs || !inRange then c := addCode c "closure"
      else if cs.any units.contains then c := addCode c "closure-drawn"
    | none => c := addCode c "closure"
  return c

/-- `draw.proved`: the drawn units and a work draw's closure, ascending. -/
def proved (draw : Json) : Array Int :=
  let get := fun (k : String) => match getD draw k with
    | .arr xs => xs.filterMap int?
    | _ => #[]
  (get "units" ++ get "closure").qsort (· < ·)

/-- `draw.check_proved`'s failure codes: the proved statement's instances are exactly the units the draw proves, in
ascending order. -/
def checkProved (draw : Json) (provedUnits : Array Json) (instances : Json) : Array String := Id.run do
  let want := proved draw
  let mut c := #[]
  if int? instances != some (want.size : Int) then c := addCode c "instances"
  if provedUnits.map int? != want.map some then c := addCode c "units"
  return c

end SampledProofs.Draw
