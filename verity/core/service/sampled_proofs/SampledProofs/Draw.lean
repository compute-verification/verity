/-
The draw (`one_stage/draw.py`): the verifier's law expanded over its own program and query (`derive`), a draw held to the
registration and to that law (`checkDraw`), and the proved statement's units held to the draw (`checkProved`). The strata,
their rules, the law object and the samplers are C-Flock's (`Flock.Draw`). The check samples nothing: the draw comes from
the verifier's own randomness, raw bytes with no derivation, which `drawFirst` reads as `two_stage.draw_first` does.
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

/-- `NUM/DEN` in ASCII digits with `0 < DEN` and `NUM ≤ DEN`. -/
def probability? (p : String) : Option (Nat × Nat) :=
  match p.splitOn "/" with
  | [a, b] => if digits a && digits b && 0 < b.toNat! && a.toNat! ≤ b.toNat! then some (a.toNat!, b.toNat!) else none
  | _ => none

private def sumSizes (rs : Array (Int × Int)) : Int := rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0

/-- `draw.strata_problem`'s words for why a stratified or work law is not well formed over `population` units, in its
order: `k` and the strata, a work stratum's keys, work and floor, the population's work, then each stratum's keys and
template, its ranges, its floor and its `k`, and last the tiling. -/
def strataMessage (law : Json) (population : Int) : Option String := Id.run do
  let work := jeq (getD law "law") "work"
  let (.arr ss, some k) := (getD law "strata", int? (getD law "k")) | return some "strata and 1 <= k <= population"
  if ss.isEmpty || k < 1 || population < k then return some "strata and 1 <= k <= population"
  let keys := if work then WORK_STRATUM_KEYS else STRATUM_KEYS
  let mut total : Int := 0
  if work then
    if !ss.all fun s => keys? s == some keys && (nat? (getD s "work")).isSome && (nat? (getD s "floor")).isSome
        && (ranges? (getD s "units")).isSome then
      return some "a work stratum is exactly {template, units, work, floor, k}, its work and floor natural numbers"
    total := ss.foldl (fun a s => a + ((int? (getD s "work")).getD 0) * sumSizes ((ranges? (getD s "units")).getD #[])) 0
    if total == 0 then return some "a work law's population does no work"
  let mut seen : Array String := #[]
  let mut ranges : Array (Int × Int) := #[]
  for s in ss do
    let .str t := getD s "template"
      | return some (if work then "a stratum is exactly {template, units, work, floor, k}" else "a stratum is exactly {template, units, k}")
    if keys? s != some keys then
      return some (if work then "a stratum is exactly {template, units, work, floor, k}" else "a stratum is exactly {template, units, k}")
    if seen.contains t then return some s!"template {pyRepr t} twice"
    seen := seen.push t
    let some rs := ranges? (getD s "units") | return some s!"stratum {pyRepr t}: units are non-empty [lo, hi) ranges"
    if (List.range (rs.size - 1)).any fun i => rs[i + 1]!.1 ≤ rs[i]!.2 then
      return some s!"stratum {pyRepr t}: ranges ascending, disjoint and not adjacent"
    let floor := (int? (getD s "floor")).getD 0
    if work && floor < 1 then return some s!"stratum {pyRepr t}: a floor is at least one"
    let size := sumSizes rs
    let rule : Int := if work then
        Flock.Draw.workK k.toNat ((int? (getD s "work")).getD 0).toNat size.toNat total.toNat floor.toNat
      else Flock.Draw.stratumK k.toNat size.toNat population.toNat
    if int? (getD s "k") != some rule then
      return some s!"stratum {pyRepr t}: k is not the {if work then "work " else ""}rule's"
    ranges := ranges ++ rs
  let mut at_ : Int := 0
  for (lo, hi) in ranges.qsort (fun a b => a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)) do
    if lo != at_ then return some "the strata don't tile the population"
    at_ := hi
  return if at_ == population then none else some "the strata don't tile the population"

/-- `draw.check_draw`: the draw is well formed, over the registered population, under the verifier's law (for
`stratified` and `work`, the expanded law of `derive`), each failure with Python's detail. -/
def checkDrawFull (draw population law : Json) : Check := Id.run do
  let kind := match getD draw "law" with | .str s => s | _ => ""
  if !(draw matches .obj _) || !["subset", "bernoulli", "stratified", "work"].contains kind then
    return Check.fail {} "not-a-draw" draw
  let closure := kind == "work" && (field? draw "closure").isSome
  let want := if closure then sortStrings ((lawKeys kind).push "closure") else lawKeys kind
  let have_ := (keys? draw).getD #[]
  if have_ != want then
    return Check.fail {} "keys" (Json.arr (sortStrings ((have_.filter (!want.contains ·)) ++ want.filter (!have_.contains ·)) |>.map Json.str))
  let mut c : Check := {}
  let n := getD draw "population"
  if !sameLaw (lawOf draw) law then c := c.fail "law" (Json.mkObj [("drawn", lawOf draw), ("verifier", law)])
  if !jeq n population then c := c.fail "population" (Json.mkObj [("drawn", n), ("registered", population)])
  let us := getD draw "units"
  let some units := (match us with | .arr xs => xs.mapM int? | _ => none) | return c.fail "units" us
  if !ascending units then c := c.fail "units-order" (intsJson (units.extract 0 16))
  if h : 0 < units.size then
    let inRange := 0 ≤ units[0] && match int? n with
      | some n => units[units.size - 1]! < n
      | none => false
    if !inRange then
      let out := units.filter fun u => !(0 ≤ u && match int? n with | some n => u < n | none => false)
      c := c.fail "units-range" (intsJson (out.extract 0 16))
  if kind == "subset" && int? (getD draw "k") != some (units.size : Int) then
    c := c.fail "units-count" (Json.mkObj [("units", Lean.toJson units.size), ("k", getD draw "k")])
  if kind == "bernoulli" then
    let ok := match getD draw "p" with
      | .str p => (probability? p).isSome
      | _ => false
    if !ok then c := c.fail "p" (getD draw "p")
  if kind == "stratified" || kind == "work" then
    let problem := match int? n with
      | some n => strataMessage draw n
      | none => some "population"
    match problem with
    | some why => c := c.fail "strata" (Json.str why)
    | none =>
      let wrong := (match getD draw "strata" with | .arr ss => ss | _ => #[]).filterMap fun s =>
        let rs := (ranges? (getD s "units")).getD #[]
        let drawn := units.foldl (fun acc u => if rs.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0
        if int? (getD s "k") != some (drawn : Int) then
          match getD s "template" with
          | .str t => some (t, Json.mkObj [("drawn", Lean.toJson drawn), ("k", getD s "k")])
          | _ => none
        else none
      if !wrong.isEmpty then c := c.fail "strata-count" (Json.mkObj wrong.toList)
  if closure then
    let raw := getD draw "closure"
    let cs := match raw with
      | .arr cs => cs.mapM int?
      | _ => none
    let bad := match raw with | .arr _ => raw | _ => Json.null
    match cs with
    | some cs =>
      let inRange := match cs[0]?, cs.back?, int? n with
        | some lo, some hi, some n => 0 ≤ lo && hi < n
        | _, _, _ => false
      if cs.isEmpty || !ascending cs || !inRange then c := c.fail "closure" bad
      else
        let both := (cs.filter units.contains).qsort (· < ·)
        if !both.isEmpty then c := c.fail "closure-drawn" (intsJson (both.extract 0 16))
    | none => c := c.fail "closure" bad
  return c

/-- `draw.check_draw`'s failure codes. -/
def checkDraw (draw population law : Json) : Array String := (checkDrawFull draw population law).codes

/-- `draw.proved`: the drawn units and a work draw's closure, ascending. -/
def proved (draw : Json) : Array Int :=
  let get := fun (k : String) => match getD draw k with
    | .arr xs => xs.filterMap int?
    | _ => #[]
  (get "units" ++ get "closure").qsort (· < ·)

/-- `draw.check_proved`'s failure codes: the proved statement's instances are exactly the units the draw proves, in
ascending order. -/
def checkProvedFull (draw : Json) (provedUnits : Array Json) (instances : Json) : Check := Id.run do
  let want := proved draw
  let mut c : Check := {}
  if int? instances != some (want.size : Int) then
    c := c.fail "instances" (Json.mkObj [("statement", instances), ("drawn", Lean.toJson want.size)])
  if provedUnits.map int? != want.map some then
    c := c.fail "units" (Json.mkObj [("statement", Json.arr (provedUnits.extract 0 16)), ("drawn", intsJson (want.extract 0 16))])
  return c

def checkProved (draw : Json) (provedUnits : Array Json) (instances : Json) : Array String :=
  (checkProvedFull draw provedUnits instances).codes

/-- `src`'s bytes handed out in order, `need` at a time (`OwnRandomness.read`); an error when fewer are left. -/
private def take (src : ByteArray) (need : Nat) : StateT Nat (Except String) ByteArray := do
  let at_ ← get
  if src.size < at_ + need then throw "the randomness source ran out"
  set (at_ + need)
  return src.extract at_ (at_ + need)

/-- `two_stage.draw_first` without a closure map: the one-stage draw object of the verifier's law over `[0, population)`
(the law as `derive` expands it, with `population` and the drawn `units`), sampled as `Flock.Draw.drawWith` samples it
from `src` read in order: `subset` and `bernoulli` (at its `NUM/DEN` as written) on one stream, `stratified` and `work` one
stream per stratum, each extended and rerun when it runs out. An error when the law is not exactly a one-stage law's keys
(`two_stage._law_problem`), a subset exceeds the population, a stratified or work law isn't well formed over it
(`strataProblem`), or `src` runs out. -/
def drawFirst (law : Json) (population : Nat) (src : ByteArray) : Except String Json := do
  let kind := match getD law "law" with | .str s => s | _ => ""
  let flock ← match kind with
    | "subset" =>
      if keys? law != some (sortStrings #["law", "k"]) then throw "a subset law is exactly {law, k}"
      let some k := nat? (getD law "k") | throw "a subset law's k is a natural number"
      if population < k then throw s!"a subset draw of {k} exceeds the population of {population}"
      pure (Flock.Draw.Law.subset k)
    | "bernoulli" =>
      if keys? law != some (sortStrings #["law", "p"]) then throw "a bernoulli law is exactly {law, p}"
      let .str p := getD law "p" | throw "a bernoulli law's p is NUM/DEN"
      let some (a, b) := probability? p | throw "a bernoulli law's p is NUM/DEN with 0 <= NUM <= DEN and DEN > 0"
      pure (.bernoulli a b)
    | "stratified" | "work" =>
      if keys? law != some (sortStrings #["law", "k", "strata"]) then throw s!"a {kind} law is exactly \{law, k, strata}"
      let (some k, .arr ss) := (nat? (getD law "k"), getD law "strata")
        | throw s!"a {kind} law's k is a natural number and its strata a list"
      if strataProblem law population then throw s!"the {kind} law's strata are not well formed over {population} units"
      let some strata := ss.mapM (stratum? (kind == "work")) | throw s!"the {kind} law's strata"
      pure (if kind == "work" then .work k strata else .stratified k (strata.map (·.toStratum)))
    | _ => throw "a one-stage law is subset, bernoulli, stratified or work"
  let (some d, _) ← (Flock.Draw.drawWith (fun _ need => take src need) flock population).run 0
    | throw "the draw did not end"
  return Json.mkObj (entries law |>.toList |>.append [("population", Lean.toJson population), ("units", natsJson d.units)])

end SampledProofs.Draw
