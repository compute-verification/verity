/-
The challenger's draw from a Spec's selection rule on (`service.selection_law`, `draw.derive` and `service.select_draw`,
which `Stream.register` and `Stream.select` run): the law the rule names over the population of the verifier's own
program and query, that law expanded over them, and the draw from the challenger's own bytes (every unit for `All`), with
a work draw's closure under the challenger's closure map, held to `Draw.checkDraw` as the stream holds it. With the draw,
the number of bytes of the coins it read.
-/
import SampledProofs.Draw

namespace SampledProofs.Select

open Lean (Json)
open Flock (V)
open Flock.TemplateQuery (Desc)

/-- `draw.digits`: one or more ASCII digits. -/
private def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- Python's `s.partition(sep)` without the separator: `s` split at its first `sep`, and `(s, "")` when it has none. -/
private def split1 (s sep : String) : String × String :=
  match s.splitOn sep with
  | a :: rest => (a, sep.intercalate rest)
  | [] => (s, "")

/-- `draw.parse_law`: `subset:K`, `stratified:K` or `work:K` (`K ≥ 1` for the last two), or `bernoulli:NUM/DEN` with
`0 < DEN` and `NUM ≤ DEN`, split at the first `:` and the first `/`, each number one or more ASCII digits; the law object,
a Bernoulli law's `p` written without leading zeros. -/
def parseLaw (text : String) : V Json := do
  let (kind, arg) := split1 text ":"
  if kind == "subset" && digits arg then
    return Json.mkObj [("law", "subset"), ("k", Lean.toJson arg.toNat!)]
  if (kind == "stratified" || kind == "work") && digits arg && 1 ≤ arg.toNat! then
    return Json.mkObj [("law", Json.str kind), ("k", Lean.toJson arg.toNat!)]
  if kind == "bernoulli" then
    let (num, den) := split1 arg "/"
    if digits num && digits den && 0 < den.toNat! && num.toNat! ≤ den.toNat! then
      return Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{num.toNat!}/{den.toNat!}")]
  throw s!"a law is subset:K, stratified:K, work:K or bernoulli:NUM/DEN, not {pyRepr text}"

/-- The selection rules a `Stream` runs: every unit (`All`), a one-stage law in `parseLaw`'s syntax (`Law`), and a
two-stage rule's replay law with every proof unit of a drawn RU checked (`TwoStage` at `check` `"all"`, each RU one
instance of the check's unit). -/
inductive Rule where
  | all
  | law (text : String)
  | twoStage (replay : String)

/-- A rule as `to_json` writes it, exactly its keys: `{rule: all}`, `{rule: law, law}` or
`{rule: two-stage, replay, check: all}`. A stream refuses a two-stage rule's count and a sequential rule. -/
def Rule.ofJson (j : Json) : V Rule := do
  match getD j "rule" with
  | .str "all" => if keys? j == some #["rule"] then return .all
  | .str "law" =>
    if keys? j == some #["law", "rule"] then
      if let .str t := getD j "law" then return .law t
  | .str "two-stage" =>
    if keys? j == some #["check", "replay", "rule"] && jeq (getD j "check") "all" then
      if let .str t := getD j "replay" then return .twoStage t
  | _ => pure ()
  throw "a selection rule the stream runs is {rule: all}, {rule: law, law} or {rule: two-stage, replay, check: all}"

/-- `service.selection_law`: the law a rule names over `n` units, before `Draw.derive` expands it. -/
def Rule.lawOver : Rule → Nat → V Json
  | .all, n => return Json.mkObj [("law", "subset"), ("k", Lean.toJson n)]
  | .law t, _ | .twoStage t, _ => parseLaw t

/-- The challenger's closure map as `[[unit, [units]], ...]`, integers, each unit named once. -/
def closureMap? (j : Json) : V (Array (Int × Array Int)) := do
  let .arr ps := j | throw "a closure map is a list of [unit, [units]]"
  let m ← ps.mapM fun p => do
    let .arr p := p | throw "a closure map's entry is [unit, [units]]"
    let (some u, .arr cs) := (if p.size == 2 then (int? p[0]!, p[1]!) else (none, Json.null))
      | throw "a closure map's entry is [unit, [units]], integers"
    let some cs := cs.mapM int? | throw "a closure map's entry is [unit, [units]], integers"
    pure (u, cs)
  if (m.map (·.1)).toList.eraseDups.length != m.size then throw "a closure map names a unit once"
  return m

private def inRange (n : Nat) (u : Int) : Bool := 0 ≤ u && u < n

/-- `selection_law`, `draw.derive` and `select_draw` on the coins `src`, then `Stream.select`'s `draw.check_draw`: the
draw object over the population of `d` under `query` (`Partition.units`), the units it proves (`Draw.proved`'s: the drawn
units, then a work draw's closure) and the number of bytes of `src` it read. The law is the rule's over that population,
expanded (`Draw.derive`, with the work `table`); `All` draws every unit and reads nothing, any other rule draws its C-Flock
law (`Draw.flockLaw`) by `Draw.drawUnits`, as `Draw.drawFirstRead` does. With the closure map `cm`, a work draw carries
the drawn units' closure (`Flock.Draw.closureOf`) when it is not empty. An error where the Python raises: the query
doesn't apply, the rule's law or its expansion is refused, a closure map is given for another law or names a unit outside
the population, the draw is refused (`drawFirstRead`), or it is not well formed under the law (`Draw.checkDraw`). -/
def select (rule : Rule) (d : Desc) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Json × Array Nat × Nat) := do
  let n ← Partition.units d query
  let law ← Draw.derive d query (← rule.lawOver n) table
  if let some cm := cm then
    if !(jeq (getD law "law") "work" && cm.all fun (u, cs) => inRange n u && cs.all (inRange n)) then
      throw s!"a closure map over [0, {n}) widens a work draw only"
  let (units, read) ← match rule with
    | .all => pure (Array.range n, 0)
    | _ => Draw.drawUnits (← Draw.flockLaw law n) n src
  let draw := Json.mkObj ((entries law).toList ++ [("population", Lean.toJson n), ("units", natsJson units)])
  let closure := match cm with
    | some cm => Flock.Draw.closureOf (cm.foldl (fun m (u, cs) => m.insert u.toNat (cs.map Int.toNat)) {}) units
    | none => #[]
  let draw := if closure.isEmpty then draw else Json.mkObj ((entries draw).toList ++ [("closure", natsJson closure)])
  let codes := Draw.checkDraw draw (Lean.toJson n) law
  if !codes.isEmpty then throw s!"the draw is not well formed under the law: {codes}"
  return (draw, units ++ closure, read)

end SampledProofs.Select
