/-
The challenger's draw from a Spec's selection rule on (`service.selection_law`, `draw.derive` and `service.select_draw`,
which `Stream.register` and `Stream.select` run): the law the rule names over the population of the verifier's own
program and query, that law expanded over them, and the draw from the challenger's own bytes (every unit for `All`), with
a work draw's closure under the challenger's closure map, held to `Draw.checkDraw` as the stream holds it. With the draw,
the number of bytes of the coins it read.

A rule's law text is a rate law (`parseRate`: `rate:NUM/DEN[:T]`, what a new Spec states) or one `parseLaw` reads. A rate
law names no law until the population is known: over `n` units it is `subset:k` at `k = Rate.count` (`Rule.over`,
`resolve`), so the draw, the registration's law and the audit are a uniform `k`-subset's, and `select` runs on that rule.
-/
import SampledProofs.Draw
import SampledProofs.Rate

namespace SampledProofs.Select

open Lean (Json)
open Flock (V)
/-- `draw.digits`: one or more ASCII digits. -/
private def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- Python's `s.partition(sep)` without the separator: `s` split at its first `sep`, and `(s, "")` when it has none. -/
private def split1 (s sep : String) : String × String :=
  match s.splitOn sep with
  | a :: rest => (a, sep.intercalate rest)
  | [] => (s, "")

/-- `draw.parse_law`: `subset:K`, `stratified:K` or `work:K` (`K ≥ 1`), or `bernoulli:NUM/DEN` with `0 < DEN` and
`NUM ≤ DEN`, split at the first `:` and the first `/`, each number one or more ASCII digits; the law object, a Bernoulli
law's `p` written without leading zeros. -/
def parseLaw (text : String) : V Json := do
  let (kind, arg) := split1 text ":"
  if kind == "subset" && digits arg && 1 ≤ arg.toNat! then
    return Json.mkObj [("law", "subset"), ("k", Lean.toJson arg.toNat!)]
  if (kind == "stratified" || kind == "work") && digits arg && 1 ≤ arg.toNat! then
    return Json.mkObj [("law", Json.str kind), ("k", Lean.toJson arg.toNat!)]
  if kind == "bernoulli" then
    let (num, den) := split1 arg "/"
    if digits num && digits den && 0 < den.toNat! && num.toNat! ≤ den.toNat! then
      return Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{num.toNat!}/{den.toNat!}")]
  throw s!"a law is subset:K, stratified:K, work:K or bernoulli:NUM/DEN, not {pyRepr text}"

/-- `draw.parse_rate`: the rate law `rate:NUM/DEN` or `rate:NUM/DEN:T`, a public rate `NUM/DEN` with `0 < NUM ≤ DEN` over
`T ≥ 1` tables (one when it names none), split at the first `:`, then at the first `/` and `:`, each number one or more
ASCII digits, as `(NUM, DEN, T)`; none for any other text. -/
def parseRate (text : String) : Option (Nat × Nat × Nat) :=
  let (kind, arg) := split1 text ":"
  let (q, tables) : String × Option Nat := match arg.splitOn ":" with
    | [q] => (q, some 1)
    | q :: rest => let t := ":".intercalate rest; (q, if digits t then some t.toNat! else none)
    | [] => (arg, none)
  let (num, den) := split1 q "/"
  match tables with
  | some t =>
    if kind == "rate" && digits num && digits den && 0 < num.toNat! && num.toNat! ≤ den.toNat! && 1 ≤ t then
      some (num.toNat!, den.toNat!, t)
    else none
  | none => none

/-- The selection rules a `Stream` runs: every unit (`All`) and a one-stage law, a rate law (`parseRate`) or one in
`parseLaw`'s syntax (`Law`). -/
inductive Rule where
  | all
  | law (text : String)

/-- The refusal of a two-stage selection rule (`service.RETIRED_TWO_STAGE`). -/
def retiredTwoStage : String :=
  "two-stage sampling is retired (everything is committed at serve time); a selection rule is all or a law"

/-- A rule as `to_json` writes it, exactly its keys: `{rule: all}` or `{rule: law, law}`, its law text a rate law or one
`parseLaw` reads, as `service.Law` refuses any other when it is made. A stream refuses a sequential rule, and a two-stage
rule with `retiredTwoStage`: two-stage sampling is retired. -/
def Rule.ofJson (j : Json) : V Rule := do
  match getD j "rule" with
  | .str "all" => if keys? j == some #["rule"] then return .all
  | .str "law" =>
    if keys? j == some #["law", "rule"] then
      if let .str t := getD j "law" then
        if (parseRate t).isNone then let _ ← parseLaw t
        return .law t
  | .str "two-stage" => throw retiredTwoStage
  | _ => pure ()
  throw "a selection rule the stream runs is {rule: all} or {rule: law, law}"

/-- `service.selection_law`, after `Rule.over`: the law a rule names over `n` units, before `Draw.derive` expands it. -/
def Rule.lawOver : Rule → Nat → V Json
  | .all, n => return Json.mkObj [("law", "subset"), ("k", Lean.toJson n)]
  | .law t, _ => parseLaw t

/-- A law text over `n` units: a rate law is `subset:k` at `k = Rate.count NUM DEN T n`, none unless `1 ≤ k ≤ n`; any
other text is itself. -/
def textOver (t : String) (n : Nat) : Option String :=
  match parseRate t with
  | some (num, den, tables) =>
    let k := Rate.count num den tables n
    if 1 ≤ k && k ≤ n then some s!"subset:{k}" else none
  | none => some t

/-- `service.selection_law`'s first step: the rule over `n` units, its rate law `subset:k` (`textOver`); none where
`selection_law` refuses (`rate`). It follows `lawOver`, whose matcher it shares, so that `lawOver` stays the definition
the Properties lock records. -/
def Rule.over : Rule → Nat → Option Rule
  | .all, _ => some .all
  | .law t, n => (textOver t n).map .law

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

/-- The rule over the population of `p` under `query` (`Partition.units`, `Rule.over`), which `select` runs; an error
where the query doesn't apply or `selection_law` refuses the rule's rate law. -/
def resolve (rule : Rule) (p : Program) (query : Json) : V Rule := do
  let n ← Partition.units p query
  let some r := rule.over n | throw s!"a rate law over {n} units draws none of them or more than {n}"
  return r

/-- `selection_law`, `draw.derive` and `select_draw` on the coins `src`, then `Stream.select`'s `draw.check_draw`, under
a rule `resolve` gives (`parseLaw` refuses a rate law itself): the
draw object over the population of `p` under `query` (`Partition.units`), the units it proves (`Draw.proved`'s: the drawn
units, then a work draw's closure) and the number of bytes of `src` it read. The law is the rule's over that population,
expanded (`Draw.derive`, with the work `table`); `All` draws every unit and reads nothing, any other rule draws its C-Flock
law by `Draw.drawUnits`, as `Draw.drawFirstRead` does: the law `Draw.flockLaw` reads from the law object, or, for
`stratified` and `work`, once `flockLaw` accepts that object, the expansion it was written from (`Draw.deriveLaw`), the
same law. With the closure map `cm`, a work draw carries
the drawn units' closure (`Flock.Draw.closureOf`) when it is not empty. An error where the Python raises: the query
doesn't apply, the rule's law or its expansion is refused, a closure map is given for another law or names a unit outside
the population, the draw is refused (`drawFirstRead`), or it is not well formed under the law (`Draw.checkDraw`). -/
def select (rule : Rule) (p : Program) (query : Json) (table : Option Json) (cm : Option (Array (Int × Array Int)))
    (src : ByteArray) : V (Json × Array Nat × Nat) := do
  let n ← Partition.units p query
  let named ← rule.lawOver n
  let derived ← Draw.deriveLaw p query named table
  let law := (derived.map (·.toJson)).getD named
  if let some cm := cm then
    if !(jeq (getD law "law") "work" && cm.all fun (u, cs) => inRange n u && cs.all (inRange n)) then
      throw s!"a closure map over [0, {n}) widens a work draw only"
  let (units, read) ← match rule with
    | .all => pure (Array.range n, 0)
    | _ => do
      let parsed ← Draw.flockLaw law n
      Draw.drawUnits (derived.getD parsed) n src
  let draw := Json.mkObj ((entries law).toList ++ [("population", Lean.toJson n), ("units", natsJson units)])
  let closure := match cm with
    | some cm => Flock.Draw.closureOf (cm.foldl (fun m (u, cs) => m.insert u.toNat (cs.map Int.toNat)) {}) units
    | none => #[]
  let draw := if closure.isEmpty then draw else Json.mkObj ((entries draw).toList ++ [("closure", natsJson closure)])
  let codes := Draw.checkDraw draw (Lean.toJson n) law
  if !codes.isEmpty then throw s!"the draw is not well formed under the law: {codes}"
  return (draw, units ++ closure, read)

end SampledProofs.Select
