/-
The challenger's sample from a Spec's selection rule on (`service.selection_law`, `draw.derive` and `service.select_draw`,
which `Stream.register` and `Stream.select` run): the sampling design the rule names over the population of the verifier's own
program and query, that sampling design expanded over them, and the sample from the challenger's own bytes (every unit for `All`), with
a work sample's closure under the challenger's closure map, held to `Draw.checkDraw` as the stream holds it. With the sample,
the number of bytes of the coins it read.

A rule's law text is a rate: a rate law (`parseRate`: `rate:NUM/DEN[:T]`), or a stratified or work rate law
(`parseStrataRate`: `stratified-rate:NUM/DEN`, `work-rate:NUM/DEN`). The population a rule samples is the registration's,
which the prover sizes, so a fixed count (`subset:K`, `stratified:K`, `work:K`), whose proof blocks are each sampled with
`K/N` as the prover pads `N`, is refused (`Rule.ofJson`, `FIXED`); `bernoulli:NUM/DEN` parses and is refused at
registration (`Draw.bernoulli`). A rate names no sampling design until the population is known. Over `n` units a rate law
is `subset:k` at `k = Rate.count` (`Rule.over`, `resolve`), so the sample, the registration's law and the audit are a
uniform `k`-subset's, and `select` runs on that rule. A stratified rate law is `stratified:K` at the `K` its strata need
(`Rate.stratK` over `strataSizes`), and a work rate law `work:K` at `K = ⌈NUM·n/DEN⌉` over the work table
`Rule.tableOver` writes, every stratum's floor at the rate (`Rate.floor`), so under either each stratum samples at least
`⌈NUM·n_s/DEN⌉` of its `n_s` proof blocks (`Rate.stratK_ge`, `Rate.workK_ge`).

A two-stage rule (`Rule.twoStage grain replay k`) samples twice. Its first stage is `select` over the replay blocks, the
population of the boundary registration's program under the replay grain's query, by its replay law (a rate law); its
second stage (`selectSecond`, on fresh coins after the interior registration) samples `k_t` proof blocks of each cell, a
sampled replay block's `n_t` instances of template `t` (`replayCells`, `Counts.over`), one stratum per cell over the
template-major interior (`secondLaw`). One-stage is the rule `law`: a two-stage rule whose replay blocks are single proof
blocks is refused, so each design has one form.
-/
module

public import Spot.Draw
public import Spot.Rate

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Select

open Lean (Json)
open Flock (V)
/-- `draw.digits`: one or more ASCII digits. -/
private def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- Python's `s.partition(sep)` without the separator: `s` split at its first `sep`, and `(s, "")` when it has none. -/
private def split1 (s sep : String) : String × String :=
  match s.splitOn sep with
  | a :: rest => (a, sep.intercalate rest)
  | [] => (s, "")

/-- `draw.parse_design`: `subset:K`, `stratified:K` or `work:K` (`K ≥ 1`), or `bernoulli:NUM/DEN` with `0 < DEN` and
`NUM ≤ DEN`, split at the first `:` and the first `/`, each number one or more ASCII digits; the sampling design object, a Bernoulli
law's `p` written without leading zeros. -/
def parseDesign (text : String) : V Json := do
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

/-- `draw.parse_strata_rate`: the stratified rate law `stratified-rate:NUM/DEN` or the work rate law `work-rate:NUM/DEN`,
a public rate `NUM/DEN` with `0 < NUM ≤ DEN`, split at the first `:` and then the first `/`, each number one or more ASCII
digits, as `(work, NUM, DEN)`, `work` true for the work rate law; none for any other text. -/
def parseStrataRate (text : String) : Option (Bool × Nat × Nat) :=
  let (kind, arg) := split1 text ":"
  let (num, den) := split1 arg "/"
  if (kind == "stratified-rate" || kind == "work-rate") && digits num && digits den && 0 < num.toNat! &&
      num.toNat! ≤ den.toNat! then
    some (kind == "work-rate", num.toNat!, den.toNat!)
  else none

/-- A rule's law text is a rate: a rate law, a stratified or work rate law, or `bernoulli:NUM/DEN`, which registration
refuses (`Draw.bernoulli`). -/
def rate? (t : String) : Bool :=
  (parseRate t).isSome || (parseStrataRate t).isSome ||
    match parseDesign t with
    | .ok l => jeq (getD l "law") "bernoulli"
    | .error _ => false

/-- The refusal of a law text that is no rate (`rate?`), `service.FIXED`'s: a fixed count's proof blocks are each sampled
with `K/N` as the prover pads the `N` it registers. -/
def FIXED (t : String) : String :=
  s!"a law is rate:NUM/DEN or rate:NUM/DEN:T (0 < NUM <= DEN, T >= 1), stratified-rate:NUM/DEN or work-rate:NUM/DEN " ++
    s!"(0 < NUM <= DEN) or bernoulli:NUM/DEN, not {pyRepr t}: a fixed count samples each proof block with K/N, which " ++
    "falls as the prover pads the N it registers"

/-- A two-stage rule's second-stage counts (`service.TwoStage.k`): `k` in every cell, or `k_t` in each cell of template
`t`, by its descriptor id. -/
inductive Counts where
  | each (k : Nat)
  | per (ks : Array (String × Nat))

/-- The selection rules a `Stream` runs: every unit (`All`), a one-stage law, a rate (`rate?`, `Law`), and a two-stage
design (`TwoStage`): replay blocks, the template instances of the replay partition's query `grain`, sampled by the first
stage's law `replay` (a rate law), then `k`'s count of proof blocks inside each cell of each sampled replay block
(`selectSecond`). -/
inductive Rule where
  | all
  | law (text : String)
  | twoStage (grain : Json) (replay : String) (k : Counts)

/-- A two-stage rule's counts as `to_json` writes them: a positive integer, or an object of positive integers, one per
template's descriptor id, at least one. -/
def Counts.ofJson? (j : Json) : Option Counts :=
  match nat? j, j with
  | some k, _ => if k == 0 then none else some (.each k)
  | none, .obj _ =>
    if (entries j).isEmpty then none
    else
      let ks : Option (Array (String × Nat)) := (entries j).mapM fun (t, v) =>
        (nat? v).bind fun k => if k == 0 then none else some (t, k)
      ks.map .per
  | none, _ => none

/-- `service.second_counts`: the count in each cell of template `ids[t]`, `n_t = ns[t]` proof blocks: `k` (or the
template's `k_t`), and `n_t` for a cell of fewer (proved whole). An error for counts that don't name each of `ids`
once. -/
def Counts.over : Counts → Array String → Array Nat → V (Array Nat)
  | .each k, _, ns => return ns.map (min k ·)
  | .per ks, ids, ns => do
    if ks.size != ids.size || !(ids.all fun t => ks.any (·.1 == t)) then
      throw s!"a two-stage rule's counts name each of the check's templates once: {ks.map (·.1)} vs {ids}"
    return (ids.zip ns).map fun (t, n) => min (((ks.find? (·.1 == t)).map (·.2)).getD 0) n

/-- A replay grain as `partition_object.template_instance_query` writes it: exactly `{name, version, params}`,
`Q_template_instance` v0 over one template's descriptor id. -/
def grain? (g : Json) : Bool :=
  keys? g == some #["name", "params", "version"] && jeq (getD g "name") "Q_template_instance" &&
    jeq (getD g "version") (Lean.toJson (0 : Nat)) && keys? (getD g "params") == some #["template"] &&
    (getD (getD g "params") "template" matches .str _)

/-- A two-stage rule's first-stage law text: a rate law, a uniform sample of replay blocks at the rate however many replay
blocks the prover registers. -/
def replayLaw? (t : String) : Bool := (parseRate t).isSome

/-- A rule as `to_json` writes it, exactly its keys: `{rule: all}`, `{rule: law, law}`, its law text a rate (`rate?`), as
`service.Law` refuses any other when it is made (`FIXED`), or `{rule: two-stage, grain, replay, k}`, its grain the replay
partition's query (a `Q_template_instance` query object), its replay law a rate law (`replayLaw?`) and `k` a positive
integer or an object of them by template (`Counts.ofJson?`), as `service.TwoStage` refuses any other. A stream refuses a
sequential rule. -/
def Rule.ofJson (j : Json) : V Rule := do
  match getD j "rule" with
  | .str "all" => if keys? j == some #["rule"] then return .all
  | .str "law" =>
    if keys? j == some #["law", "rule"] then
      if let .str t := getD j "law" then
        if !rate? t then throw (FIXED t)
        return .law t
  | .str "two-stage" =>
    if keys? j == some #["grain", "k", "replay", "rule"] then
      let grain := getD j "grain"
      match getD j "replay" with
      | .str t =>
        if grain? grain then
          if !replayLaw? t then
            throw s!"a two-stage rule's replay law is a rate law, rate:NUM/DEN or rate:NUM/DEN:T, not {pyRepr t}"
          let some k := Counts.ofJson? (getD j "k")
            | throw "a two-stage rule samples k >= 1 proof blocks in each cell: one count, or one per template {descriptor id: k_t}"
          return .twoStage grain t k
      | _ => pure ()
  | _ => pure ()
  throw "a selection rule the stream runs is {rule: all}, {rule: law, law} or {rule: two-stage, grain, replay, k}"

/-- `service.selection_law`, after `Rule.over`: the sampling design a rule names over `n` units, before `Draw.derive` expands it; a
two-stage rule's is its first stage's, over `n` replay blocks. -/
def Rule.lawOver : Rule → Nat → V Json
  | .all, n => return Json.mkObj [("law", "subset"), ("k", Lean.toJson n)]
  | .law t, _ => parseDesign t
  | .twoStage _ t _, _ => parseDesign t

/-- A sampling design text over `n` units in strata of `sizes` units: a rate law is `subset:k` at
`k = Rate.count NUM DEN T n`, a stratified rate law `stratified:k` at `k = Rate.stratK NUM DEN sizes n`, none without
strata, and a work rate law `work:k` at `k = Rate.count NUM DEN 1 n` over `Rule.tableOver`'s work table, each none unless
`1 ≤ k ≤ n`; any other text is itself. -/
def textOver (t : String) (n : Nat) (sizes : List Nat) : Option String :=
  match parseRate t, parseStrataRate t with
  | some (num, den, tables), _ =>
    let k := Rate.count num den tables n
    if 1 ≤ k && k ≤ n then some s!"subset:{k}" else none
  | none, some (false, num, den) =>
    let k := Rate.stratK num den sizes n
    if !sizes.isEmpty && 1 ≤ k && k ≤ n then some s!"stratified:{k}" else none
  | none, some (true, num, den) =>
    let k := Rate.count num den 1 n
    if 1 ≤ k && k ≤ n then some s!"work:{k}" else none
  | none, none => some t

/-- `service.selection_table`: the work table a rule's sampling design is expanded over, from the verifier's own `table`.
A work rate law's names each stratum of `p` under `query` (`Partition.strata`) with the table's work and its floor at the
rate, `Rate.floor NUM DEN n_s f_s`, so each stratum samples at least `⌈NUM·n_s/DEN⌉` of its `n_s` proof blocks. Any other
rule's is `table`. An error where the query doesn't apply, a work rate law has no table or its table names no work for a
stratum's template. -/
def Rule.tableOver (rule : Rule) (p : Program) (query : Json) (table : Option Json) : V (Option Json) := do
  let .law t := rule | return table
  let some (true, num, den) := parseStrataRate t | return table
  let strata ← Partition.strata p query
  let given ← match table with
    | some j => Flock.Draw.workTableOfJson j
    | none => throw "a work rate law needs the verifier's work table"
  let entries ← strata.mapM fun (tmpl, rs) => do
    let n := rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0
    let some (_, w, f) := given.find? (·.1 == tmpl) | throw s!"the work table names no work for template '{tmpl}'"
    pure (tmpl, Json.mkObj [("work", Lean.toJson w), ("floor", Lean.toJson (Rate.floor num den n f))])
  return some (Json.mkObj entries.toList)

/-- `draw.strata_sizes`: the units of each stratum of `p` under `query` (`Partition.strata`), which a stratified rate law's
`K` reads (`textOver`); none where the query doesn't apply. -/
def strataSizes (p : Program) (query : Json) : List Nat :=
  match Partition.strata p query with
  | .ok ss => ss.toList.map fun (_, rs) => rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0
  | .error _ => []

/-- `service.selection_law`'s first step: the rule over `n` units in strata of `sizes` units (`strataSizes`), its rate
`subset:k`, `stratified:k` or `work:k` (`textOver`); none where `selection_law` refuses (`rate`). It follows `lawOver`,
whose matcher it shares, so that `lawOver` stays the definition the Properties lock records. -/
def Rule.over : Rule → Nat → List Nat → Option Rule
  | .all, _, _ => some .all
  | .law t, n, sizes => (textOver t n sizes).map .law
  | .twoStage g t k, n, sizes => (textOver t n sizes).map (.twoStage g · k)

/-- The templates a template query names, in its order (`Flock.TemplateQuery.templatesOf`); an error for any other
query. -/
def templateIds (query : Json) : V (Array String) :=
  match Flock.TemplateQuery.templatesOf ((query.getObjValAs? String "name").toOption.getD "")
      ((nat? (getD query "version")).getD 0) (getD query "params") with
  | some ts => ts
  | none => throw "a two-stage design's check is a template query"

/-- `service.replay_cells`: each template's proof blocks in one replay block, `n_t`, in the order the check's template
query names them: the query applied to the replay block's program (`Partition.evaluate`), counted by template. An
error where the query doesn't apply, and where it doesn't name each template of the replay block once, in the order of
its first instance in it, so that template `t`'s cells are the statement's stratum `t`. -/
def replayCells (p : Program) (query : Json) : V (Array Nat) := do
  let rs ← Partition.evaluate p query
  let ids ← templateIds query
  let first := rs.foldl (fun acc r => if acc.contains r.template then acc else acc.push r.template) #[]
  if first != ids then
    throw s!"the check's templates {ids} are not the replay block's, each once, in the order of its first instance in it ({first})"
  return ids.map fun t => rs.foldl (fun s r => if r.template == t then s + r.count else s) 0

/-- The cell of `n` proof blocks from `o + i·n` of template `t` in sampled replay block `r` sampling `k`: a work stratum of
`secondLaw`, named `r/t`, of unit work and floor `k`. -/
def cellObj (o n k i r t : Nat) : Json :=
  Json.mkObj [("template", Json.str s!"{r}/{t}"), ("units", Json.arr #[natsJson #[o + i * n, o + (i + 1) * n]]),
    ("work", Lean.toJson (1 : Nat)), ("floor", Lean.toJson k), ("k", Lean.toJson k)]

/-- `service.second_law`: the second stage's law over the interior's `sample.size · Σ n_t` proof blocks, template-major:
template `t`'s from `off_t = sample.size · Σ_{t'<t} n_t'`, the `i`-th sampled replay block's (replay block `sample[i]`
of the registration) instances of it `[off_t + i·n_t, off_t + (i+1)·n_t)`, one cell. One stratum per cell, template by
template and in each by sampled replay block, `ks[t]` of its `ns[t]` proof blocks in each (`cellObj`); the strata tile
`[0, sample.size · Σ n_t)` in order. The stratified law sizes each stratum's count in proportion to its size, which
cells of several templates' sizes don't follow, so the law is the work law at `k = 1` and unit work: its rule
(`Flock.Draw.workK`) gives each stratum `max floor 1` capped at its size, and each cell's floor is its count. -/
def secondLaw (sample : Array Nat) (ns ks : Array Nat) : Json :=
  Json.mkObj [("law", "work"), ("k", Lean.toJson (1 : Nat)),
    ("strata", Json.arr ((List.range ns.size).flatMap fun t => (List.range sample.size).map fun i =>
      cellObj (sample.size * (ns.toList.take t).sum) (ns.getD t 0) (ks.getD t 0) i (sample.getD i 0) t).toArray)]

/-- Why the second stage refuses a replay sample `sample` with cells of `ns` proof blocks sampling `ks`, the first
reason in the Python's order: a replay sample with a block twice or out of order, cells that aren't one or more
templates' of at least one proof block each with a count each, a replay block of fewer than two proof blocks, a count
not in `[1, n_t]`, or counts proving every cell whole (whole replay blocks). -/
def secondProblem (sample : Array Nat) (ns ks : Array Nat) : Option String :=
  if sample.isEmpty || !(sample.toList.zip sample.toList.tail).all (fun (a, b) => a < b) then
    some "a replay sample is one or more replay blocks, ascending, each once"
  else if ns.isEmpty || ks.size != ns.size || ns.toList.any (· == 0) then
    some "a replay block's cells are one or more templates', each of at least one proof block, with a count each"
  else if ns.toList.sum < 2 then
    some s!"a replay block of {ns.toList.sum} proof blocks is a proof block: its rule is {pyRepr "law"}"
  else if !(ks.toList.zip ns.toList).all (fun (k, n) => 1 ≤ k && k ≤ n) then
    some s!"the second stage samples 1 to n_t of each cell's n_t proof blocks: counts {ks} over cells of {ns}"
  else if ks == ns then some s!"counts {ks} prove every cell of {ns} whole, and so whole replay blocks"
  else none

/-- `service.select_second`: the second stage on the coins `src`, read after the interior registration's digest: the
sample object of `secondLaw` over the interior's `sample.size · Σ n_t` proof blocks (`Draw.drawFirstRead`), held to
`Draw.checkDraw`, and the number of bytes it read. An error where the Python raises: `secondProblem`, or a sample the
coins don't end. -/
def selectSecond (sample : Array Nat) (ns ks : Array Nat) (src : ByteArray) : V (Json × Nat) := do
  if let some e := secondProblem sample ns ks then throw e
  let law := secondLaw sample ns ks
  let m := sample.size * ns.toList.sum
  let (draw, read) ← Draw.drawFirstRead law m src
  let codes := Draw.checkDraw draw (Lean.toJson m) law
  if !codes.isEmpty then throw s!"the second stage's draw is not well formed under its law: {codes}"
  return (draw, read)

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

/-- The rule over the population of `p` under `query` (`Partition.units`, `Rule.over` over its `strataSizes`), which
`select` runs over `Rule.tableOver`'s work table; an error where the query doesn't apply or `selection_law` refuses the
rule's rate. -/
def resolve (rule : Rule) (p : Program) (query : Json) : V Rule := do
  let n ← Partition.units p query
  let some r := rule.over n (strataSizes p query)
    | throw s!"a rate law over {n} units draws none of them or more than {n}"
  return r

/-- `selection_law`, `draw.derive` and `select_draw` on the coins `src`, then `Stream.select`'s `draw.check_draw`, under
a rule `resolve` gives (`parseDesign` refuses a rate law itself): the
sample object over the population of `p` under `query` (`Partition.units`), the units it proves (`Draw.proved`'s: the sampled
proof blocks, then a work sample's closure) and the number of bytes of `src` it read. The sampling design is the rule's over that population,
expanded (`Draw.derive`, with the work `table`); `All` samples every unit and reads nothing, any other rule samples its Flock
law by `Draw.drawUnits`, as `Draw.drawFirstRead` does: the sampling design `Draw.flockLaw` reads from the sampling design object, or, for
`stratified` and `work`, once `flockLaw` accepts that object, the expansion it was written from (`Draw.deriveLaw`), the
same law. With the closure map `cm`, a work sample carries
the sampled proof blocks' closure (`Flock.Draw.closureOf`) when it is not empty. An error where the Python raises: the query
doesn't apply, the rule's law or its expansion is refused, a closure map is given for another law or names a unit outside
the population, the sample is refused (`drawFirstRead`), or it is not well formed under the sampling design (`Draw.checkDraw`). -/
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

end Spot.Select
