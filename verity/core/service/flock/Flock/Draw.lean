/-
PROTOCOL.md §7.3: the unit draw. After serving registers its roots and before the session's statements are fixed, the
verifier draws the sampled units from its own randomness with exact samplers and sends them in the clear. The drawn units'
statement carries the draw (the public header's `unit_draw`, under Σ) and the session record keeps it. Offline, the
verifier checks that the two agree, that the draw is well formed, and that the statement's instances are the drawn units.
-/
import Lean.Data.Json
import Std.Data.HashMap
import Flock.Canon

namespace Flock.Draw

open Lean (Json)

/-- Bytes per word for `uniform n`: `n`'s bit length plus 64, rounded up to bytes (as `verity.randomness.Key.uniform`). -/
def width (n : Nat) : Nat := (Nat.log2 n + 1 + 64 + 7) / 8

def beNat (b : ByteArray) (off len : Nat) : Nat :=
  (List.range len).foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) 0

/-- An exactly uniform integer in `[0, n)` from the stream at `pos` (`verity.randomness.uniform_from_words`): the first
`width n`-byte big-endian word below the largest multiple of `n` under `2^(8·width)`, mod `n`, and the position after it.
`none` when the stream runs out. -/
def uniform (n : Nat) (src : ByteArray) : Nat → Nat → Option (Nat × Nat)
  | 0, _ => none
  | fuel + 1, pos =>
    let w := width n
    if pos + w > src.size then none
    else
      let space := 2 ^ (8 * w)
      let x := beNat src pos w
      if x < space - space % n then some (x % n, pos + w) else uniform n src fuel (pos + w)

/-- A uniformly random `k`-subset of `[0, n)`, sorted (`Key.subset`'s partial Fisher–Yates, on one stream). -/
def subset (k n : Nat) (src : ByteArray) : Option (Array Nat) := Id.run do
  let mut sw : Std.HashMap Nat Nat := {}
  let mut pos := 0
  for i in [0:k] do
    match uniform (n - i) src (src.size + 1) pos with
    | none => return none
    | some (r, p) =>
      let j := i + r
      let (vi, vj) := (sw.getD i i, sw.getD j j)
      sw := (sw.insert i vj).insert j vi
      pos := p
  return some (((Array.range k).map fun i => sw.getD i i).qsort (· < ·))

/-- Each of `[0, n)` independently with probability exactly `num/den` (`Key.bernoulli`: `uniform den < num`). -/
def bernoulli (num den n : Nat) (src : ByteArray) : Option (Array Nat) := Id.run do
  if num == 0 then return some #[]
  if num == den then return some (Array.range n)
  let mut out := #[]
  let mut pos := 0
  for u in [0:n] do
    match uniform den src (src.size + 1) pos with
    | none => return none
    | some (r, p) =>
      if r < num then out := out.push u
      pos := p
  return some out

/-- A stratum of the stratified law: a template's units, as ascending, disjoint half-open ranges `[lo, hi)`, and its draw `k`. -/
structure Stratum where
  template : String
  units : Array (Nat × Nat)
  k : Nat
  deriving BEq, Inhabited

def Stratum.size (s : Stratum) : Nat := s.units.foldl (fun acc (lo, hi) => acc + (hi - lo)) 0

/-- The stratum's `j`-th unit, counting through its ranges. -/
def Stratum.unit (s : Stratum) (j : Nat) : Nat := Id.run do
  let mut j := j
  for (lo, hi) in s.units do
    if j < hi - lo then return lo + j
    j := j - (hi - lo)
  return 0

/-- A stratum's draw under the stratified rule (`draw.stratum_k`): `max(1, round(k · size / population))`, half up. -/
def stratumK (k size population : Nat) : Nat := max 1 ((2 * k * size + population) / (2 * population))

/-- The strata of a template query's per-node ranges `(template, first unit, count)` in canonical order (`partition.strata`):
one per template, its ranges with adjacent ones merged, in the order of their first unit. -/
def strataOf (ranges : Array (String × Nat × Nat)) : Array (String × Array (Nat × Nat)) := Id.run do
  let mut out : Array (String × Array (Nat × Nat)) := #[]
  for (t, lo, n) in ranges do
    match out.findIdx? (·.1 == t) with
    | none => out := out.push (t, #[(lo, lo + n)])
    | some i =>
      let rs := out[i]!.2
      let rs := match rs.back? with
        | some (a, b) => if b == lo then rs.pop.push (a, lo + n) else rs.push (lo, lo + n)
        | none => rs.push (lo, lo + n)
      out := out.set! i (t, rs)
  return out

/-- A stratum of the work law: a stratified law's stratum, the PoUW work of each of its units, an integer in any fixed unit
(`0` for units that do none: dequantization, the quantizer, attention), and its floor, the fewest units it draws (at least
one), both from the verifier's work table. -/
structure WorkStratum extends Stratum where
  work : Nat
  floor : Nat
  deriving BEq, Inhabited

/-- A stratum's draw under the work law: `⌈K · work · n_s / W⌉`, with `W` the population's work `Σ_s work_s · n_s`, never
below the stratum's floor and never above `n_s`. Rounding up keeps each unit's inclusion probability at least `K · work / W`,
which the work bound (`FlockSoundness.Audit.Law.work_escape_le`, through `workRule_eq_draw`) reads. -/
def workK (k work size total floor : Nat) : Nat := min size (max floor ((k * work * size + total - 1) / total))

/-- A work law's population work, `Σ_s work_s · n_s`. -/
def totalWork (strata : Array WorkStratum) : Nat := strata.foldl (fun acc s => acc + s.work * s.size) 0

inductive Law where
  | subset (k : Nat)
  | bernoulli (num den : Nat)
  /-- one uniform `k_s`-subset per stratum, independently (the default law; `k` is the agreement's) -/
  | stratified (k : Nat) (strata : Array Stratum)
  /-- the work law, the stratified law's second version: its sampler, each stratum's `k_s` sized by work (`workK`) above
  its floor -/
  | work (k : Nat) (strata : Array WorkStratum)
  deriving BEq, Inhabited

/-- `stratified:K` expanded over the strata (`draw.derive`): each stratum with `stratumK`, for `1 ≤ K ≤ population`. -/
def derive (k : Nat) (strata : Array (String × Array (Nat × Nat))) : Except String Law := do
  let n := strata.foldl (fun acc (_, rs) => acc + rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) 0
  if k < 1 || k > n then throw s!"a stratified law needs 1 <= k <= {n}, not {k}"
  return .stratified k (strata.map fun (t, rs) =>
    { template := t, units := rs, k := stratumK k (rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) n })

/-- `work:K` expanded over the strata and the verifier's work table (template descriptor id ↦ work per unit and floor):
each stratum with `workK` above its floor, for `1 ≤ K ≤ population` and a population that does some work. A template the
table does not name is refused, not taken as doing none. -/
def deriveWork (k : Nat) (strata : Array (String × Array (Nat × Nat))) (table : Array (String × Nat × Nat)) :
    Except String Law := do
  let n := strata.foldl (fun acc (_, rs) => acc + rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) 0
  if k < 1 || k > n then throw s!"a work law needs 1 <= k <= {n}, not {k}"
  let ws ← strata.mapM fun (t, rs) => match table.find? (·.1 == t) with
    | some (_, w, f) => pure ({ template := t, units := rs, k := 0, work := w, floor := f } : WorkStratum)
    | none => throw s!"the work table names no work for template '{t}'"
  let total := totalWork ws
  if total == 0 then throw "a work law needs a population that does some work"
  return .work k (ws.map fun s => { s with k := workK k s.work s.size total s.floor })

/-- A work table: a JSON object from template descriptor id to the PoUW work of one of its units, or to
`{"work": w, "floor": f}` with its floor `f ≥ 1` (one when the table gives only the work). -/
def workTableOfJson (j : Json) : Except String (Array (String × Nat × Nat)) := do
  let .ok kvs := j.getObj? | throw "a work table is an object from template id to work per unit"
  kvs.toArray.mapM fun (t, v) => match v.getNat?, v.getObj? with
    | .ok w, _ => pure (t, w, 1)
    | _, .ok o => do
      let bad := s!"work table: template '{t}': neither a work nor an object of work and floor"
      if o.toArray.any (fun (key, _) => key != "work" && key != "floor") then throw bad
      let some w := (v.getObjValAs? Nat "work").toOption | throw bad
      match v.getObjVal? "floor" with
      | .error _ => pure (t, w, 1)
      | .ok fj => match fj.getNat? with
        | .ok f => if f < 1 then throw s!"work table: template '{t}': a floor is at least one" else pure (t, w, f)
        | .error _ => throw bad
    | _, _ => throw s!"work table: template '{t}': the work is not a natural number"

/-- The bytes a stratum's stream starts with: `2 k_s · width(n_s) + 64`, as a `subset` draw's. -/
def Stratum.need (s : Stratum) : Nat := 2 * s.k * width s.size + 64

/-- The stratified draw, one stream per stratum: `subset k_s n_s` on stratum `s`'s stream, local index `j` mapped to the
stratum's `j`-th unit, the union ascending. `none` when a stream runs out. -/
def stratified (strata : Array Stratum) (streams : Array ByteArray) : Option (Array Nat) := do
  let mut out := #[]
  for i in [0:strata.size] do
    let s := strata[i]!
    let local_ ← subset s.k s.size (streams.getD i ByteArray.empty)
    out := out ++ local_.map s.unit
  return out.qsort (· < ·)

/-- A draw: its law, the population `[0, population)`, the drawn units, ascending, and for a closure draw the units
proved with them, their closure beyond the drawn units, ascending. -/
structure UnitDraw where
  law : Law
  population : Nat
  units : Array Nat
  closure : Array Nat := #[]
  deriving Inhabited

/-- The units a draw proves: the drawn units and their closure, ascending. -/
def UnitDraw.proved (d : UnitDraw) : Array Nat :=
  if d.closure.isEmpty then d.units else (d.units ++ d.closure).qsort (· < ·)

/-- A closure map: a JSON object from a unit's index to the units its credit depends on (for a tile, the strips it reads
and its path's nodes), the verifier's own, from its program. -/
def closureMapOfJson (j : Json) : Except String (Std.HashMap Nat (Array Nat)) := do
  let .ok kvs := j.getObj? | throw "a closure map is an object from unit index to the units its credit depends on"
  kvs.toArray.foldlM (init := {}) fun m (key, v) => do
    let some u := key.toNat? | throw s!"closure map: '{key}' is not a unit index"
    let .arr cs := v | throw s!"closure map: unit {u}: not a list of units"
    let cs ← cs.mapM fun c => match c.getNat? with
      | .ok x => pure x
      | .error _ => throw s!"closure map: unit {u}: not a list of units"
    return m.insert u cs

/-- A draw's closure under the closure map `cm`: every unit in a drawn unit's closure that is not itself drawn,
ascending. -/
def closureOf (cm : Std.HashMap Nat (Array Nat)) (units : Array Nat) : Array Nat := Id.run do
  let drawn : Std.HashMap Nat Unit := units.foldl (fun m u => m.insert u ()) {}
  let mut seen : Std.HashMap Nat Unit := {}
  let mut out := #[]
  for u in units do
    for c in cm.getD u #[] do
      if !drawn.contains c && !seen.contains c then
        seen := seen.insert c ()
        out := out.push c
  return out.qsort (· < ·)

def Stratum.toJson (s : Stratum) : Json :=
  Json.mkObj [("template", Json.str s.template),
    ("units", Json.arr (s.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])), ("k", Lean.toJson s.k)]

def WorkStratum.toJson (s : WorkStratum) : Json :=
  Json.mkObj [("template", Json.str s.template),
    ("units", Json.arr (s.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])),
    ("work", Lean.toJson s.work), ("floor", Lean.toJson s.floor), ("k", Lean.toJson s.k)]

/-- The law object (the registration's `law`): `{law, k, strata}` for `stratified` and `work`. -/
def Law.toJson : Law → Json
  | .subset k => Json.mkObj [("law", "subset"), ("k", Lean.toJson k)]
  | .bernoulli a b => Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]
  | .stratified k strata => Json.mkObj [("law", "stratified"), ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson)))]
  | .work k strata => Json.mkObj [("law", "work"), ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson)))]

def UnitDraw.toJson (d : UnitDraw) : Json :=
  let u := Json.arr (d.units.map fun x => Lean.toJson x)
  match d.law with
  | .subset k => Json.mkObj [("law", "subset"), ("population", Lean.toJson d.population), ("k", Lean.toJson k), ("units", u)]
  | .bernoulli a b => Json.mkObj [("law", "bernoulli"), ("population", Lean.toJson d.population),
      ("p", Json.str s!"{a}/{b}"), ("units", u)]
  | .stratified k strata => Json.mkObj [("law", "stratified"), ("population", Lean.toJson d.population),
      ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson))), ("units", u)]
  | .work k strata => Json.mkObj ([("law", Json.str "work"), ("population", Lean.toJson d.population),
      ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson))), ("units", u)] ++
      if d.closure.isEmpty then [] else [("closure", Json.arr (d.closure.map fun x => Lean.toJson x))])

/-- A stratum object: exactly `{template, units, k}`, its units non-empty `[lo, hi)` ranges, ascending, disjoint and not
adjacent (`draw.strata_problem`). -/
private def stratumOfJson (j : Json) : V Stratum := do
  let .ok kvs := j.getObj? | throw "U2: a stratum is exactly {template, units, k}"
  if kvs.toArray.size != 3 || ["template", "units", "k"].any (fun key => !(j.getObjVal? key).isOk) then
    throw "U2: a stratum is exactly {template, units, k}"
  let .ok t := j.getObjValAs? String "template" | throw "U2: a stratum is exactly {template, units, k}"
  let k ← J.asNat ((j.getObjVal? "k").toOption.getD Json.null) "U2: a stratum's k"
  let .ok (.arr rs) := j.getObjVal? "units" | throw s!"U2: stratum '{t}': units are non-empty [lo, hi) ranges"
  let units ← rs.mapM fun r => do
    let .arr p := r | throw s!"U2: stratum '{t}': units are non-empty [lo, hi) ranges"
    if p.size != 2 then throw s!"U2: stratum '{t}': units are non-empty [lo, hi) ranges"
    let lo ← J.asNat p[0]! s!"U2: stratum '{t}': a range bound"
    let hi ← J.asNat p[1]! s!"U2: stratum '{t}': a range bound"
    if lo ≥ hi then throw s!"U2: stratum '{t}': units are non-empty [lo, hi) ranges"
    pure (lo, hi)
  if units.isEmpty then throw s!"U2: stratum '{t}': units are non-empty [lo, hi) ranges"
  if (List.range (units.size - 1)).any fun i => units[i + 1]!.1 ≤ units[i]!.2 then
    throw s!"U2: stratum '{t}': ranges ascending, disjoint and not adjacent"
  return { template := t, units, k }

/-- A stratified law over `population` units (`draw.strata_problem`): `1 ≤ k ≤ population`, distinct templates, each
stratum's `k` the rule's, and the strata's ranges tiling `[0, population)` exactly. -/
def strataProblem (k : Nat) (strata : Array Stratum) (population : Nat) : Option String := Id.run do
  if strata.isEmpty || k < 1 || k > population then return some "U2: strata and 1 <= k <= population"
  for i in [0:strata.size] do
    let s := strata[i]!
    if (strata.extract 0 i).any (·.template == s.template) then return some s!"U2: template '{s.template}' twice"
    if s.k != stratumK k s.size population then return some s!"U2: stratum '{s.template}': k is not the rule's"
  let mut at_ := 0
  for (lo, hi) in (strata.foldl (· ++ ·.units) #[]).qsort (fun a b => a.1 < b.1) do
    if lo != at_ then return some "U2: the strata don't tile the population"
    at_ := hi
  return if at_ == population then none else some "U2: the strata don't tile the population"

/-- A work stratum object: exactly `{template, units, work, floor, k}`, its `work` and `floor` natural numbers and the
rest a stratum's. -/
private def workStratumOfJson (j : Json) : V WorkStratum := do
  let .ok kvs := j.getObj? | throw "U2: a work stratum is exactly {template, units, work, floor, k}"
  if kvs.toArray.size != 5 || ["template", "units", "work", "floor", "k"].any (fun key => !(j.getObjVal? key).isOk)
      || !(j.getObjValAs? String "template").isOk then
    throw "U2: a work stratum is exactly {template, units, work, floor, k}"
  let get := fun (key : String) => (j.getObjVal? key).toOption.getD Json.null
  let work ← J.asNat (get "work") "U2: a stratum's work"
  let floor ← J.asNat (get "floor") "U2: a stratum's floor"
  let s ← stratumOfJson (Json.mkObj [("template", get "template"), ("units", get "units"), ("k", get "k")])
  return { toStratum := s, work, floor }

/-- A work law over `population` units (`workProblem`): `1 ≤ k ≤ population`, distinct templates, a floor of at least one
in every stratum, a population that does some work, each stratum's `k` the rule's (`workK` above its floor), and the
strata's ranges tiling `[0, population)` exactly. -/
def workProblem (k : Nat) (strata : Array WorkStratum) (population : Nat) : Option String := Id.run do
  if strata.isEmpty || k < 1 || k > population then return some "U2: strata and 1 <= k <= population"
  let total := totalWork strata
  if total == 0 then return some "U2: a work law's population does no work"
  for i in [0:strata.size] do
    let s := strata[i]!
    if (strata.extract 0 i).any (·.template == s.template) then return some s!"U2: template '{s.template}' twice"
    if s.floor < 1 then return some s!"U2: stratum '{s.template}': a floor is at least one"
    if s.k != workK k s.work s.size total s.floor then
      return some s!"U2: stratum '{s.template}': k is not the work rule's"
  let mut at_ := 0
  for (lo, hi) in (strata.foldl (· ++ ·.units) #[]).qsort (fun a b => a.1 < b.1) do
    if lo != at_ then return some "U2: the strata don't tile the population"
    at_ := hi
  return if at_ == population then none else some "U2: the strata don't tile the population"

private def ratio (s : String) : V (Nat × Nat) :=
  match s.splitOn "/" with
  | [a, b] => match a.toNat?, b.toNat? with
    | some a, some b => pure (a, b)
    | _, _ => throw "U2: unit_draw.p is not num/den"
  | _ => throw "U2: unit_draw.p is not num/den"

/-- U2: a well-formed draw. Its keys are exactly the law's, and a work draw's closure (when it has one) is non-empty,
ascending, distinct units of the population, none of them drawn; `subset` draws exactly `k ≤ population` units; `bernoulli`
has `0 ≤ p ≤ 1`, no unit when `p = 0` and every unit when `p = 1`; `stratified`'s strata are well formed and tile the
population (`strataProblem`), and exactly `k_s` of its units fall in each, and likewise `work`'s (`workProblem`); units
are ascending, distinct and in the population. -/
def UnitDraw.ofJson (j : Json) : V UnitDraw := do
  let .ok kvs := j.getObj? | throw "U2: unit_draw is not an object"
  let ctx := "unit_draw"
  let population ← J.nat j "population" ctx
  let units ← (← J.arr j "units" ctx).mapM (J.asNat · "U2: a drawn unit")
  let law ← match ← J.str j "law" ctx with
    | "subset" =>
      if kvs.toArray.size != 4 then throw "U2: unit_draw's keys are not law, population, k, units"
      pure (Law.subset (← J.nat j "k" ctx))
    | "bernoulli" =>
      if kvs.toArray.size != 4 then throw "U2: unit_draw's keys are not law, population, p, units"
      let (a, b) ← ratio (← J.str j "p" ctx)
      pure (Law.bernoulli a b)
    | "stratified" =>
      if kvs.toArray.size != 5 || !(j.getObjVal? "k").isOk || !(j.getObjVal? "strata").isOk then
        throw "U2: unit_draw's keys are not law, population, k, strata, units"
      let .ok (.arr ss) := j.getObjVal? "strata" | throw "U2: unit_draw.strata is not a list"
      pure (Law.stratified (← J.nat j "k" ctx) (← ss.mapM stratumOfJson))
    | "work" =>
      if kvs.toArray.size != (if (j.getObjVal? "closure").isOk then 6 else 5) || !(j.getObjVal? "k").isOk
          || !(j.getObjVal? "strata").isOk then
        throw "U2: unit_draw's keys are not law, population, k, strata, units and a closure draw's closure"
      let .ok (.arr ss) := j.getObjVal? "strata" | throw "U2: unit_draw.strata is not a list"
      pure (Law.work (← J.nat j "k" ctx) (← ss.mapM workStratumOfJson))
    | l => throw s!"U2: unit_draw.law {l} is neither subset, bernoulli, stratified nor work"
  if !(List.range units.size).all (fun i => units[i]! < population && (i == 0 || units[i - 1]! < units[i]!)) then
    throw "U2: the drawn units are not ascending, distinct units of the population"
  let closure ← match j.getObjVal? "closure" with
    | .error _ => pure #[]
    | .ok c => do
      let cs ← (← J.asArr c "U2: unit_draw.closure").mapM (J.asNat · "U2: a unit of the closure")
      if cs.isEmpty || !(List.range cs.size).all (fun i => cs[i]! < population && (i == 0 || cs[i - 1]! < cs[i]!)) then
        throw "U2: the closure is not non-empty, ascending, distinct units of the population"
      if cs.any units.contains then throw "U2: the closure holds a drawn unit"
      pure cs
  match law with
  | .subset k => if k > population || units.size != k then throw "U2: a subset draw of the wrong size"
  | .stratified k strata =>
    if let some why := strataProblem k strata population then throw why
    for s in strata do
      let drawn := units.foldl (fun acc u => if s.units.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0
      if drawn != s.k then throw s!"U2: stratum '{s.template}' holds {drawn} drawn units, not its k = {s.k}"
  | .work k strata =>
    if let some why := workProblem k strata population then throw why
    for s in strata do
      let drawn := units.foldl (fun acc u => if s.units.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0
      if drawn != s.k then throw s!"U2: stratum '{s.template}' holds {drawn} drawn units, not its k = {s.k}"
  | .bernoulli a b =>
    if b == 0 || a > b then throw "U2: unit_draw.p is not a probability"
    if a == 0 && units.size != 0 then throw "U2: a p = 0 draw with units"
    if a == b && units.size != population then throw "U2: a p = 1 draw without every unit"
  return { law, population, units, closure }

/-- U1–U3: the record's `unit_draw` against the statement's (the public header's, under Σ) and its `instances`. -/
def check (record statement : Option Json) (instances : Nat) : V Unit := do
  match record, statement with
  | none, none => pure ()
  | some _, none => throw "U1: the record keeps a unit draw the statement does not carry"
  | none, some _ => throw "U1: the statement carries a unit draw the record does not keep"
  | some r, some s =>
    if (← canon r) != (← canon s) then throw "U1: the record's unit draw is not the statement's"
    let d ← UnitDraw.ofJson s
    if d.proved.size != instances then throw "U3: the statement's instances are not the drawn units and their closure"

/-- A sampler run on a byte source, `get need` returning `need` fresh bytes: a stream of `need` bytes, extended by `need`
more and the same sampler rerun on it whenever it runs out, so the draw stays exact. `none` only if the loop ends, which
it doesn't. -/
def onWith {m : Type → Type} [Monad m] (get : Nat → m ByteArray) (need : Nat) (run : ByteArray → Option (Array Nat)) :
    m (Option (Array Nat)) := do
  let mut src ← get need
  repeat
    if let some units := run src then return some units
    src := src ++ (← get need)
  return none

/-- The stratified draw on a byte source, stratum `i`'s stream from `get i`: `stratified`, each stratum's `subset` run by
`onWith`. -/
def stratifiedWith {m : Type → Type} [Monad m] (get : Nat → Nat → m ByteArray) (strata : Array Stratum) :
    m (Option (Array Nat)) := do
  let mut out := #[]
  for i in [0:strata.size] do
    let s := strata[i]!
    let some local_ ← onWith (get i) s.need (subset s.k s.size) | return none
    out := out ++ local_.map s.unit
  return some (out.qsort (· < ·))

/-- The draw on a byte source, stream `i`'s bytes from `get i`: one stream per stratum for a stratified or work draw, stream
`0` for a subset or Bernoulli draw. -/
def drawWith {m : Type → Type} [Monad m] (get : Nat → Nat → m ByteArray) (law : Law) (population : Nat) :
    m (Option UnitDraw) := do
  let units ← match law with
    | .subset k => onWith (get 0) (2 * k * width population + 64) (subset k population)
    | .bernoulli a b => onWith (get 0) (2 * population * width b + 64) (bernoulli a b population)
    | .stratified _ strata => stratifiedWith get strata
    | .work _ strata => stratifiedWith get (strata.map (·.toStratum))
  return units.map fun units => { law, population, units }

/-- The draw from the verifier's operating-system randomness: `drawWith` with every stream's bytes from
`IO.getRandomBytes`. -/
def drawOS (law : Law) (population : Nat) : IO UnitDraw := do
  let some d ← drawWith (fun _ need => IO.getRandomBytes need.toUSize) law population
    | throw (IO.userError "unreachable")
  return d

/-- The draw from given bytes (a test): `subset` and `bernoulli` on the one stream; `stratified` and `work` on
consecutive segments, stratum `s`'s the next `2 k_s · width(n_s) + 64` bytes. `none` when a stream runs out. -/
def drawOn (law : Law) (population : Nat) (src : ByteArray) : Option UnitDraw := do
  let units ← match law with
    | .subset k => subset k population src
    | .bernoulli a b => bernoulli a b population src
    | .stratified _ strata =>
      let (segs, _) := strata.foldl (fun (acc : Array ByteArray × Nat) s =>
        (acc.1.push (src.extract acc.2 (acc.2 + s.need)), acc.2 + s.need)) (#[], 0)
      stratified strata segs
    | .work _ strata =>
      let strata := strata.map (·.toStratum)
      let (segs, _) := strata.foldl (fun (acc : Array ByteArray × Nat) s =>
        (acc.1.push (src.extract acc.2 (acc.2 + s.need)), acc.2 + s.need)) (#[], 0)
      stratified strata segs
  return { law, population, units }

/-- The units every draw under a law has, when that count is fixed: `k` for a subset law and `Σ_s k_s` for a stratified
one (`UnitDraw.ofJson` holds every decoded draw to them); none for a Bernoulli law, whose count is random, and a work law,
whose closure varies from draw to draw. -/
def Law.fixedCount : Law → Option Nat
  | .subset k => some k
  | .stratified _ strata => some (strata.foldl (fun acc s => acc + s.k) 0)
  | _ => none

/-- The refusal of a draw that several tables cannot split (`Stmt.setupTables`, `tablesWhy`). -/
def TABLES_LAW : String :=
  "tables: several tables need a subset or stratified draw, whose unit count is fixed (a Bernoulli draw's count is random, so the tables could not split it evenly, and a work draw's closure is not split)"

/-- Why one draw under `law` cannot be split into `count` equal tables, as `Stmt.setupTables` splits a session's draw, else
none: one table takes any law; several take a law of fixed count (`Law.fixedCount`) that is positive and a multiple of
`count`. A law's count is fixed before any coin, so `flock-verify draw --tables J` refuses here before reading one. -/
def tablesWhy (law : Law) (count : Nat) : Option String :=
  if count ≤ 1 then none
  else match law.fixedCount with
    | none => some TABLES_LAW
    | some n => if n == 0 || n % count != 0 then some s!"tables: {n} units do not split into {count} equal tables" else none

end Flock.Draw
