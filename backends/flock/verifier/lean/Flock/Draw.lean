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

inductive Law where
  | subset (k : Nat)
  | bernoulli (num den : Nat)
  /-- one uniform `k_s`-subset per stratum, independently (the default law; `k` is the agreement's) -/
  | stratified (k : Nat) (strata : Array Stratum)
  deriving BEq, Inhabited

/-- `stratified:K` expanded over the strata (`draw.derive`): each stratum with `stratumK`, for `1 ≤ K ≤ population`. -/
def derive (k : Nat) (strata : Array (String × Array (Nat × Nat))) : Except String Law := do
  let n := strata.foldl (fun acc (_, rs) => acc + rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) 0
  if k < 1 || k > n then throw s!"a stratified law needs 1 <= k <= {n}, not {k}"
  return .stratified k (strata.map fun (t, rs) =>
    { template := t, units := rs, k := stratumK k (rs.foldl (fun a (lo, hi) => a + (hi - lo)) 0) n })

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

/-- A draw: its law, the population `[0, population)`, and the drawn units, ascending. -/
structure UnitDraw where
  law : Law
  population : Nat
  units : Array Nat
  deriving Inhabited

def Stratum.toJson (s : Stratum) : Json :=
  Json.mkObj [("template", Json.str s.template),
    ("units", Json.arr (s.units.map fun (lo, hi) => Json.arr #[Lean.toJson lo, Lean.toJson hi])), ("k", Lean.toJson s.k)]

/-- The law object (the registration's `law`): `{law, k, strata}` for `stratified`. -/
def Law.toJson : Law → Json
  | .subset k => Json.mkObj [("law", "subset"), ("k", Lean.toJson k)]
  | .bernoulli a b => Json.mkObj [("law", "bernoulli"), ("p", Json.str s!"{a}/{b}")]
  | .stratified k strata => Json.mkObj [("law", "stratified"), ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson)))]

def UnitDraw.toJson (d : UnitDraw) : Json :=
  let u := Json.arr (d.units.map fun x => Lean.toJson x)
  match d.law with
  | .subset k => Json.mkObj [("law", "subset"), ("population", Lean.toJson d.population), ("k", Lean.toJson k), ("units", u)]
  | .bernoulli a b => Json.mkObj [("law", "bernoulli"), ("population", Lean.toJson d.population),
      ("p", Json.str s!"{a}/{b}"), ("units", u)]
  | .stratified k strata => Json.mkObj [("law", "stratified"), ("population", Lean.toJson d.population),
      ("k", Lean.toJson k), ("strata", Json.arr (strata.map (·.toJson))), ("units", u)]

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

private def ratio (s : String) : V (Nat × Nat) :=
  match s.splitOn "/" with
  | [a, b] => match a.toNat?, b.toNat? with
    | some a, some b => pure (a, b)
    | _, _ => throw "U2: unit_draw.p is not num/den"
  | _ => throw "U2: unit_draw.p is not num/den"

/-- U2: a well-formed draw. Its keys are exactly the law's; `subset` draws exactly `k ≤ population` units; `bernoulli`
has `0 ≤ p ≤ 1`, no unit when `p = 0` and every unit when `p = 1`; `stratified`'s strata are well formed and tile the
population (`strataProblem`), and exactly `k_s` of its units fall in each; units are ascending, distinct and in the
population. -/
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
    | l => throw s!"U2: unit_draw.law {l} is neither subset, bernoulli nor stratified"
  if !(List.range units.size).all (fun i => units[i]! < population && (i == 0 || units[i - 1]! < units[i]!)) then
    throw "U2: the drawn units are not ascending, distinct units of the population"
  match law with
  | .subset k => if k > population || units.size != k then throw "U2: a subset draw of the wrong size"
  | .stratified k strata =>
    if let some why := strataProblem k strata population then throw why
    for s in strata do
      let drawn := units.foldl (fun acc u => if s.units.any (fun (lo, hi) => lo ≤ u && u < hi) then acc + 1 else acc) 0
      if drawn != s.k then throw s!"U2: stratum '{s.template}' holds {drawn} drawn units, not its k = {s.k}"
  | .bernoulli a b =>
    if b == 0 || a > b then throw "U2: unit_draw.p is not a probability"
    if a == 0 && units.size != 0 then throw "U2: a p = 0 draw with units"
    if a == b && units.size != population then throw "U2: a p = 1 draw without every unit"
  return { law, population, units }

/-- U1–U3: the record's `unit_draw` against the statement's (the public header's, under Σ) and its `instances`. -/
def check (record statement : Option Json) (instances : Nat) : V Unit := do
  match record, statement with
  | none, none => pure ()
  | some _, none => throw "U1: the record keeps a unit draw the statement does not carry"
  | none, some _ => throw "U1: the statement carries a unit draw the record does not keep"
  | some r, some s =>
    if (← canon r) != (← canon s) then throw "U1: the record's unit draw is not the statement's"
    let d ← UnitDraw.ofJson s
    if d.units.size != instances then throw "U3: the statement's instances are not the drawn units"

/-- A sampler run on the verifier's operating-system randomness: a stream of `need` bytes, extended by `need` more and the
same sampler rerun on it whenever it runs out, so the draw stays exact. -/
def onOS (need : Nat) (run : ByteArray → Option (Array Nat)) : IO (Array Nat) := do
  let mut src ← IO.getRandomBytes need.toUSize
  repeat
    if let some units := run src then return units
    src := src ++ (← IO.getRandomBytes need.toUSize)
  throw (IO.userError "unreachable")

/-- The draw from the verifier's operating-system randomness; a stratified draw takes one stream per stratum. -/
def drawOS (law : Law) (population : Nat) : IO UnitDraw := do
  let units ← match law with
    | .subset k => onOS (2 * k * width population + 64) (subset k population)
    | .bernoulli a b => onOS (2 * population * width b + 64) (bernoulli a b population)
    | .stratified _ strata => do
      let mut out := #[]
      for s in strata do
        out := out ++ (← onOS s.need (subset s.k s.size)).map s.unit
      pure (out.qsort (· < ·))
  return { law, population, units }

/-- The draw from given bytes (a test): `subset` and `bernoulli` on the one stream; `stratified` on consecutive segments,
stratum `s`'s the next `2 k_s · width(n_s) + 64` bytes. `none` when a stream runs out. -/
def drawOn (law : Law) (population : Nat) (src : ByteArray) : Option UnitDraw := do
  let units ← match law with
    | .subset k => subset k population src
    | .bernoulli a b => bernoulli a b population src
    | .stratified _ strata =>
      let (segs, _) := strata.foldl (fun (acc : Array ByteArray × Nat) s =>
        (acc.1.push (src.extract acc.2 (acc.2 + s.need)), acc.2 + s.need)) (#[], 0)
      stratified strata segs
  return { law, population, units }

end Flock.Draw
