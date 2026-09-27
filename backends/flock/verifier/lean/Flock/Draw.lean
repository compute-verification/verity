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

inductive Law where
  | subset (k : Nat)
  | bernoulli (num den : Nat)
  deriving BEq, Inhabited

/-- A draw: its law, the population `[0, population)`, and the drawn units, ascending. -/
structure UnitDraw where
  law : Law
  population : Nat
  units : Array Nat
  deriving Inhabited

def UnitDraw.toJson (d : UnitDraw) : Json :=
  let u := Json.arr (d.units.map fun x => Lean.toJson x)
  match d.law with
  | .subset k => Json.mkObj [("law", "subset"), ("population", Lean.toJson d.population), ("k", Lean.toJson k), ("units", u)]
  | .bernoulli a b => Json.mkObj [("law", "bernoulli"), ("population", Lean.toJson d.population),
      ("p", Json.str s!"{a}/{b}"), ("units", u)]

private def ratio (s : String) : V (Nat × Nat) :=
  match s.splitOn "/" with
  | [a, b] => match a.toNat?, b.toNat? with
    | some a, some b => pure (a, b)
    | _, _ => throw "U2: unit_draw.p is not num/den"
  | _ => throw "U2: unit_draw.p is not num/den"

/-- U2: a well-formed draw. Its keys are exactly the law's; `subset` draws exactly `k ≤ population` units; `bernoulli`
has `0 ≤ p ≤ 1`, no unit when `p = 0` and every unit when `p = 1`; units are ascending, distinct and in the population. -/
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
    | l => throw s!"U2: unit_draw.law {l} is neither subset nor bernoulli"
  if !(List.range units.size).all (fun i => units[i]! < population && (i == 0 || units[i - 1]! < units[i]!)) then
    throw "U2: the drawn units are not ascending, distinct units of the population"
  match law with
  | .subset k => if k > population || units.size != k then throw "U2: a subset draw of the wrong size"
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

/-- The draw from the verifier's operating-system randomness. A stream that runs out is extended and the same sampler
rerun on it, so the draw stays exact. -/
def drawOS (law : Law) (population : Nat) : IO UnitDraw := do
  let need := match law with
    | .subset k => 2 * k * width population + 64
    | .bernoulli _ den => 2 * population * width den + 64
  let mut src ← IO.getRandomBytes need.toUSize
  repeat
    let r := match law with
      | .subset k => subset k population src
      | .bernoulli a b => bernoulli a b population src
    if let some units := r then return { law, population, units }
    src := src ++ (← IO.getRandomBytes need.toUSize)
  throw (IO.userError "unreachable")

end Flock.Draw
