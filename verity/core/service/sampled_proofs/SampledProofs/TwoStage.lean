/-
The two-stage law's samplers (`two_stage.select_replay_units` and `two_stage.select_proof_units`) on the verifier's own
coins, read in order as `OwnRandomness` reads them (`Draw.take`), with the number of bytes read.

* Stage 1 draws the replay units (RUs) in the given order: an RU of a class with a probability `num/den` on its own,
  `uniform den < num` with the probability in lowest terms (0 and 1 read nothing); the RUs of a class with a one-stage
  law together, when the class's first RU comes up, by `draw_first` over their positions in index order
  (`Flock.Draw.drawWith` under `Draw.flockLaw`, each stream extended and rerun when it runs out).
* Stage 2 checks `k` proof units of each drawn RU (all of them when `k` is null): a uniform `k`-subset of its positions
  (`subset`'s sparse partial Fisher–Yates) when it holds more, and otherwise every position, reading nothing.

An error where the Python raises: a law that isn't one (`Stage`), an RU index twice, a class the law has no stage for, a
one-stage law the class's RUs don't admit, or coins that run out.
-/
import SampledProofs.Draw

namespace SampledProofs.TwoStage

open Lean (Json)
open Flock (V)

private def digits (s : String) : Bool := !s.isEmpty && s.all Char.isDigit

/-- A class's first stage: a probability `num/den` in lowest terms, or a one-stage law object. -/
inductive First where
  | rate (num den : Nat)
  | law (law : Json)

/-- `two_stage.Stage`: the first stage and the proof units checked of a drawn RU (`none`: every one). -/
structure Stage where
  first : First
  k : Option Nat

/-- `ReplayUnit`: its index in the committed RU order, its class and its proof-unit count. -/
structure RU where
  index : Int
  cls : String
  nv : Nat

/-- `two_stage._law_problem`: a one-stage law object's kind, exactly its keys, a Bernoulli law's `NUM/DEN`, and otherwise
a natural `k` and, for `stratified` and `work`, a list of strata. -/
def lawProblem (law : Json) : Bool :=
  let kind := match getD law "law" with | .str s => s | _ => ""
  let keys := match kind with
    | "subset" => some #["law", "k"]
    | "bernoulli" => some #["law", "p"]
    | "stratified" | "work" => some #["law", "k", "strata"]
    | _ => none
  match keys with
  | none => true
  | some ks =>
    keys? law != some (sortStrings ks) ||
    if kind == "bernoulli" then
      match getD law "p" with
      | .str p => (Draw.probability? p).isNone
      | _ => true
    else (nat? (getD law "k")).isNone || (kind != "subset" && !(getD law "strata" matches .arr _))

/-- A stage as `Stage.to_json` writes it, exactly its keys: `{"p": NUM/DEN or NUM, "k"}` with `0 ≤ p ≤ 1`, or
`{"law": a one-stage law object, "k"}`, `k` null or a natural number, and null unless the law is `bernoulli`. -/
def Stage.ofJson (j : Json) : V Stage := do
  let k ← match getD j "k" with
    | .null => pure none
    | x => match nat? x with
      | some k => pure (some k)
      | none => throw "a stage needs k >= 0"
  if keys? j == some #["k", "p"] then
    let .str p := getD j "p" | throw "a stage's p is NUM/DEN"
    let (num, den) ← match p.splitOn "/" with
      | [a] => if digits a then pure (a.toNat!, 1) else throw "a stage's p is NUM/DEN"
      | [a, b] => if digits a && digits b && 0 < b.toNat! then pure (a.toNat!, b.toNat!) else throw "a stage's p is NUM/DEN"
      | _ => throw "a stage's p is NUM/DEN"
    if den < num then throw "a stage needs 0 <= p <= 1"
    let g := Nat.gcd num den
    return { first := .rate (num / g) (den / g), k }
  if keys? j == some #["k", "law"] then
    let law := getD j "law"
    if !(law matches .obj _) || lawProblem law then throw "a stage's law is a one-stage law object"
    if !jeq (getD law "law") "bernoulli" && k.isSome then
      throw "a first stage other than bernoulli checks every proof unit of a drawn RU (k = None)"
    return { first := .law law, k }
  throw "a stage is {p, k} or {law, k}"

/-- `TwoStageLaw.to_json`: `{class: stage}`. -/
def lawOfJson (j : Json) : V (Array (String × Stage)) := do
  let .obj _ := j | throw "a two-stage law is an object of stages"
  (entries j).mapM fun (c, s) => do pure (c, ← Stage.ofJson s)

/-- `[[index, class, n_v], ...]`, the index an integer and `n_v` a natural number. -/
def rusOfJson (j : Json) : V (Array RU) := do
  let .arr us := j | throw "replay units are a list of [index, class, n_v]"
  us.mapM fun u => match u with
    | .arr #[i, .str cls, nv] => match int? i, nat? nv with
      | some index, some nv => pure { index, cls, nv }
      | _, _ => throw "a replay unit is [index, class, n_v]"
    | _ => throw "a replay unit is [index, class, n_v]"

abbrev Coins := StateT Nat (Except String)

/-- `OwnRandomness.uniform`: the first `width n`-byte big-endian word below the largest multiple of `n`, mod `n`. -/
def uniform (src : ByteArray) (n : Nat) : Coins Nat := do
  if n == 0 then throw "uniform(n) needs n > 0"
  let w := Flock.Draw.width n
  let space := 2 ^ (8 * w)
  for _ in [0:src.size / w + 1] do
    let x := Flock.Draw.beNat (← Draw.take src w) 0 w
    if x < space - space % n then return x % n
  throw "the randomness source ran out"

/-- `OwnRandomness.subset`: a uniform `k`-subset of `[0, n)`, sorted. -/
def subset (src : ByteArray) (k n : Nat) : Coins (Array Nat) := do
  let mut sw : Std.HashMap Nat Nat := {}
  for i in [0:k] do
    let j := i + (← uniform src (n - i))
    let (vi, vj) := (sw.getD i i, sw.getD j j)
    sw := (sw.insert i vj).insert j vi
  return ((Array.range k).map fun i => sw.getD i i).qsort (· < ·)

private def stage (law : Array (String × Stage)) (cls : String) : Except String Stage :=
  match law.find? (·.1 == cls) with
  | some (_, s) => pure s
  | none => throw s!"the law has no stage for replay-unit class {pyRepr cls}"

private def distinct (units : Array RU) : Except String Unit :=
  if (units.map (·.index)).toList.eraseDups.length != units.size then
    throw "replay-unit indices must be distinct: each is a draw's address"
  else pure ()

/-- `select_replay_units`: the drawn RUs' indices in the given order, and the bytes of `src` read. -/
def selectReplayUnits (src : ByteArray) (law : Array (String × Stage)) (units : Array RU) :
    Except String (Array Int × Nat) := do
  distinct units
  let go : Coins (Array Int) := do
    let mut drawn : Std.HashMap String (Array Int) := {}
    let mut out := #[]
    for u in units do
      match (← stage law u.cls).first with
      | .rate num den =>
        let hit ← if num == 0 then pure false else if num == den then pure true else (· < num) <$> uniform src den
        if hit then out := out.push u.index
      | .law l =>
        if !drawn.contains u.cls then
          let mine := (units.filter (·.cls == u.cls)).qsort (·.index < ·.index)
          let some d ← Flock.Draw.drawWith (fun _ need => Draw.take src need) (← Draw.flockLaw l mine.size) mine.size
            | throw "the draw did not end"
          drawn := drawn.insert u.cls (d.units.filterMap fun j => mine[j]?.map (·.index))
        if (drawn.getD u.cls #[]).contains u.index then out := out.push u.index
    return out
  go.run 0

/-- `select_proof_units`: for each selected RU in order, its index and the sorted positions checked in it, and the bytes
of `src` read. -/
def selectProofUnits (src : ByteArray) (law : Array (String × Stage)) (selected : Array RU) :
    Except String (Array (Int × Array Nat) × Nat) := do
  distinct selected
  let go : Coins (Array (Int × Array Nat)) := selected.mapM fun u => do
    let k := ((← stage law u.cls).k.map (min · u.nv)).getD u.nv
    return (u.index, ← if k < u.nv then subset src k u.nv else pure (Array.range u.nv))
  go.run 0

end SampledProofs.TwoStage
