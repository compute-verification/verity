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
import SampledProofs.Record

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

/-- The profile's only level: the replay units. -/
def REPLAY : String := "replay"

/-- `Stage.p`: the probability each RU is drawn under a Bernoulli first stage (a probability, or a `bernoulli` law),
in lowest terms; `none` for a subset, stratified or work law. -/
def Stage.p (s : Stage) : Option Core.Ratio :=
  match s.first with
  | .rate num den => Core.Ratio.mk' num den
  | .law l => match getD l "law", getD l "p" with
    | .str "bernoulli", .str p => (Draw.probability? p).bind fun (a, b) => Core.Ratio.mk' a b
    | _, _ => none

private def kindOf (s : Stage) : String :=
  match s.first with
  | .law l => match getD l "law" with | .str k => k | _ => ""
  | .rate _ _ => "bernoulli"

/-- `Stage.checked`: the proof units checked of a drawn RU of `nv`. -/
def Stage.checked (s : Stage) (nv : Nat) : Nat := (s.k.map (min · nv)).getD nv

/-- `TwoStageLaw.profile`'s exhaustively checked fields: `programs`, `prescribed`, `linked`, `covered`, and `drawn`
({`replay`: RUs drawn}), as `IntegrityProfile` takes them. -/
structure Facts where
  programs : Array String := #[]
  prescribed : Array String := #[]
  linked : Bool := false
  covered : Array String := #[]
  drawn : Array (String × Int) := #[]

/-- `TwoStageLaw.profile`: the profile an accepted transcript leaves over the RUs of class `cls`, one level `replay` of
the class's `n_r` RUs, the only level committed before the first draw.

* Under a Bernoulli first stage a wrong RU of `n_v` proof units is caught only when it is drawn and a wrong proof unit is
  among the `k` checked, so it escapes with probability at most `1 - p k / n_v`, more often the larger it is: the level
  is drawn `Bernoulli(p k / n_v)` at the class's largest RU (`k = n_v` when every proof unit is checked).
* Under a subset, stratified or work first stage every proof unit of a drawn RU is checked, so the level is drawn as the
  one-stage law draws it: `Subset(k)`, or its strata (a work law's at its sizes, `Record.profileDraw`).

`units` are the class's RUs (others ignored, all distinct): `n_r` is their count and `n_v` their largest proof-unit
count, a caller's `n_r` equal to it and its `n_v` no smaller. Without them `n_r` is needed, and `n_v` under a Bernoulli
first stage. Then `IntegrityProfile`'s own checks: `drawn` keyed by `replay` and at least 0, and `0 < δ < 1`. An error
where Python raises. -/
def profile (law : Array (String × Stage)) (cls : String) (units : Option (Array RU)) (nr nv : Option Nat)
    (delta : Core.Ratio) (facts : Facts) : Except String Core.Integrity := do
  let s ← stage law cls
  let (nr, nv) ← match units with
    | none => pure (nr, nv)
    | some us => do
      distinct us
      let mine := us.filter (·.cls == cls)
      if mine.isEmpty then throw s!"no replay unit of class {pyRepr cls}"
      if let some n := nr then
        if n != mine.size then throw s!"n_r = {n}, but the class has {mine.size} RUs"
      let largest := mine.foldl (fun m u => max m u.nv) 0
      if let some n := nv then
        if n < largest then
          throw (s!"n_v = {n} is smaller than the class's largest RU ({largest} proof units): " ++
            "the profile would overstate detection")
      pure (some mine.size, some (max largest (nv.getD 0)))
  let draw ← match s.p, s.first with
    | none, .law l => do
      let some n := nr | throw "pass the class's RUs, or n_r"
      if jeq (getD l "law") "subset" then
        let k := (nat? (getD l "k")).getD 0
        if k > n then throw s!"a subset first stage of {k} exceeds the class's {n} RUs"
        pure (Core.Draw.subset k)
      else
        if let some why := Draw.strataMessage l n then
          throw s!"the class's {kindOf s} first stage over its {n} RUs: {why}"
        let some d := Record.profileDraw l | throw "the first stage's law has no profile"
        pure d
    | some p, _ => do
      let (some _, some v) := (nr, nv) | throw "pass the class's RUs, or n_r and n_v (its largest RU's proof-unit count)"
      if v < 1 then throw "an RU holds at least one proof unit"
      let some rate := Core.Ratio.mk' (p.num * s.checked v) (p.den * v) | throw "an RU holds at least one proof unit"
      pure (Core.Draw.bernoulli rate)
    | none, .rate _ _ => throw "a stage's p is NUM/DEN"
  let some n := nr | throw "pass the class's RUs, or n_r"
  if facts.drawn.any (·.1 != REPLAY) then throw "draws and drawn counts are keyed by level names"
  if facts.drawn.any (·.2 < 0) then throw "drawn counts are >= 0"
  if let .stratified ss := draw then
    if ss.any fun t => t.name.isEmpty || t.size < t.k then
      throw "a stratum needs a name, size >= 0 and 0 <= k <= size"
    let held := ss.foldl (· + ·.size) 0
    if held != n then throw s!"the strata hold {held} units, the checked level {n}"
  if delta.num == 0 || delta.den ≤ delta.num then throw "a profile needs 0 < delta < 1"
  return { programs := facts.programs, prescribed := facts.prescribed.toList.eraseDups.toArray, linked := facts.linked,
           covered := facts.covered.toList.eraseDups.toArray,
           levels := #[{ name := REPLAY, query := none, size := n, draw := some draw }], delta,
           drawn := facts.drawn.map fun (k, v) => (k, v.toNat) }

/-- `check_probability`: the marginal probability that one given proof unit of an RU of class `cls` holding `n_v` proof
units is checked, under a Bernoulli first stage, `p · checked(n_v) / n_v` (0 for an empty RU); an error for a subset,
stratified or work first stage, whose marginal depends on the population. -/
def checkProbability (law : Array (String × Stage)) (cls : String) (nv : Nat) : Except String Core.Ratio := do
  let s ← stage law cls
  let some p := s.p
    | throw s!"class {pyRepr cls}: a {kindOf s} first stage's marginal depends on the population"
  if nv == 0 then return Core.Ratio.ofNat 0
  let some r := Core.Ratio.mk' (p.num * s.checked nv) (p.den * nv) | throw "unreachable: n_v > 0"
  return r

end SampledProofs.TwoStage
