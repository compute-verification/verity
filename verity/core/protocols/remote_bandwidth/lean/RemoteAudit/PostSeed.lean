import Lean.Data.Json
import Flock.Canon
import Flock.Program
import RemoteAudit.Property.Judge

/-!
# The post-seed verdict (Cut 1, PROTOCOL.md §8)

The remote audit's verdict on the windows after the seed: property/v1 (`Property.judge` of `Property.analyze`) on the
writers' Program W at its step ends, under the windows and lifetime cap the session's record states, read in census steps
(`reading`, binding 4), and bound to the session by equalities between things the verifier holds, each refused by name:

1. ``program``: W's SHA-512 (`Flock.Program.Prog.sha512`, `codec.program_digest`) is `audited`, the one-stage audit's;
2. ``values``: each memory-map unit's ``val{k}`` row is the one-stage value commitments' leaf at ``pos[k]``
   (`mapValueLeaves` of its public row);
3. ``mapped``: the session's appends are the map units' payloads (the memory-map unit's verdict, given);

and ``property``: P(c, R_in, L) holds at B operand bits. The census is W's by construction: the verdict computes what it
reads of W at the ends it is given.

`reading` is the conversion (memory-accounting's MEM-38): pass i (in order) names step s_i, with s_0 = −1, and covers the
census steps (s_{i−1}, s_i]; the steps after the last pass belong to the closing window's; an unpaced session whose last
pass is before the census's last step is refused ("steps after the last pass are uncovered"). L in steps is the most
census steps any L_w + 1 consecutive windows (window w and the L_w after it) cover.

`ml.circuits.post_seed` mirrors `verdict` in Python, held to it by `lean/tests/test_post_seed_difftest.py`.
-/

namespace RemoteAudit.PostSeed

open Lean (Json toJson)
open Flock
open RemoteAudit.Property

/-- A window of the session's record: the steps its passes name, its receipts and its span. -/
structure RecordWindow where
  steps : Array Int
  receipts : Nat
  spanNs : Nat
  deriving Inhabited

/-- `audit.property_reading`: the record's windows as census-step windows, and its lifetime cap in steps. -/
def reading (wins : Array RecordWindow) (censusSteps : Nat) (paced : Bool) (blockBytes lifetime : Nat) :
    Except String (Array Window × Int) := do
  if censusSteps < 1 then throw "a census has one or more steps"
  if wins.isEmpty || wins.any (·.steps.isEmpty) then throw "a window's passes name no steps: read a session's own record"
  let steps : Array Int := wins.foldl (fun acc w => acc ++ w.steps) #[]
  if (Array.range (steps.size - 1)).any (fun i => steps[i]! ≥ steps[i + 1]!) || steps.back! ≥ censusSteps then
    throw s!"the passes' steps strictly increase below the census's {censusSteps}"
  if steps.back! < (censusSteps : Int) - 1 && !paced then throw "steps after the last pass are uncovered"
  let mut out : Array Window := #[]
  let mut start : Int := 0
  for i in [0:wins.size] do
    let w := wins[i]!
    let stop : Int := if i == wins.size - 1 then censusSteps else w.steps.back! + 1
    out := out.push { start, stop, reads := w.receipts * blockBytes, seconds := Q.make (w.spanNs : Int) (10 ^ 9) }
    start := stop
  let held := (Array.range out.size).foldl (fun acc i =>
    let held := out[min (i + lifetime) (out.size - 1)]!.stop - out[i]!.start
    match acc with
    | some m => some (max m held)
    | none => some held) none
  return (out, held.getD 0)

/-- `requirement_units.map_value_leaves`'s value rows: ``val{k}`` of each present k at leaf ``pos[k]``. -/
def mapValueLeaves (C K V : Nat) (pub : Json) : Except String (Array (String × Nat)) := do
  let count ← J.nat pub "count" "map_public"
  let present ← J.nat pub "present" "map_public"
  let pos ← (← J.arr pub "pos" "map_public").mapM (J.asNat · "map_public.pos")
  let mut out := #[]
  let bad := count > C || present >>> K != 0
  for k in [0:K] do
    if (present >>> k) % 2 == 1 then
      let some p := pos[k]? | throw s!"a public row's pos names K = {K} values"
      if p ≥ V then throw s!"a public row of at most C = {C} blocks and K = {K} values below V = {V}"
      out := out.push (s!"val{k}", p)
  if bad then throw s!"a public row of at most C = {C} blocks and K = {K} values below V = {V}"
  return out

private def windowJ (w : Window) : Json := .arr #[toJson w.start, toJson w.stop, toJson w.reads, .str w.seconds.str]

/-- The definitions of a descriptor that declare ports. -/
def ported (desc : Json) : Array String :=
  match desc.getObjVal? "definitions" with
  | .ok (.obj kvs) => (kvs.toArray.filter fun (_, d) => J.has d "ports").map (·.1)
  | _ => #[]

def decode (desc : Json) : Except String Program.Prog := Program.decode desc.compress.toUTF8

def classesOf (j : Json) : Except String (Array Q) := do
  let .obj kvs := j | throw "c: an object"
  if kvs.size != 4 then throw s!"c gives units per certified byte for exactly the four budgeted classes"
  BUDGETED.mapM fun cls => do Q.ofJson (← J.get j cls.name "c") s!"c.{cls.name}"

def endsOf (j : Json) : Except String (Array Nat) := do (← J.arr j "ends" "input").mapM (J.asNat · "ends")

def windowsOf (j : Json) : Except String (Array RecordWindow) := do
  (← J.arr j "windows" "input").mapM fun w => do
    pure { steps := ← (← J.arr w "steps" "window").mapM (J.asInt · "window.steps"),
           receipts := ← J.nat w "receipts" "window", spanNs := ← J.nat w "span_ns" "window" }

private def bytesOf (j : Json) (ctx : String) : Except String ByteArray := do
  match unhex (← J.asStr j ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}: {e}"

/-- The verdict: `{"refused", "program", "census_steps", "L", "windows", "verdict"}`. -/
def verdict (j : Json) : Except String Json := do
  let desc ← J.get j "program" "input"
  let P ← decode desc
  let ends ← endsOf j
  let (wins, L) ← reading (← windowsOf j) ends.size (← J.bool j "paced" "input") (← J.nat j "block_bytes" "input")
    (← J.nat j "lifetime" "input")
  if L < 0 then throw "a verdict's L is a whole number of census steps, the record's cap read in them"
  let mp ← J.get j "map" "input"
  let (C, K, V) := (← J.nat mp "C" "map", ← J.nat mp "K" "map", ← J.nat mp "V" "map")
  let pubs ← J.arr j "map_public" "input"
  let rows ← J.arr j "map_rows" "input"
  if rows.size != pubs.size then throw s!"{rows.size} memory-map units' rows for {pubs.size} public rows"
  let values ← (← J.arr j "values" "input").mapM (bytesOf · "values")
  let a ← analyze P ends (ported desc)
  let v ← judge a (← classesOf (← J.get j "c" "input")) wins (← Q.ofJson (← J.get j "R_in" "input") "R_in") (some L.toNat)
    (← J.nat j "B" "input")
  let mut badValue := false
  for (pub, row) in pubs.zip rows do
    for (name, at_) in ← mapValueLeaves C K V pub do
      let got ← match row.getObjVal? name with
        | .ok h => some <$> bytesOf h s!"map_rows.{name}"
        | .error _ => pure none
      if at_ ≥ values.size || got != some values[at_]! then badValue := true
  let digest := hexOf P.sha512
  let ok := match v.getObjVal? "ok" with
    | .ok (.bool b) => b
    | _ => false
  let refused := [("program", digest != (← J.str j "audited" "input")), ("values", badValue),
    ("mapped", !(← J.bool j "mapped" "input")), ("property", !ok)].filterMap fun (n, bad) => if bad then some n else none
  return Json.mkObj [("refused", .arr (refused.toArray.map .str)), ("program", .str digest),
    ("census_steps", toJson ends.size), ("L", toJson L), ("windows", .arr (wins.map windowJ)), ("verdict", v)]

/-- property/v1 alone: `{"program", "ends", "c", "windows": [[start, stop, reads, seconds]], "R_in", "L", "B"}`. -/
def propertyCheck (j : Json) : Except String Json := do
  let desc ← J.get j "program" "input"
  let P ← decode desc
  let a ← analyze P (← endsOf j) (ported desc)
  let wins ← (← J.arr j "windows" "input").mapM fun w => do
    let xs ← J.asArr w "window"
    if xs.size != 4 then throw "a window is [start, stop, reads, seconds]"
    pure ({ start := ← J.asInt xs[0]! "window", stop := ← J.asInt xs[1]! "window", reads := ← J.asNat xs[2]! "window",
            seconds := ← Q.ofJson xs[3]! "window" } : Window)
  let L ← match ← J.get j "L" "input" with
    | .null => pure none
    | l => some <$> J.asNat l "L"
  judge a (← classesOf (← J.get j "c" "input")) wins (← Q.ofJson (← J.get j "R_in" "input") "R_in") L (← J.nat j "B" "input")

/-- `reading` alone: `{"windows", "census_steps", "paced", "block_bytes", "lifetime"}`. -/
def readingFn (j : Json) : Except String Json := do
  let (wins, L) ← reading (← windowsOf j) (← J.nat j "census_steps" "input") (← J.bool j "paced" "input")
    (← J.nat j "block_bytes" "input") (← J.nat j "lifetime" "input")
  return Json.mkObj [("windows", .arr (wins.map windowJ)), ("L", toJson L)]

/-- The table's classes of ids: `{"digest", "classes": [class | null], "gathers": [[index, bits, kind, lifted] | null]}`. -/
def classOfFn (j : Json) : Except String Json := do
  let ids ← (← J.arr j "ids" "input").mapM (J.asStr · "ids")
  return Json.mkObj [("digest", .str digest),
    ("classes", .arr (ids.map fun f => match classOf f with | some c => .str c.name | none => .null)),
    ("gathers", .arr (ids.map fun f => match gather? f with
      | some (i, b, k, l) => .arr #[toJson i, toJson b, .str k.name, .bool l]
      | none => .null))]

def fns : List (String × (Json → Except String Json)) :=
  [("post_seed", verdict), ("property_check", propertyCheck), ("property_reading", readingFn), ("class_of", classOfFn)]

end RemoteAudit.PostSeed
