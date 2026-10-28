/-
A window circuit's wires, checked at the size of their description (`verity.ml.windows.interface.Wire`).

A window's circuit is a population of template instances (`TemplatePopulations_v1`), whose inputs are all Input gates: the
program holds no dataflow between units. The wires are that dataflow. A wire states, once for many units, that reader
template `reader`'s units read rows of their parameter `port` whole as row `out` of a unit of template `writer`:

* **The reader rows a wire covers.** A reader port is cut into `k` rows (`cuts`, default 1): row `j` is the run of the
  parameter's outer elements `[j·n/k, (j+1)·n/k)`, `k` dividing the outer length `n` and the port's words, each its own
  hm96 row. A wire covers rows `cut0 … cut0 + cuts − 1` of units `at …` of the reader: reader index `i < units` (the
  product of `radix`) is unit `at + i / cuts`, row `cut0 + i % cuts`, and when `cuts > 1` the last digit of `radix` is
  `cuts`, so the row is that digit. With no cut the wire is a whole-port wire.
* **Its writers.** Reader index `i`'s writer is unit `base + Σ_k d_k · step_k` of `writer`, `d` the digits of `i` in mixed
  radix `radix` (the first digit the slowest); a step of 0 is a fan-out. Each index counts its own template's units.

`check` decides a circuit's wires in work proportional to its templates and wires, never to the units they cover (AGENTS.md
"Design rules"): each wire's writers lie in `span`, the least and greatest writer in closed
form; and two wires wire no reader row twice exactly when they read different ports or their rectangles (reader units ×
rows) are disjoint, compared pair by pair (#wires² comparisons, each of a few fields). Its summary gives each reader row's writer (`Summary.writerOf`, by the
wires alone) and the wired rows and bits multiplied through (`wiredRows`, `wiredBits`), the wired term of the count of a
window's free inputs.

`checkRef` is the reference it is held to, every covered reader row expanded (`Wire.entries`): the difftest (the `wires`
and `wires_unrolled` pieces) holds both to the same Python reference. The verifier never runs it.

The refusals, in the order checked (`Code`): `cuts` (a cut names no template or port, or `k` is 0 or doesn't divide the
port's outer length and words), then wire by wire `template` (its reader or writer is no template), `port` (its port is no
input of the reader, or `out` no output of the writer), `cut` (its rows are not among the port's, or the last digit isn't
`cuts`, or `cuts` doesn't divide the wire's reader rows), `width` (a reader row's words are not the writer row's), `bounds` (a reader unit or a writer outside its
template), then `overlap` (some reader row wired twice).
-/
import Lean.Data.Json

namespace SampledProofs.Wires

open Lean (Json JsonNumber toJson)

/-- A reader input port: its u16 words, and its outer length (the parameter's outermost array length, 1 when it is not
an array), which its cut divides. -/
structure Port where
  words : Nat
  outer : Nat
  deriving Repr, BEq

/-- A template of the circuit: its units, its input ports and its output rows' words. -/
structure Template where
  units : Nat
  rows : List (String × Port)
  outs : List (String × Nat)
  deriving Repr

/-- A wire (the module docstring); `start` is the record's `at`. -/
structure Wire where
  reader : String
  port : String
  writer : String
  out : String
  start : Nat
  radix : List Nat
  base : Int
  step : List Int
  cut0 : Nat
  cuts : Nat
  deriving Repr, BEq

/-- The templates by descriptor id, the cuts (`{tid: {port: k}}`) and the wires. -/
structure Circuit where
  templates : List (String × Template)
  cuts : List (String × List (String × Nat))
  wires : List Wire
  deriving Repr

/-- The number of rows template `tid`'s port `p` is cut into (absent: 1). -/
def Circuit.cutsOf (c : Circuit) (tid p : String) : Nat := (((c.cuts.lookup tid).getD []).lookup p).getD 1

/-- The digits' radices and steps, the fastest digit first. -/
def Wire.pairs (w : Wire) : List (Nat × Int) := (w.radix.zip w.step).reverse

/-- The product of the radices of `ps`. -/
def prodR (ps : List (Nat × Int)) : Nat := ps.foldr (fun p n => p.1 * n) 1

/-- The reader rows the wire covers: the product of its radices. -/
def Wire.units (w : Wire) : Nat := prodR w.pairs

/-- `Σ_k d_k · step_k` for index `i`, `ps` the radices and steps fastest first (`Wire.pairs`). -/
def accum : List (Nat × Int) → Nat → Int
  | [], _ => 0
  | (r, s) :: ps, i => ((i % r : Nat) : Int) * s + accum ps (i / r)

/-- Reader index `i`'s writer unit. -/
def Wire.writerAt (w : Wire) (i : Nat) : Int := w.base + accum w.pairs i

/-- `Σ_k min 0 (step_k · (radix_k − 1))`. -/
def lowOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => min 0 (s * ((r : Int) - 1)) + lowOf ps

/-- `Σ_k max 0 (step_k · (radix_k − 1))`. -/
def highOf : List (Nat × Int) → Int
  | [] => 0
  | (r, s) :: ps => max 0 (s * ((r : Int) - 1)) + highOf ps

/-- The least and the greatest writer the wire reads, in closed form. -/
def Wire.span (w : Wire) : Int × Int := (w.base + lowOf w.pairs, w.base + highOf w.pairs)

/-- Reader index `i`'s reader unit and row. -/
def Wire.readerAt (w : Wire) (i : Nat) : Nat × Nat := (w.start + i / w.cuts, w.cut0 + i % w.cuts)

/-- The reader units the wire covers. -/
def Wire.readerUnits (w : Wire) : Nat := w.units / w.cuts

/-! ## Refusals -/

inductive Code where
  | cuts | template | port | cut | width | bounds | overlap
  deriving Repr, BEq, DecidableEq

def Code.name : Code → String
  | .cuts => "cuts" | .template => "template" | .port => "port" | .cut => "cut" | .width => "width"
  | .bounds => "bounds" | .overlap => "overlap"

/-- A refusal: its code, and for a wire's own refusal the wire's index. -/
structure Refusal where
  code : Code
  wire : Option Nat
  deriving Repr, BEq

/-- The first malformed cut: a template or port it names that isn't one, or `k` that is 0 or doesn't divide the port's outer
length and words. -/
def cutsOk (c : Circuit) : Bool :=
  c.cuts.all fun (tid, kv) => match c.templates.lookup tid with
    | some t => kv.all fun (p, k) => match t.rows.lookup p with
      | some q => 0 < k && q.outer % k == 0 && q.words % k == 0
      | none => false
    | none => false

/-- The wire's own refusal, before its bounds: its templates, ports, rows and widths. With them, the reader and writer
templates, the reader row's words and the reader's cut. -/
def shape (c : Circuit) (w : Wire) : Except Code (Template × Template) :=
  match c.templates.lookup w.reader, c.templates.lookup w.writer with
  | some rt, some wt =>
    match rt.rows.lookup w.port, wt.outs.lookup w.out with
    | some p, some ow =>
      let k := c.cutsOf w.reader w.port
      if !(1 ≤ w.cuts && w.cut0 + w.cuts ≤ k && (w.cuts == 1 || w.radix.getLast? == some w.cuts) &&
          w.units % w.cuts == 0) then .error .cut
      else if p.words / k != ow then .error .width
      else .ok (rt, wt)
    | _, _ => .error .port
  | _, _ => .error .template

/-- The bounds in closed form: reader units `[start, start + readerUnits)` among the reader's, the span among the writer's. -/
def boundsOk (w : Wire) (rt wt : Template) : Bool :=
  w.start + w.readerUnits ≤ rt.units && 0 ≤ w.span.1 && w.span.2 < (wt.units : Int)

/-- Two wires of one reader port wire disjoint rectangles: their reader units or their rows are disjoint. -/
def disjoint (a b : Wire) : Bool :=
  a.reader != b.reader || a.port != b.port ||
  a.start + a.readerUnits ≤ b.start || b.start + b.readerUnits ≤ a.start ||
  a.cut0 + a.cuts ≤ b.cut0 || b.cut0 + b.cuts ≤ a.cut0

/-- Every pair of `ws` disjoint. -/
def pairwiseDisjoint : List Wire → Bool
  | [] => true
  | w :: ws => ws.all (disjoint w) && pairwiseDisjoint ws

/-- What `check` establishes: the wires, in order. -/
structure Summary where
  wires : List Wire
  rowWords : List Nat
  deriving Repr

/-- The first wire refusal, by index. -/
def firstWire (c : Circuit) (f : Wire → Option Code) : Option Refusal :=
  (c.wires.zipIdx.find? fun (w, _) => (f w).isSome).bind fun (w, i) => (f w).map fun e => ⟨e, some i⟩

/-- The wire's refusal, with the closed-form bounds. -/
def wireCode (c : Circuit) (w : Wire) : Option Code :=
  match shape c w with
  | .error e => some e
  | .ok (rt, wt) => if boundsOk w rt wt then none else some .bounds

/-- The reader row's words of a wire whose shape passed. -/
def rowWords (c : Circuit) (w : Wire) : Nat :=
  match c.templates.lookup w.reader with
  | some rt => match rt.rows.lookup w.port with
    | some p => p.words / c.cutsOf w.reader w.port
    | none => 0
  | none => 0

/-- **The check at the size of the description**: the refusals in the module docstring's order, else the summary. -/
def check (c : Circuit) : Except Refusal Summary :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCode c) with
    | some r => .error r
    | none =>
      if !pairwiseDisjoint c.wires then .error ⟨.overlap, none⟩
      else .ok ⟨c.wires, c.wires.map (rowWords c)⟩

/-- Whether wire `w` covers reader row `(tid, u, port, row)`, and if so its index. -/
def Wire.indexOf (w : Wire) (tid : String) (u : Nat) (port : String) (row : Nat) : Option Nat :=
  if w.reader == tid && w.port == port && w.start ≤ u && u < w.start + w.readerUnits && w.cut0 ≤ row &&
     row < w.cut0 + w.cuts then some ((u - w.start) * w.cuts + (row - w.cut0)) else none

/-- The writer `(template, unit, out)` of reader row `(tid, u, port, row)`, by the wires alone; none when no wire covers it. -/
def Summary.writerOf (s : Summary) (tid : String) (u : Nat) (port : String) (row : Nat) : Option (String × Int × String) :=
  s.wires.findSome? fun w => (w.indexOf tid u port row).map fun i => (w.writer, w.writerAt i, w.out)

/-- The reader rows wired, multiplied through. -/
def Summary.wiredRows (s : Summary) : Nat := (s.wires.map Wire.units).sum

/-- The bits of the reader rows wired, multiplied through. -/
def Summary.wiredBits (s : Summary) : Nat := ((s.wires.zip s.rowWords).map fun (w, n) => w.units * (16 * n)).sum

/-! ## The reference: every covered reader row expanded

The verifier never runs it: it is what `check` is held to, by the difftest and the agreement lemmas. -/

/-- A reader row `(template, unit, port, row)` and its writer `(template, unit, out)`. -/
abbrev Entry := (String × Nat × String × Nat) × (String × Int × String)

/-- Every reader row the wire covers, with its writer, index by index. -/
def Wire.entries (w : Wire) : List Entry :=
  (List.range w.units).map fun i =>
    ((w.reader, (w.readerAt i).1, w.port, (w.readerAt i).2), (w.writer, w.writerAt i, w.out))

/-- The bounds entry by entry. -/
def boundsRef (w : Wire) (rt wt : Template) : Bool :=
  w.entries.all fun ((_, u, _, _), (_, v, _)) => u < rt.units && 0 ≤ v && v < (wt.units : Int)

def wireCodeRef (c : Circuit) (w : Wire) : Option Code :=
  match shape c w with
  | .error e => some e
  | .ok (rt, wt) => if boundsRef w rt wt then none else some .bounds

/-- No reader row twice among `es` (quadratic in the entries). -/
def distinctRows (es : List Entry) : Bool := decide (es.map (·.1)).Nodup

/-- **The unrolled reference**: `check`'s refusals with the bounds and the overlap taken entry by entry, else every entry. -/
def checkRef (c : Circuit) : Except Refusal (List Entry) :=
  if !cutsOk c then .error ⟨.cuts, none⟩
  else match firstWire c (wireCodeRef c) with
    | some r => .error r
    | none =>
      let es := c.wires.flatMap Wire.entries
      if !distinctRows es then .error ⟨.overlap, none⟩ else .ok es

/-! ## JSON: the difftest's pieces

An instance is `{"templates", "cuts", "wires", "queries"}`:

* `templates`: `{tid: {"units", "rows": {port: [words, outer]}, "outs": {out: words}}}`, every number a natural, `units`
  and `words` at least 1;
* `cuts`: `{tid: {port: k}}`;
* `wires`: V3a's wire records, `{reader, port, writer, out, at, radix, base, step}` and optionally `cut: [c0, C]`: strings,
  `at` a natural, `radix` at least one natural each at least 1, `base` an integer, `step` one integer a digit, and `C` at
  least 1;
* `queries`: reader rows `[tid, unit, port, row]` to look up (a drawn unit's).

Anything else is refused as malformed. The output is `{"ok": false, "code", "wire"}` (`wire` null but for a wire's own
refusal) or `{"ok": true, "wired_rows", "wired_bits", "spans", "lookups"}`: per wire its least and greatest writer, and per
query its writer `[tid, unit, out]` or null. -/

def int? : Json → Option Int
  | .num n => if n.exponent == 0 then some n.mantissa else none
  | _ => none

def nat? (j : Json) : Option Nat := (int? j).bind fun i => if 0 ≤ i then some i.toNat else none

def keysOf : Json → Option (List String)
  | .obj kvs => some (kvs.toArray.map (·.1)).toList
  | _ => none

def fields : Json → Except String (List (String × Json))
  | .obj kvs => .ok (kvs.toArray.map fun ⟨k, v⟩ => (k, v)).toList
  | _ => .error "not an object"

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"no {k}"

def natAt (j : Json) (k : String) : Except String Nat := do
  match nat? (← field j k) with
  | some n => .ok n
  | none => .error s!"{k} is not a natural"

def natOf (what : String) (j : Json) : Except String Nat :=
  match nat? j with
  | some n => .ok n
  | none => .error s!"{what} is not a natural"

def intOf (what : String) (j : Json) : Except String Int :=
  match int? j with
  | some n => .ok n
  | none => .error s!"{what} is not an integer"

def strOf (what : String) : Json → Except String String
  | .str s => .ok s
  | _ => .error s!"{what} is not a string"

def arrOf (what : String) : Json → Except String (List Json)
  | .arr a => .ok a.toList
  | _ => .error s!"{what} is not a list"

def sameKeys (j : Json) (ks : List String) (what : String) : Except String Unit :=
  match keysOf j with
  | some got => if got.mergeSort (· ≤ ·) == ks.mergeSort (· ≤ ·) then .ok () else .error s!"{what}: keys {got}, not {ks}"
  | none => .error s!"{what} is not an object"

def WIRE_KEYS : List String := ["reader", "port", "writer", "out", "at", "radix", "base", "step"]

def Wire.ofJson (j : Json) : Except String Wire := do
  let ks := (keysOf j).getD []
  if ks.contains "cut" then sameKeys j ("cut" :: WIRE_KEYS) "a wire" else sameKeys j WIRE_KEYS "a wire"
  let s := fun k => do strOf k (← field j k)
  let radix ← (← arrOf "radix" (← field j "radix")).mapM (natOf "a radix")
  let step ← (← arrOf "step" (← field j "step")).mapM (intOf "a step")
  if radix.isEmpty || radix.length != step.length || radix.any (· < 1) then
    throw "a wire is at least one digit of radix 1 or more, and one step a digit"
  let (cut0, cuts) ← match j.getObjVal? "cut" with
    | .ok cj => do
      let xs ← (← arrOf "cut" cj).mapM (natOf "a cut")
      match xs with
      | [a, b] => if 1 ≤ b then pure (a, b) else throw "a wire's cut is [c0, C], C at least 1"
      | _ => throw "a wire's cut is [c0, C]"
    | .error _ => pure (0, 1)
  return { reader := ← s "reader", port := ← s "port", writer := ← s "writer", out := ← s "out",
           start := ← natAt j "at", radix, base := ← intOf "base" (← field j "base"), step, cut0, cuts }

def Template.ofJson (j : Json) : Except String Template := do
  sameKeys j ["units", "rows", "outs"] "a template"
  let units ← natAt j "units"
  if units < 1 then throw "a template has at least one unit"
  let rows ← (← fields (← field j "rows")).mapM fun (p, v) => do
    match ← arrOf "a row" v with
    | [a, b] =>
      let (words, outer) := (← natOf "words" a, ← natOf "outer" b)
      if words < 1 || outer < 1 then throw s!"row {p}: words and outer at least 1"
      pure (p, ({ words, outer } : Port))
    | _ => throw s!"row {p} is [words, outer]"
  let outs ← (← fields (← field j "outs")).mapM fun (o, v) => do
    let n ← natOf "an output's words" v
    if n < 1 then throw s!"output {o}: at least one word"
    pure (o, n)
  return { units, rows, outs }

def Circuit.ofJson (j : Json) : Except String Circuit := do
  let ts ← (← fields (← field j "templates")).mapM fun (tid, v) => do pure (tid, ← Template.ofJson v)
  let cs ← (← fields (← field j "cuts")).mapM fun (tid, v) => do
    pure (tid, ← (← fields v).mapM fun (p, k) => do pure (p, ← natOf "a cut" k))
  let wires ← (← arrOf "wires" (← field j "wires")).mapM Wire.ofJson
  return { templates := ts, cuts := cs, wires }

def queriesOf (j : Json) : Except String (List (String × Nat × String × Nat)) := do
  (← arrOf "queries" (← field j "queries")).mapM fun q => do
    match ← arrOf "a query" q with
    | [a, b, c, d] => pure (← strOf "a query's template" a, ← natOf "a query's unit" b, ← strOf "a query's port" c,
                            ← natOf "a query's row" d)
    | _ => throw "a query is [template, unit, port, row]"

def refusalJson (r : Refusal) : Json :=
  Json.mkObj [("ok", false), ("code", r.code.name), ("wire", match r.wire with | some i => toJson i | none => Json.null)]

def writerJson : Option (String × Int × String) → Json
  | some (t, v, o) => Json.arr #[toJson t, toJson v, toJson o]
  | none => Json.null

def instanceCheck (sameKeysOk : Json → Except String Unit) (j : Json) : Except String (Circuit × List (String × Nat × String × Nat)) := do
  sameKeysOk j
  return (← Circuit.ofJson j, ← queriesOf j)

/-- The `wires` piece: `check` on an instance (the section's docstring). -/
def piece (j : Json) : Except String Json := do
  let (c, qs) ← instanceCheck (sameKeys · ["templates", "cuts", "wires", "queries"] "an instance") j
  match check c with
  | .error r => return refusalJson r
  | .ok s =>
    return Json.mkObj [("ok", true), ("wired_rows", toJson s.wiredRows), ("wired_bits", toJson s.wiredBits),
      ("spans", Json.arr (s.wires.map fun w => Json.arr #[toJson w.span.1, toJson w.span.2]).toArray),
      ("lookups", Json.arr (qs.map fun (t, u, p, r) => writerJson (s.writerOf t u p r)).toArray)]

/-- The `wires_unrolled` piece: `checkRef` on an instance, its summary read off the entries. -/
def pieceRef (j : Json) : Except String Json := do
  let (c, qs) ← instanceCheck (sameKeys · ["templates", "cuts", "wires", "queries"] "an instance") j
  match checkRef c with
  | .error r => return refusalJson r
  | .ok es =>
    let spans := c.wires.map fun w =>
      let vs := w.entries.map (·.2.2.1)
      Json.arr #[toJson (vs.foldl min (vs.headD 0)), toJson (vs.foldl max (vs.headD 0))]
    let bits := (c.wires.map fun w => (w.entries.map fun _ => 16 * rowWords c w).sum).sum
    return Json.mkObj [("ok", true), ("wired_rows", toJson es.length), ("wired_bits", toJson bits),
      ("spans", Json.arr spans.toArray),
      ("lookups", Json.arr (qs.map fun (t, u, p, r) =>
        writerJson ((es.find? (·.1 == (t, u, p, r))).map (·.2))).toArray)]

end SampledProofs.Wires
