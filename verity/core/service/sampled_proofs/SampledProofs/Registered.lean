/-
Registered values (`one_stage/registered.py`) as the registration reads them: the verifier's own entry for each parameter
it expects registered (`entries`, Python's `registered_entries`), from its own program.

A program's descriptor names no parameters: its root body starts with the input nodes each parameter of the program
function lowers to (PROTOCOL.md §5.1), and the function's signature (`program.fn.signature()`, which Python's verifier holds
in its `Program`) is not part of it. So the verifier holds its program function's parameters too (`params`: `[[name, type],
...]`, written as a body's params are), and they must lower to exactly the descriptor's input nodes; a parameter's Input
gates are then its own nodes' gates.

* `Param.gates`: the parameter's Input gates, when they are one interval (`parameter_gates`; a parameter without leaves has
  none);
* rows: its leaves read `words` at a time (`shape`); in a bit row (`hm96-sha512/row/v2`, `hm96-sha512/row-seg/v1`, the
  value's row format when the verifier names one), as many of its leaves as `words` leaves of `width` bits hold
  (`row_leaves`; H1's A, K int8 codes in K/2 u16 words);
* `width`: its leaf width rounded up to whole u16 words, one width of at most 32 bits;
* `domain`: `CommitmentDomain(B, -2, RangeIndexedDomain(0, rows), hash="sha512").domain_id`, `B` the bytes of
  `identity_digest("verity/one-stage/registered-domain/v0", {"program", "value", "schema", "words", "leaves"})` with
  `"word_bits"` beside them when it is not 16: SHA-512 of the frame-v3 `domain` message over `B`, `uint(-2 + 2)`, the range's
  identity `identity_digest_sha512("veritor/indexed-domain/range/v1", {"start": 0, "step": 1, "stop": rows})` and
  `uint(rows)`; a per-window value's binding adds `"window": R`, the window's open root (`domainAt`), so a root
  registered in one window opens under no other window's domain, and a value held across windows leaves it out.
-/
import Flock.Program
import Flock.Public
import SampledProofs.Partition

namespace SampledProofs.Registered

open Lean (Json)
open Flock (V)
open Flock.Program (Ty)
open Flock.TemplateQuery (Desc)

def REGISTERED_DOMAIN : String := "verity/one-stage/registered-domain/v0"
def SCHEMA : String := Flock.HmRow.ROW_SCHEMA
/-- The row formats a registered value may be read in (`registered.SCHEMAS`). -/
def SCHEMAS : List String := [Flock.HmRow.ROW_SCHEMA, Flock.HmRow.ROW_SCHEMA_V2, Flock.HmRow.ROW_SCHEMA_SEG]
/-- The default leaf width, which the domain and the entry name only by leaving `word_bits` out. -/
def WORD_BITS : Nat := 16
def WIDTHS : List Nat := [16, 32]
def WIDTH_KEY : String := "word_bits"
/-- The domain binding's key a per-window value adds: the window's open root. -/
def WINDOW_KEY : String := "window"
def ENTRY_KEYS : Array String := sortStrings #["value", "root", "leaves", "words", "domain"]

/-! A type's element or field is smaller than it: `widthSet`'s and `lower`'s termination. -/

private theorem lt_array (n : Nat) (e : Ty) : sizeOf e < sizeOf (Ty.array n e) := by simp; omega

private theorem lt_tuple {f : Ty} {fs : Array Ty} (h : f ∈ fs) : sizeOf f < sizeOf (Ty.tuple fs) := by
  have := Array.sizeOf_lt_of_mem h; simp; omega

private theorem lt_record {f : String × Ty} {fs : Array (String × Ty)} (h : f ∈ fs) :
    sizeOf f.2 < sizeOf (Ty.record fs) := by
  have := Array.sizeOf_lt_of_mem h; cases f; simp at *; omega

/-- `Type.width_set`: the leaf widths, computed structurally (an `Array` of no elements has none). -/
def widthSet : Ty → List Nat
  | .value w => [w]
  | .array n e => if n == 0 then [] else widthSet e
  | .tuple fs => (fs.toList.attach.flatMap fun ⟨f, _⟩ => widthSet f).eraseDups
  | .record fs => (fs.toList.attach.flatMap fun ⟨f, _⟩ => widthSet f.2).eraseDups
termination_by t => sizeOf t
decreasing_by
  · exact lt_array _ _
  · exact lt_tuple (Array.mem_def.mpr ‹_›)
  · exact lt_record (Array.mem_def.mpr ‹_›)

/-- §5.1 (`program._lower_input`): the input nodes a parameter of type `t` lowers to, `(width, n)` each: one node when its
leaves are one width, else each field or element in order. -/
def lower (t : Ty) : Array (Nat × Nat) :=
  match widthSet t with
  | [w] => #[(w, t.leaves)]
  | _ => match t with
    | .array n e => let l := lower e; (Array.range n).foldl (fun acc _ => acc ++ l) #[]
    | .tuple fs => fs.attach.foldl (fun acc ⟨f, _⟩ => acc ++ lower f) #[]
    | .record fs => fs.attach.foldl (fun acc ⟨f, _⟩ => acc ++ lower f.2) #[]
    | .value _ => #[]
termination_by sizeOf t
decreasing_by
  · exact lt_array _ _
  · exact lt_tuple ‹_›
  · exact lt_record ‹_›

/-- A parameter of the verifier's program function: its name, its type, and its Input gates `(first, count)` when they are
one non-empty interval. -/
structure Param where
  name : String
  ty : Ty
  gates : Option (Nat × Nat)

/-- The root body's input nodes, in body order: each one's width, member count and first gate. -/
def inputNodes (d : Desc) : Array (Nat × Nat × Nat) := Id.run do
  let some root := d.def? d.root | return #[]
  let mut out := #[]
  let mut at_ := 0
  for x in root.nodes do
    let id := d.callees.getD x.fn ""
    if let some w := Flock.Program.inputWidth? id then out := out.push (w, x.n, at_)
    at_ := at_ + x.members * d.gates id
  return out

/-- The verifier's program function's parameters (`[[name, type], ...]`), distinct names whose types lower to exactly the
root body's input nodes, with each one's Input gates. -/
def params (d : Desc) (j : Json) : V (Array Param) := do
  let .arr ps := j | throw "parameters is not a list of [name, type]"
  let ps ← ps.mapM fun p => match p with
    | .arr #[.str k, t] => do pure (k, ← Ty.ofJson t)
    | _ => throw "parameters is not a list of [name, type]"
  if (sortStrings (ps.map (·.1))).toList.eraseDups.length != ps.size then throw "parameters: a name twice"
  let nodes := inputNodes d
  let lowered := ps.map (lower ·.2)
  if (lowered.foldl (· ++ ·) #[]) != nodes.map (fun (w, n, _) => (w, n)) then
    throw "parameters: they do not lower to the program's input nodes"
  let mut out := #[]
  let mut at_ := 0
  for (name, ty) in ps, l in lowered do
    let own := nodes.extract at_ (at_ + l.size)
    at_ := at_ + l.size
    let gates := match own[0]? with
      | none => none
      | some (_, _, first) =>
        let count := own.foldl (fun acc (_, n, _) => acc + n) 0
        let contiguous := (List.range own.size).all fun i =>
          let (_, _, g) := own[i]!
          g == first + (own.extract 0 i).foldl (fun acc (_, n, _) => acc + n) 0
        if contiguous && count > 0 then some (first, count) else none
    out := out.push { name, ty, gates }
  return out

/-- `registered.width`: the parameter's leaf width rounded up to whole u16 words, one width, at most 32 bits. -/
def width (p : Param) : V Nat :=
  match widthSet p.ty with
  | [w] => if 0 < w && w ≤ 32 then pure (16 * ((w + 15) / 16)) else throw s!"parameter {p.name}: its leaves are {w} bits"
  | ws => throw s!"parameter {p.name}: its leaves are not one width ({ws})"

private def tagged (frame : String) (H : ByteArray → ByteArray) (tag : String) (manifest : Json) : V ByteArray := do
  let t := tag.toUTF8
  let m := (← Flock.canonUtf8 manifest).toUTF8
  return H (frame.toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 m.size.toUInt64 ++ m)

/-- `registered.domain(program, value, words, leaves, word_bits).domain_id`, `leaves` the value's rows. -/
def domain (program value : String) (words leaves wordBits : Nat) : V ByteArray := do
  if !WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  let manifest := Json.mkObj ([("program", Json.str program), ("value", Json.str value), ("schema", Json.str SCHEMA),
    ("words", Lean.toJson words), ("leaves", Lean.toJson leaves)] ++
    (if wordBits == WORD_BITS then [] else [(WIDTH_KEY, Lean.toJson wordBits)]))
  let binding ← tagged "veritor/tagged-sha256/v1" Flock.Sha256.hash REGISTERED_DOMAIN manifest
  let positions ← tagged "verity/tagged-sha512/v1" Flock.Sha512.hash "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson leaves)])
  return Flock.FV3.hash "domain" [binding, Flock.FV3.uint 0, positions, Flock.FV3.uint leaves] Flock.Sha512.hash

/-- `domain` with `(key, hex)` beside its binding's keys: a per-window value's `"window"`, a produced value's
`"produced"` (`SampledProofs.Produced`). -/
def domainBound (program value : String) (words leaves wordBits : Nat) (key r : String) : V ByteArray := do
  if !WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  let manifest := Json.mkObj ([("program", Json.str program), ("value", Json.str value), ("schema", Json.str SCHEMA),
    ("words", Lean.toJson words), ("leaves", Lean.toJson leaves)] ++
    (if wordBits == WORD_BITS then [] else [(WIDTH_KEY, Lean.toJson wordBits)]) ++ [(key, Json.str r)])
  let binding ← tagged "veritor/tagged-sha256/v1" Flock.Sha256.hash REGISTERED_DOMAIN manifest
  let positions ← tagged "verity/tagged-sha512/v1" Flock.Sha512.hash "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson leaves)])
  return Flock.FV3.hash "domain" [binding, Flock.FV3.uint 0, positions, Flock.FV3.uint leaves] Flock.Sha512.hash

/-- `registered.domain(program, value, words, leaves, word_bits, window).domain_id`: `domain`, and for a per-window value
(`some r`, `r` the window's open root) the same with `"window": r` in the binding. -/
def domainAt (program value : String) (words leaves wordBits : Nat) : Option String → V ByteArray
  | none => domain program value words leaves wordBits
  | some r => domainBound program value words leaves wordBits WINDOW_KEY r

/-- The verifier's own entry for one registered value: its rows (`leaves`), their `words`, `word_bits` and domain (hex). -/
structure Entry where
  value : String
  leaves : Nat
  words : Nat
  wordBits : Nat
  domain : String

/-- `registered.row_leaves`: the parameter's leaves in one row of `words` leaves of `b` bits, `words` in a `row/v1` row,
else as many of its own leaves as the row's `words * b` bits hold, an error when they hold no whole number. -/
def rowLeaves (p : Param) (schema : String) (words b : Nat) : V Nat :=
  if schema == SCHEMA || words < 1 then pure words else
  match widthSet p.ty with
  | [w] => if w == 0 || (words * b) % w != 0 then
      throw s!"parameter {p.name}: a {schema} row of {words} {b}-bit leaves is no whole number of its {w}-bit leaves"
    else pure (words * b / w)
  | ws => throw s!"parameter {p.name}: its leaves are not one width ({ws})"

/-- `registered_entries`: the verifier's own entry for each parameter it expects registered (`values`: `(parameter,
words)`, in order), each in its row format (`schemas`: `(parameter, schema)`, `SCHEMA` for one it doesn't name), or an
error where Python raises: no such parameter, its gates not one interval, its leaves not one width of at most 32 bits,
or a row not dividing them into rows. `window`: the window's open root, which every value's domain binds but the ones in
`fixed`, the values held across windows (`domainAt`). -/
def entries (d : Desc) (ps : Array Param) (values : Array (String × Nat)) (schemas : Array (String × String) := #[])
    (window : Option String := none) (fixed : Array String := #[]) : V (Array Entry) :=
  values.mapM fun (v, w) => do
    let some p := ps.find? (·.name == v) | throw s!"{v} is not a parameter of the program"
    let some (_, n) := p.gates | throw s!"parameter {v}: its gates are not one interval"
    let b ← width p
    let k ← rowLeaves p ((schemas.find? (·.1 == v)).elim SCHEMA (·.2)) w b
    if k < 1 || n % k != 0 then throw s!"parameter {v}: {n} leaves are not rows of {w}"
    return { value := v, leaves := n / k, words := w, wordBits := b,
             domain := Flock.hexOf (← domainAt (Flock.hexOf d.sha512) v w (n / k) b
               (if fixed.contains v then none else window)) }

def Entry.toJson (e : Entry) : Json :=
  Json.mkObj [("leaves", Lean.toJson e.leaves), ("words", Lean.toJson e.words), ("word_bits", Lean.toJson e.wordBits),
    ("domain", Json.str e.domain)]

end SampledProofs.Registered
