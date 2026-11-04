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

The v2 audit (`OneStageZk`) holds a statement to the registration record: each port of the verifier's registered reads
to the record's entry for its value (`entryClash`), and every other port of the statement to the record's root and
domain for `<member>/<port>` (`statementHeld`, at the record's `roots`: `rootsOf`).
-/
import Flock.Program
import Flock.Public
import Flock.Registered
import SampledProofs.Partition
import Std.Data.HashMap

namespace SampledProofs.Registered

open Lean (Json)
open Flock (V)
open Flock.Program (Ty)
open Flock.TemplateQuery (Desc)

/- A registered value's domain, its leaf widths, its row formats (`registered.SCHEMAS`) and the default width, which the
domain and the entry name only by leaving `word_bits` out: C-Flock's, which reads the same registered values. -/
export Flock.Registered (REGISTERED_DOMAIN SCHEMAS WORD_BITS WIDTHS WIDTH_KEY domain)

def SCHEMA : String := Flock.HmRow.ROW_SCHEMA
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

/-- `inputNodes` with each root callee's gates read off `Partition.gTable`, computed once. -/
def inputNodesFast (d : Desc) : Array (Nat × Nat × Nat) := Id.run do
  let some root := d.def? d.root | return #[]
  let G := Partition.gTable d
  let mut out := #[]
  let mut at_ := 0
  for x in root.nodes do
    let id := d.callees.getD x.fn ""
    if let some w := Flock.Program.inputWidth? id then out := out.push (w, x.n, at_)
    at_ := at_ + x.members * Partition.look d G id
  return out

@[csimp] theorem inputNodes_eq_inputNodesFast : @inputNodes = @inputNodesFast := by
  funext d
  simp only [inputNodes, inputNodesFast, Partition.look_gTable]

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

/-- `domain` with `(key, hex)` beside its binding's keys: a per-window value's `"window"`, a produced value's
`"produced"` (`SampledProofs.Produced`). -/
def domainBound (program value : String) (words leaves wordBits : Nat) (key r : String) : V ByteArray :=
  Flock.Registered.domainWith program value words leaves wordBits [(key, Json.str r)]

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

/-! ### The verifier's registered ports against the record's entries -/

/-- `registered.entry_of`: the record's entry for `value`, the last of its `registered` entries naming it (as R7 reads
it, and Python's dict). -/
def entryOf (rec : Json) (value : String) : Option Json :=
  match getD rec "registered" with
  | .arr es => es.reverse.find? fun e => jeq (getD e "value") (.str value)
  | _ => none

/-- The leaf schema the verifier counts `value`'s rows in (`entries`'): its row format's (`schemas`), else `SCHEMA`. -/
def schemaOf (schemas : Array (String × String)) (value : String) : String :=
  (schemas.find? (·.1 == value)).elim SCHEMA (·.2)

/-- Port `q` of the verifier's registered reads against the record's entry `e` for its value, field by field, each key
with whether they agree: the root and domain (lowercase hex), the leaf count, the words (none when the port names none),
the word bits (16 when the entry leaves them out), and the leaf schema, which the entry leaves to the verifier
(`schemaOf`). -/
def entryFields (schemas : Array (String × String)) (q : Flock.Registered.Port) (e : Json) : List (String × Bool) :=
  [("root", jeq (getD e "root") (.str (Flock.hexOf q.root))),
   ("leaves", int? (getD e "leaves") == some (q.leaves : Int)),
   ("words", int? (getD e "words") == q.words.map fun w => (w : Int)),
   (WIDTH_KEY, (field? e WIDTH_KEY).elim (some (WORD_BITS : Int)) int? == some (q.wordBits : Int)),
   ("domain", jeq (getD e "domain") (.str (Flock.hexOf q.domain))),
   (Flock.Registered.SCHEMA_KEY, q.schema == schemaOf schemas q.value)]

/-- The first field of `entryFields` on which port `q` and the entry `e` differ. -/
def entryDiff? (schemas : Array (String × String)) (q : Flock.Registered.Port) (e : Json) : Option String :=
  ((entryFields schemas q e).find? (!·.2)).map (·.1)

/-- `registered.entry_clash`: the first port of `own` that differs from the record's entry for its value (`entryOf`),
with the field (`entryDiff?`). A port of a value the record has no entry for is held by nothing here. -/
def entryClash (schemas : Array (String × String)) (rec : Json) (own : Array Flock.Registered.Port) :
    Option (Flock.Registered.Port × String) :=
  own.findSome? fun q => (entryOf rec q.value).bind fun e => (entryDiff? schemas q e).map (q, ·)

theorem entryDiff?_none {schemas : Array (String × String)} {q : Flock.Registered.Port} {e : Json}
    (h : entryDiff? schemas q e = none) : ∀ f ∈ entryFields schemas q e, f.2 = true := by
  intro f hf
  simpa using List.find?_eq_none.mp (Option.map_eq_none_iff.mp h) f hf

/-- **No port differs from its value's entry**: where `entryClash` finds none, every port of `own` whose value the record
has an entry for agrees with it on every field of `entryFields`: root, leaf count, words, word bits, domain and schema. -/
theorem entryClash_none {schemas : Array (String × String)} {rec : Json} {own : Array Flock.Registered.Port}
    (h : entryClash schemas rec own = none) :
    ∀ q ∈ own, ∀ e, entryOf rec q.value = some e → ∀ f ∈ entryFields schemas q e, f.2 = true := by
  intro q hq e he
  apply entryDiff?_none
  have h1 := Array.findSome?_eq_none_iff.mp h q hq
  rw [he] at h1
  exact Option.map_eq_none_iff.mp h1

/-! ### A statement's other ports against the record's roots -/

/-- `registered.roots_of`: the registration record's port roots (`roots`) by port name, each its root and domain as bytes,
the last entry naming the port, none when that entry's root or domain is not lowercase hex (R3 refuses a record naming a
port twice). In the per-member form a port is `<member>/<port>`, the member's name in the parameters. -/
def rootsOf (rec : Json) : Std.HashMap String (ByteArray × ByteArray) :=
  let roots := match getD rec "roots" with | .arr rs => rs | _ => #[]
  roots.foldl (init := {}) fun m r =>
    match getD r "port" with
    | .str port =>
      match getD r "root", getD r "domain" with
      | .str root, .str dom =>
        match Flock.unhex root, Flock.unhex dom with
        | .ok rb, .ok db => m.insert port (rb, db)
        | _, _ => m.erase port
      | _, _ => m.erase port
    | _ => m

/-- `registered.statement_clash`: of the statement's ports `ports`, the first that is none of `registered` and whose
frame-v3-sha512 root and domain in the public header (`Public.fv3Bytes`) are not the ones `rootAt` gives for
`<member>/<port>`: with `false` when it gives none, `true` when they differ. -/
def statementClash (rootAt : String → Option (ByteArray × ByteArray)) (member : String) (registered : Array String)
    (pub : Flock.Public) (ports : Array String) : Option (String × Bool) :=
  ports.findSome? fun q =>
    if registered.contains q then none else
    match rootAt s!"{member}/{q}" with
    | none => some (q, false)
    | some (r, d) =>
      if (pub.fv3Bytes "roots" q).toOption == some r && (pub.fv3Bytes "domain_ids" q).toOption == some d then none
      else some (q, true)

/-- **No other port of the statement leaves the registration's roots**: where `statementClash` finds none, every port of
`ports` that is none of `registered` has, in the public header, the frame-v3-sha512 root and domain `rootAt` gives for
`<member>/<port>`. -/
theorem statementClash_none {rootAt : String → Option (ByteArray × ByteArray)} {member : String}
    {registered : Array String} {pub : Flock.Public} {ports : Array String}
    (h : statementClash rootAt member registered pub ports = none) :
    ∀ q ∈ ports, q ∉ registered → ∃ r d, rootAt s!"{member}/{q}" = some (r, d) ∧
      ((pub.fv3Bytes "roots" q).toOption == some r) = true ∧
      ((pub.fv3Bytes "domain_ids" q).toOption == some d) = true := by
  intro q hq hr
  have h1 := Array.findSome?_eq_none_iff.mp h q hq
  have hc : ¬registered.contains q = true := fun ht => hr (Array.contains_iff_mem.mp ht)
  rw [if_neg hc] at h1
  cases hx : rootAt s!"{member}/{q}" with
  | none => rw [hx] at h1; cases h1
  | some rd =>
    obtain ⟨r, d⟩ := rd
    rw [hx] at h1
    dsimp only at h1
    refine ⟨r, d, rfl, ?_⟩
    by_cases he : ((pub.fv3Bytes "roots" q).toOption == some r &&
        (pub.fv3Bytes "domain_ids" q).toOption == some d) = true
    · simpa only [Bool.and_eq_true] using he
    · rw [if_neg he] at h1; cases h1

/-- **A statement's other ports are the window's committed ones** (`statementClash`): every port of the circuit `c` that
no port of the registered reads `own` names has, in the public header `pub`, the frame-v3-sha512 root and domain of the
registration's `<member>/<port>` (`rootAt`), refused by name. `HmOut.checkPublic` holds the header's roots to the
population's rows, and a drawn header keeps the population's, so a writer's statement is proved about the rows its
readers open: O(ports) lookups, the statement's units never read. -/
def statementHeld (rootAt : String → Option (ByteArray × ByteArray)) (member : String)
    (own : Option (Array Flock.Registered.Port)) (c : Flock.Circuit) (pub : Flock.Public) : V Unit :=
  match statementClash rootAt member ((own.getD #[]).map (·.name)) pub (c.ports.map (·.name)) with
  | none => pure ()
  | some (q, false) =>
    throw (s!"registration: the statement's port {q} is no registered port, and the registration commits no root " ++
      s!"for {member}/{q}: a port's rows are the window's committed ones, which its readers open")
  | some (q, true) =>
    throw (s!"registration: the statement's port {q} has another frame-v3-sha512 root or domain in the public file " ++
      s!"than the registration's {member}/{q}: a port's rows are the window's committed ones, which its readers open")

/-- What `statementHeld` accepts: every port of the circuit that no registered port names has the registration's root
and domain for `<member>/<port>` in the public header. -/
theorem statementHeld_ok {rootAt : String → Option (ByteArray × ByteArray)} {member : String}
    {own : Option (Array Flock.Registered.Port)} {c : Flock.Circuit} {pub : Flock.Public}
    (h : statementHeld rootAt member own c pub = .ok ()) :
    ∀ q ∈ c.ports, (∀ o ∈ own.getD #[], o.name ≠ q.name) → ∃ r d, rootAt s!"{member}/{q.name}" = some (r, d) ∧
      ((pub.fv3Bytes "roots" q.name).toOption == some r) = true ∧
      ((pub.fv3Bytes "domain_ids" q.name).toOption == some d) = true := by
  intro q hq hn
  unfold statementHeld at h
  split at h
  · rename_i hc
    refine statementClash_none hc q.name (Array.mem_map_of_mem hq) fun hm => ?_
    obtain ⟨o, ho, he⟩ := Array.mem_map.mp hm
    exact hn o ho he
  · cases h
  · cases h

private def hexT (c : Char) : String := "".pushn c 128

/-- A writer `W/c0`'s output row `o0` and a reader `R/c0`'s input row `x`, registered at roots `wo` and `rx`. -/
private def recT (wo rx : String) (more : Array Json := #[]) : Json :=
  Json.mkObj [("roots", Json.arr (#[
    Json.mkObj [("port", "W/c0/o0"), ("root", .str wo), ("domain", .str (hexT 'd'))],
    Json.mkObj [("port", "R/c0/x"), ("root", .str rx), ("domain", .str (hexT 'e'))]] ++ more))]

private def pubT (port root dom : String) : Flock.Public :=
  { header := Json.mkObj [("frame_v3", Json.mkObj [("roots", Json.mkObj [(port, .str root)]),
      ("domain_ids", Json.mkObj [(port, .str dom)])])], n := 0, digests := #[], outputs := #[], sha := .empty }

private def clashT (rec : Json) (member : String) (pub : Flock.Public) (port : String) (registered : Array String := #[]) :
    Option (String × Bool) :=
  statementClash ((rootsOf rec)[·]?) member registered pub #[port]

#guard clashT (recT (hexT 'a') (hexT 'b')) "W/c0" (pubT "o0" (hexT 'a') (hexT 'd')) "o0" == none
-- the writer: the record's `<writer>/o0` changed, its files kept
#guard clashT (recT (hexT 'c') (hexT 'b')) "W/c0" (pubT "o0" (hexT 'a') (hexT 'd')) "o0" == some ("o0", true)
-- the altered reader: its altered rows' root in its public file, refused, and with the record remade to it, passed
#guard clashT (recT (hexT 'a') (hexT 'b')) "R/c0" (pubT "x" (hexT 'f') (hexT 'e')) "x" == some ("x", true)
#guard clashT (recT (hexT 'a') (hexT 'f')) "R/c0" (pubT "x" (hexT 'f') (hexT 'e')) "x" == none
#guard clashT (recT (hexT 'a') (hexT 'b')) "R/c0" (pubT "x" (hexT 'b') (hexT 'd')) "x" == some ("x", true)
#guard clashT (recT (hexT 'a') (hexT 'b')) "m1" (pubT "x" (hexT 'b') (hexT 'e')) "x" == some ("x", false)
#guard clashT (recT (hexT 'a') (hexT 'b')) "R/c0" (pubT "y" (hexT 'b') (hexT 'e')) "x" == some ("x", true)
#guard clashT (recT (hexT 'a') (hexT 'b')) "m1" (pubT "x" (hexT 'f') (hexT 'e')) "x" #["x"] == none
-- a later entry naming the port wins (R3 refuses a record naming it twice), and one whose root isn't lowercase hex
-- leaves the port no root
private def againT (root : String) : Array Json :=
  #[Json.mkObj [("port", "R/c0/x"), ("root", .str root), ("domain", .str (hexT 'e'))]]

#guard clashT (recT (hexT 'a') (hexT 'b') (againT "AB")) "R/c0" (pubT "x" (hexT 'b') (hexT 'e')) "x" == some ("x", false)
#guard clashT (recT (hexT 'a') (hexT 'f') (againT (hexT 'b'))) "R/c0" (pubT "x" (hexT 'b') (hexT 'e')) "x" == none

end SampledProofs.Registered
