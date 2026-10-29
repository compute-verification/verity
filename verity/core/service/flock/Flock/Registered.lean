/-
PROTOCOL.md §16.10 (registered reads); `protocols/sampled_proofs/README.md`, "Registered values": a statement's rows of a
registered value (a program's weights, constants and tables, committed once at registration under `hm96-sha512/row/v1`
leaves, or a served tree's rows under its port's `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` leaves), opened against
the verifier's own registered roots at the positions its own program reads.
-/
import Flock.Canon
import Flock.HmRow

namespace Flock
namespace Registered

open Lean (Json)

/-- A file's format: a port's `reads` start at unit `reads_from` (`READS_FROM_KEY`, default 0), and an object of the file
holds only its keys (`FILE_KEYS`, `PORT_KEYS`). -/
def FORMAT : String := "verity/registered-reads/v1"
/-- The earlier format, still read: every port's `reads` start at unit 0, and keys beside its own are ignored. -/
def FORMAT_V0 : String := "verity/registered-reads/v0"
/-- A `FORMAT` port's first unit: unit `reads_from + i` reads `reads[i]`, and every other unit reads nothing. -/
def READS_FROM_KEY : String := "reads_from"

def REGISTERED_DOMAIN : String := "verity/one-stage/registered-domain/v0"
/-- The default leaf width, which the domain and the entry name only by leaving `word_bits` out. -/
def WORD_BITS : Nat := 16
def WIDTHS : List Nat := [16, 32]
def WIDTH_KEY : String := "word_bits"
/-- The key of a port whose rows are not `hm96-sha512/row/v1` rows: its leaves' schema, one of `SCHEMAS`. -/
def SCHEMA_KEY : String := "row_schema"
def SCHEMAS : List String := [HmRow.ROW_SCHEMA, HmRow.ROW_SCHEMA_V2, HmRow.ROW_SCHEMA_SEG]

/-- The key of a `--registered` file that names every port of the statement (`checkComplete`); absent, false. -/
def COMPLETE_KEY : String := "complete"
/-- The keys a `FORMAT` file holds. -/
def FILE_KEYS : List String := ["format", "ports", COMPLETE_KEY]
/-- The keys a `FORMAT` file's port holds; `known` is `RegKnown.KNOWN_KEY`, `param` `PARAM_KEY`. -/
def PORT_KEYS : List String :=
  ["value", "root", "leaves", "words", WIDTH_KEY, "domain", SCHEMA_KEY, "reads", READS_FROM_KEY, "known", "param"]

/-- Every key of the object `j` is one of `keys`. -/
def keysIn (j : Json) (keys : List String) : Bool :=
  match j with
  | .obj kvs => kvs.toArray.all fun (k, _) => keys.contains k
  | _ => false

private def tagged (frame : String) (H : ByteArray → ByteArray) (tag : String) (manifest : Json) : V ByteArray := do
  let t := tag.toUTF8
  let m := (← Flock.canon manifest).toUTF8
  return H (frame.toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 m.size.toUInt64 ++ m)

/-- `registered.domain(program, value, words, leaves, word_bits, ...).domain_id` (one-stage `registered.py`), `leaves`
the value's rows: `CommitmentDomain(B, -2, RangeIndexedDomain(0, leaves), hash="sha512").domain_id`, `B` the bytes of
`identity_digest("verity/one-stage/registered-domain/v0", {"program", "value", "schema", "words", "leaves"})` with
`"word_bits"` beside them when it is not 16, and `extra` after them (a per-window value's `"window"`, a produced value's
`"produced"`). -/
def domainWith (program value : String) (words leaves wordBits : Nat) (extra : List (String × Json)) :
    V ByteArray := do
  if !WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  let manifest := Json.mkObj ([("program", Json.str program), ("value", Json.str value), ("schema", Json.str HmRow.ROW_SCHEMA),
    ("words", Lean.toJson words), ("leaves", Lean.toJson leaves)] ++
    (if wordBits == WORD_BITS then [] else [(WIDTH_KEY, Lean.toJson wordBits)]) ++ extra)
  let binding ← tagged "veritor/tagged-sha256/v1" Flock.Sha256.hash REGISTERED_DOMAIN manifest
  let positions ← tagged "verity/tagged-sha512/v1" Flock.Sha512.hash "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson leaves)])
  return FV3.hash "domain" [binding, FV3.uint 0, positions, FV3.uint leaves] Flock.Sha512.hash

/-- A registered value's domain with nothing beside its binding's keys (`domainWith` at `[]`). -/
def domain (program value : String) (words leaves wordBits : Nat) : V ByteArray :=
  domainWith program value words leaves wordBits []

/-- A port of the verifier's own `--registered` file: the value its rows are rows of, the value's registered root, leaf
count, row width and frame-v3-sha512 domain, the position every unit of the population reads (none: a unit of
another template), and its rows' leaf schema. A row is `words` leaves of `wordBits` bits (16 or 32); a file that names
neither (`words` none, `wordBits` 16) has rows of 16-bit words, the port's own. A `hm96-sha512/row/v1` row is laid out
in whole SHA-512 blocks (`rowWords`); a `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` row is a bit row of exactly
`words · wordBits` bits, as the circuit's port of that schema is. The reads are `readsRel` from unit `readsFrom` on: unit
`readsFrom + i` reads `readsRel[i]`, and every other unit reads nothing (`readAt`; `reads`, the population-long array). -/
structure Port where
  name : String
  value : String
  root : ByteArray
  leaves : Nat
  words : Option Nat
  wordBits : Nat
  domain : ByteArray
  readsRel : Array (Option Nat)
  schema : String := HmRow.ROW_SCHEMA
  readsFrom : Nat := 0
  deriving Inhabited

/-- The position every unit reads, from unit 0: none for each unit below `readsFrom`, then `readsRel`. The verifier
reads one unit's by `readAt`, which builds no array; the proofs read it by index (`readAt_eq`). -/
def Port.reads (p : Port) : Array (Option Nat) :=
  if p.readsFrom = 0 then p.readsRel else Array.replicate p.readsFrom none ++ p.readsRel

@[simp] theorem Port.reads_of_readsFrom {p : Port} (h : p.readsFrom = 0) : p.reads = p.readsRel := by
  simp [Port.reads, h]

/-- Unit `u`'s read, `reads[u]?` (`readAt_eq`) in O(1): `some none` below `readsFrom`, then `readsRel`'s. -/
def Port.readAt (p : Port) (u : Nat) : Option (Option Nat) :=
  if u < p.readsFrom then some none else p.readsRel[u - p.readsFrom]?

theorem Port.readAt_eq (p : Port) (u : Nat) : p.readAt u = p.reads[u]? := by
  unfold Port.readAt Port.reads
  by_cases h0 : p.readsFrom = 0
  · simp [h0]
  · rw [if_neg h0]
    by_cases hu : u < p.readsFrom
    · rw [if_pos hu, Array.getElem?_append_left (by simpa using hu)]
      simp [hu]
    · rw [if_neg hu, Array.getElem?_append_right (by simp; omega)]
      simp

def ofJson (j : Json) : V (Array Port) := do
  let ctx := "registered reads"
  let format ← J.str j "format" ctx
  if format != FORMAT && format != FORMAT_V0 then throw s!"{ctx}: not a {FORMAT} file (nor a {FORMAT_V0} one)"
  let v1 := format == FORMAT
  if v1 && !keysIn j FILE_KEYS then throw s!"{ctx}: a {FORMAT} file holds the keys {FILE_KEYS} and no other"
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM fun (name, e) => do
    if v1 && !keysIn e PORT_KEYS then throw s!"{ctx}: port {name}: a {FORMAT} port holds keys of {PORT_KEYS} and no other"
    if !v1 && J.has e READS_FROM_KEY then
      throw s!"{ctx}: port {name}: {READS_FROM_KEY} is a {FORMAT} key, and a {FORMAT_V0} file's reads start at unit 0"
    let readsFrom ← if J.has e READS_FROM_KEY then J.nat e READS_FROM_KEY ctx else pure 0
    let root ← unhex (← J.str e "root" ctx)
    let domain ← unhex (← J.str e "domain" ctx)
    let leaves ← J.nat e "leaves" ctx
    if root.size != 64 || domain.size != 64 || leaves == 0 then
      throw s!"{ctx}: port {name}: a 64-byte root and domain, and at least one leaf"
    let wordBits ← if J.has e "word_bits" then J.nat e "word_bits" ctx else pure 16
    let words ← if J.has e "words" then some <$> J.nat e "words" ctx else pure none
    if !(wordBits == 16 || wordBits == 32) || words == some 0 || (words.isNone && wordBits != 16) then
      throw s!"{ctx}: port {name}: leaves of 16 or 32 bits, at least one to a row, and a row's words when they are 32"
    let schema ← if J.has e SCHEMA_KEY then J.str e SCHEMA_KEY ctx else pure HmRow.ROW_SCHEMA
    if !SCHEMAS.contains schema then throw s!"{ctx}: port {name}: {SCHEMA_KEY} is one of {SCHEMAS}, not {schema}"
    if schema != HmRow.ROW_SCHEMA && words.isNone then
      throw s!"{ctx}: port {name}: a {schema} row is a bit row of its words' bits, and the port names no words"
    let reads ← (← J.arr e "reads" ctx).mapM fun r => match r with
      | .null => pure none
      | r => some <$> J.asNat r s!"{ctx}: a read position"
    return { name, value := ← J.str e "value" ctx, root, leaves, words, wordBits, domain, readsRel := reads, schema, readsFrom }

/-- The u16 words of a registered row of `words` leaves of `wordBits` bits, as the statement lays out the port's row:
each leaf `wordBits / 16` words, low first, zero-padded to whole SHA-512 blocks of 64 words. -/
def rowWords (words wordBits : Nat) : Nat := (words * (wordBits / 16) + 63) / 64 * 64

/-- The leaf schema of the circuit's port `p`, as the plain verifier frames its rows (`HmRow.checkPublic`). -/
def portSchema (p : Flock.Port) : String :=
  if p.segs.isSome then HmRow.ROW_SCHEMA_SEG else if p.v2 then HmRow.ROW_SCHEMA_V2 else HmRow.ROW_SCHEMA

/-- The circuit's port `p`'s rows are `reg`'s registered rows: a `hm96-sha512/row/v1` port's rows of u16 words are the
registered rows in whole SHA-512 blocks (`rowWords`), and a bit row's bits are exactly the registered row's
`words · wordBits`. -/
def Port.fits (reg : Port) (p : Flock.Port) : Bool :=
  if reg.schema == HmRow.ROW_SCHEMA then reg.words.all fun w => p.words == rowWords w reg.wordBits
  else reg.words.any fun w => p.bits == w * reg.wordBits

export FV3.Sha512 (node climbL sibs climb leafAs)

/-- The frame-v3-sha512 leaf at `pos` of a registered value's row whose `hm96-sha512/row/v1` leaf is `payload`. -/
def rowLeaf (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  FV3.leaf dom pos HmRow.ROW_SCHEMA payload Sha512.hash

/-- A row whose hm96 leaf of `reg`'s schema is `payload` opens `reg`'s registered root at `pos` through `path`: a
position among its leaves, one 64-byte sibling per level, and the climb ends at the root. -/
def opensLeaf (reg : Port) (payload : ByteArray) (pos : Nat) (path : ByteArray) : Bool :=
  decide (pos < reg.leaves) && decide (path.size = 64 * log2ceil reg.leaves) &&
    decide (climb reg.domain (leafAs reg.schema reg.domain pos payload) pos (log2ceil reg.leaves) path = reg.root)

/-- A table row's `b ‖ c` opens `reg`'s registered root at `pos` through `path`. -/
def opens (reg : Port) (bc : ByteArray) (pos : Nat) (path : ByteArray) : Bool :=
  opensLeaf reg (Hm96.Default512.treeLeaf bc) pos path

/-- What a verdict reports: each registered port's value and the root its reads were opened against. -/
def finding (own : Array Port) : Json :=
  Json.mkObj (own.toList.map fun p => (p.name, Json.mkObj [("value", Json.str p.value), ("root", Json.str (hexOf p.root))]))

/-- The statement check derived the statement's units from the verifier's own program (its partition finding,
`HmRow.PartitionFinding`), rather than taking them as stated. -/
def derived (partition : Json) : Bool := (partition.getObjValAs? String "units").toOption == some "derived"

/-- The unit each of the public file's instances is (its header's `units.indices`). -/
def unitIndices (pub : Public) : V (Array Nat) := do
  (← J.arr (← J.get pub.header "units" "public file") "indices" "public file").mapM (J.asNat · "a unit index")

/-- Port `reg`'s entry in the header's `registered`: the position and the path of each row of its table. -/
def entry (pub : Public) (reg : Port) : V (Array Nat × Array ByteArray) := do
  let ctx := "public file.registered"
  let .ok e := pub.header.getObjVal? "registered" >>= (·.getObjVal? reg.name)
    | throw s!"registered reads: the statement commits port {reg.name}'s rows of {reg.value} afresh, with no path to its registered root"
  if (← J.str e "value" ctx) != reg.value then throw s!"registered reads: port {reg.name} reads {reg.value}, not another value"
  let positions ← (← J.arr e "positions" ctx).mapM (J.asNat · "a registered position")
  let paths ← (← J.arr e "paths" ctx).mapM fun x => do unhex (← J.asStr x "a registered path")
  return (positions, paths)

/-- Row `r` of port `p`'s table: its `b ‖ c` among the file's row digests (`Public.digest` reads the same). -/
def tableRow (pub : Public) (t : Tables) (p r : Nat) : ByteArray := pub.digests.getD (t.starts.getD p 0 + r) ByteArray.empty

/-- Instance `i`'s ref into port `p`'s table, among `np` ports. -/
def ref (t : Tables) (np p i : Nat) : Nat := t.refs.getD (i * np + p) 0

/-- The first of port `p`'s `rows` table rows that does not open `reg`'s registered root at its position through its path. -/
def badRow (reg : Port) (pub : Public) (t : Tables) (p rows : Nat) (positions : Array Nat) (paths : Array ByteArray) :
    Option Nat :=
  (List.range rows).find? fun r => !opens reg (tableRow pub t p r) (positions.getD r 0) (paths.getD r ByteArray.empty)

/-- The first of `n` instances whose ref names no row of port `p`'s table, or whose row's position is not the one the
verifier's program reads for the instance's unit. -/
def badInstance (reg : Port) (t : Tables) (np p rows n : Nat) (positions indices : Array Nat) : Option Nat :=
  (List.range n).find? fun i =>
    !(decide (ref t np p i < rows) && decide (reg.readAt (indices.getD i 0) = some (some (positions.getD (ref t np p i) 0))))

/-- One port of the verifier's registered reads: a row of the circuit (an input port's, or a hidden-output statement's
output row, §16.13, whose table is its instances' own rows, instance `i`'s ref `i`), whose leaf schema is the
registration's (`portSchema`), whose rows are the registered rows (`Port.fits`), whose table rows all open the
registered root, and whose instances each read the row at the position the verifier's program reads for their unit. -/
def checkPort (c : Circuit) (pub : Public) (t : Tables) (indices : Array Nat) (reg : Port) : V Unit := do
  let np := c.ports.size
  let some p := (List.range np).find? (fun p => c.ports[p]!.name == reg.name)
    | throw s!"registered reads: {reg.name} is not a row of the circuit (an input port, or an output port's row)"
  if portSchema c.ports[p]! != reg.schema then
    throw s!"registered reads: port {reg.name} is a {portSchema c.ports[p]!} row, and {reg.value}'s registered rows are {reg.schema} rows"
  if !reg.fits c.ports[p]! then
    throw s!"registered reads: port {reg.name}'s rows are {c.ports[p]!.bits} bits ({c.ports[p]!.words} u16 words), not {reg.value}'s registered rows of {reg.words.getD 0} {reg.wordBits}-bit leaves"
  let (positions, paths) ← entry pub reg
  let rows := t.counts.getD p 0
  if positions.size != rows || paths.size != rows then
    throw s!"registered reads: port {reg.name}: one position and one path per table row"
  match badRow reg pub t p rows positions paths with
  | some r =>
    throw s!"registered reads: port {reg.name}: table row {r} does not open {reg.value}'s registered root at position {positions.getD r 0}"
  | none =>
    match badInstance reg t np p rows pub.n positions indices with
    | none => pure ()
    | some i =>
      let u := indices.getD i 0
      if ref t np p i ≥ rows then throw s!"registered reads: port {reg.name}: instance {i}'s ref names no row of its table"
      match reg.readAt u with
      | some (some want) =>
        throw s!"registered reads: port {reg.name}: unit {u} reads position {positions.getD (ref t np p i) 0}, and the verifier's program reads {want}"
      | _ => throw s!"registered reads: port {reg.name}: the verifier's program reads no row of {reg.value} in unit {u}"

/-- The statement's registered reads against the verifier's own `own` (`--registered`). The header's `registered` must
name exactly `own`'s ports, each a port of the circuit and a table of a shared-row file: every table row's `b ‖ c`, as the
value's row at its stated position, opens the registered root through its path, and every instance reads, by its ref,
the position the verifier's program reads for its unit (`checkPort`; `FlockProofs.Registered`: `check_ok`). Without
`own`, a statement that names registered reads is refused: its rows would be checked against no root. With `own`, the
units must be `derived`: a unit's position is the verifier's, but which unit an instance is comes from its label, which
binds nothing unless the verifier derived it. -/
def check (c : Circuit) (pub : Public) (own : Option (Array Port)) (derived : Bool) : V Unit := do
  let hdr := pub.header.getObjVal? "registered"
  let some own := own
    | if hdr.isOk then throw "registered reads: the statement reads registered values, and the verifier holds no registered roots (--registered)"
  if !derived then
    throw "registered reads: the statement's units are taken as stated, and a read's position binds only as far as its unit does: the verifier derives them from its own program (--partition, --program)"
  let .ok (.obj kvs) := hdr
    | throw "registered reads: the statement names no registered rows: a registered value's rows committed afresh open no registered root"
  for (name, _) in kvs.toArray do
    if !own.any (·.name == name) then throw s!"registered reads: the statement names port {name} registered, and the verifier's registered reads do not"
  let some t := pub.tables | throw "registered reads: a registered port is a table of a shared-row file"
  let indices ← unitIndices pub
  own.toList.forM (checkPort c pub t indices)

/-- The callees of the verifier's own program bound at registration: each `RegisteredInput<w>_v1`
(`Program.inputId?`'s `registration`) among its descriptor's callees, which every node of every Definition names its
callee in (`TemplateQuery.Desc`, the descriptor as the template queries read it). A node reads a registered input exactly
when its callee is one of these, wherever it sits. -/
def programInputs (d : TemplateQuery.Desc) : Array String :=
  d.callees.filter fun id => (Program.inputId? id).map (·.1) == some Program.REGISTRATION

/-- **A program that reads a registered value is verified only against registered roots** (PROTOCOL.md §16.10): with the
verifier's own program (`program`, `--program`), a statement whose program has a `RegisteredInput<w>_v1` callee
(`programInputs`) is refused unless the verifier holds its own registered reads (`registered`, `--registered`), whatever
the statement's files say. The public file's header `registered` is the prover's, and the circuit marks no port
registered, so neither can tell the verifier its rows of a registered value must open a registered root: a statement that
names none would commit them afresh. With the registered reads, `check` opens every port of them, header or not. A
program that does not parse is refused. -/
def checkProgram (program : Option ByteArray) (registered : Bool) : V Unit :=
  match program, registered with
  | none, _ => .ok ()
  | some _, true => .ok ()
  | some b, false =>
    match TemplateQuery.Desc.parse b with
    | .error e => .error e
    | .ok d =>
      match (programInputs d)[0]? with
      | none => .ok ()
      | some id =>
        .error s!"registered reads: the verifier's program reads a registered value ({id}), and the verifier holds no registered roots (--registered): its rows would be committed afresh, whatever the statement's public file names"

/-- `checkProgram` accepts the verifier's program `b` only with its registered reads, or when `b` parses to a program
with no registered input. -/
theorem checkProgram_ok {b : ByteArray} {registered : Bool} (h : checkProgram (some b) registered = .ok ()) :
    registered = true ∨ ∃ d, TemplateQuery.Desc.parse b = .ok d ∧ programInputs d = #[] := by
  cases registered with
  | true => exact .inl rfl
  | false =>
    refine .inr ?_
    simp only [checkProgram] at h
    split at h
    · cases h
    · next P hd =>
      refine ⟨P, hd, ?_⟩
      split at h
      · next hn =>
        have : (programInputs P).size ≤ 0 := Array.getElem?_eq_none_iff.mp hn
        exact Array.eq_empty_of_size_eq_zero (by omega)
      · cases h

/-- The file's `complete` (`COMPLETE_KEY`), false when absent. -/
def completeOf (j : Json) : V Bool :=
  match j.getObjVal? COMPLETE_KEY with
  | .error _ => .ok false
  | .ok (.bool b) => .ok b
  | .ok _ => .error s!"registered reads: {COMPLETE_KEY} is true or false"

/-- Every port of the circuit, each input port and each output row, is a port of `own`. -/
def complete (c : Circuit) (own : Array Port) : Bool := c.ports.all fun p => own.any (·.name == p.name)

/-- **Every port a statement's outputs read is registered** (PROTOCOL.md §16.10, a `--registered` file that sets
`complete`): every port of the circuit is a port of the verifier's own registered reads (`checkComplete_ok`), so no
output's cone reads a row the statement commits afresh and its public file states (a lane's index, say), and `check`
opens each of them at its registered root. It reads the circuit's ports, not its gates: whatever an output's cone reads
is a port's row. -/
def checkComplete (c : Circuit) (own : Array Port) : V Unit :=
  if complete c own then .ok ()
  else
    let p := ((c.ports.find? fun p => !own.any (·.name == p.name)).map (·.name)).getD ""
    .error s!"registered reads (complete): the statement's port {p} is no port of the verifier's registered reads, which name every port the statement's outputs read: its rows would be committed afresh, stated by the public file"

theorem checkComplete_ok {c : Circuit} {own : Array Port} (h : checkComplete c own = .ok ()) :
    ∀ p ∈ c.ports, ∃ r ∈ own, r.name = p.name := by
  unfold checkComplete at h
  split at h
  · rename_i hc
    intro p hp
    obtain ⟨r, hr, he⟩ := Array.any_eq_true'.mp (Array.all_eq_true_iff_forall_mem.mp hc p hp)
    exact ⟨r, hr, by simpa using he⟩
  · cases h

/-! ### Every registered input the statement's template reads has a port -/

/-- The key of a `FORMAT` port that names the template parameter it reads (`registered.reads`' key: the parameter's name,
or `name[i]` for row `i` of an array of rows), when the port is not named after it (`reads_file`'s `ports`). -/
def PARAM_KEY : String := "param"

/-- The template parameter each port of a `--registered` file reads, by `PARAM_KEY`, else the port's own name. -/
def paramsOf (j : Json) : V (Array String) := do
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw "registered reads: ports is not an object"
  kvs.toArray.mapM fun (name, e) => if J.has e PARAM_KEY then J.str e PARAM_KEY "registered reads" else pure name

/-- Definition `fn` of `P` is an input bound at registration (`RegisteredInput<w>_v1`). -/
def regDef (P : Program.Prog) (fn : Nat) : Bool :=
  match P.defs[fn]? with
  | some f => f.prim && f.binding == some Program.REGISTRATION
  | none => false

/-- The root's nodes of `P`. -/
def rootNodes (P : Program.Prog) : Array Program.Node := ((P.defs[P.root]?).map (·.body.nodes)).getD #[]

/-- Root node `k` of `P` is a registered input. -/
def regNode (P : Program.Prog) (k : Nat) : Bool := ((rootNodes P)[k]?).any (regDef P ·.fn)

/-- The root nodes a reference part reads (`n` space; the root has no parameters). -/
def partNodes : Program.Part → Array Nat
  | .affine p i _ _ => if p then #[] else #[i]
  | .strided p i _ _ _ _ _ => if p then #[] else #[i]
  | .explicit it => (it.filter (!·.p)).map (·.i)

/-- A root node's argument reads a registered input: a part of it reads a `RegisteredInput<w>_v1` root node. -/
def argReads (P : Program.Prog) (a : Program.Arg) : Bool := a.seq.parts.any fun q => (partNodes q).any (regNode P)

/-- **The template parameters that read a registered input** (PROTOCOL.md §16.10), from the verifier's own program: for
each root node that calls `t` (a template query's instance of it), each parameter of `t` whose argument reads a
`RegisteredInput<w>_v1` root node, with its type. Input nodes appear only in the root's body (`Program.decode`), so a
template instance reads a registered value exactly through these arguments. It reads the root's nodes and their argument
parts, the descriptor, never the unrolled circuit. -/
def regParams (P : Program.Prog) (t : String) : Array (String × Program.Ty) :=
  ((rootNodes P).filter fun x => (P.defs[x.fn]?).any (·.key == t)).flatMap fun x =>
    ((Array.range x.args.size).filter fun i => (x.args[i]?).any (argReads P)).filterMap fun i =>
      (P.defs[x.fn]?).bind (·.params[i]?)

/-- The program has a registered input: a `RegisteredInput<w>_v1` root node. -/
def regRoots (P : Program.Prog) : Bool := (rootNodes P).any (regDef P ·.fn)

/-- Parameter `p` is read by a port whose parameter (`paramsOf`) is `params`' : one port of the whole parameter, or, for
an array, one port of each row (`name[i]`), as `registered.reads` keys an array of whole rows. -/
def coversParam (params : Array String) (p : String × Program.Ty) : Bool :=
  params.contains p.1 || match p.2 with
    | .array n _ => n > 0 && (List.range n).all fun i => params.contains s!"{p.1}[{i}]"
    | _ => false

/-- The first template parameter that reads a registered input and that no port's parameter covers. -/
def coverMiss (P : Program.Prog) (t : String) (params : Array String) : Option (String × Program.Ty) :=
  (regParams P t).find? fun p => !coversParam params p

/-- The verifier's own partition object's query is a template query (`TemplateQuery.templatesOf`), whose instances are
the program's root nodes. -/
def templatePartition (partition : Option Json) : Bool :=
  match partition with
  | none => false
  | some o =>
    let q := (o.getObjVal? "query").toOption.getD Json.null
    (TemplateQuery.templatesOf ((q.getObjValAs? String "name").toOption.getD "")
      ((q.getObjValAs? Nat "version").toOption.getD 0) ((q.getObjVal? "params").toOption.getD Json.null)).isSome

/-- **Every registered input the statement reads has a port** (PROTOCOL.md §16.10): with the verifier's own program and
registered reads, a program with a `RegisteredInput<w>_v1` root node is verified only under a template query (whose units
are the root's instances of the circuit's template, `HmRow.queryTemplate`), and every parameter of that template that
reads one (`regParams`) is read by a port of the verifier's registered reads (`coversParam`, by the ports' `params`,
`paramsOf`), whose rows `check` opens at the registered root. Without it, a parameter the registered reads leave out would
have its rows committed afresh, whatever the reads hold for the others. Without the program or the registered reads it
holds nothing: `checkProgram` refuses a program that reads a registered value without the reads, and `verify` refuses a
session without the program. A program that does not decode is refused. -/
def checkCover (c : Circuit) (program : Option ByteArray) (partition : Option Json) (own : Option (Array Port))
    (params : Array String) : V Unit :=
  match program, own with
  | some b, some _ =>
    match Program.decode b with
    | .error e => .error e
    | .ok P =>
      if !regRoots P then .ok ()
      else match templatePartition partition, HmRow.queryTemplate c with
        | false, _ => .error "registered reads (cover): the verifier's program reads a registered value, and its partition is not a template query: a registered value is read by a template query's instances, whose parameters the verifier holds to its registered reads"
        | true, none => .error "registered reads (cover): META program_digests does not name the circuit's one template, whose parameters the verifier holds to its registered reads"
        | true, some (t, _) =>
          match coverMiss P t params with
          | none => .ok ()
          | some (name, _) => .error s!"registered reads (cover): parameter {name} of {t} reads a registered input of the verifier's program, and no port of the verifier's registered reads reads it: its rows would be committed afresh"
  | _, _ => .ok ()

theorem coverMiss_none {P : Program.Prog} {t : String} {params : Array String} (h : coverMiss P t params = none) :
    ∀ p ∈ regParams P t, coversParam params p = true := by
  intro p hp
  have := Array.find?_eq_none.mp h p hp
  simpa using this

/-- **An accepted cover check covers every registered input the template reads**: where `checkCover` accepts the
verifier's program `b` (decoding to `P`) and registered reads, and `P` has a registered input, the partition is a template
query, the circuit's template is some `t`, and every parameter of `t` that reads a registered input of `P` is covered by a
port's parameter. -/
theorem checkCover_ok {c : Circuit} {b : ByteArray} {partition : Option Json} {own : Array Port} {params : Array String}
    {P : Program.Prog} (h : checkCover c (some b) partition (some own) params = .ok ()) (hd : Program.decode b = .ok P)
    (hr : regRoots P = true) :
    templatePartition partition = true ∧ ∃ t x, HmRow.queryTemplate c = some (t, x) ∧
      ∀ p ∈ regParams P t, coversParam params p = true := by
  revert h
  simp only [checkCover, hd, hr, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  cases hp : templatePartition partition with
  | false => intro h; simp at h
  | true =>
    cases hq : HmRow.queryTemplate c with
    | none => intro h; simp at h
    | some tx =>
      obtain ⟨t, x⟩ := tx
      intro h
      refine ⟨rfl, t, x, rfl, coverMiss_none ?_⟩
      cases hm : coverMiss P t params with
      | none => rfl
      | some y =>
        obtain ⟨n, ty⟩ := y
        simp [hm] at h

/-- The registered values that are a round's commitment by an index, `<prefix><i>`: round `i`'s commitment `rec-c<i>`,
which V*'s algebra reads; level `l`'s committed tops `rec-top-L<l>`, which its opening statement reads; R's registered
opening's round `i`, `rec-opening-c<i>`, and R's level-`l` tops, `rec-top-stacked-L<l>`; and the sparse phase's messages
`rec-sparse-pre<j>` and `rec-sparse-sc<j>`. -/
def roundIndexed : List String :=
  ["rec-c", "rec-top-L", "rec-opening-c", "rec-top-stacked-L", "rec-sparse-pre", "rec-sparse-sc"]

/-- The sparse phase's other messages, each a round's commitment by its name. -/
def roundNamed : List String := ["rec-sparse-v2", "rec-sparse-u", "rec-sparse-vt"]

/-- A registered value a recursive statement reads a round's commitment from: a prefix of `roundIndexed` followed by an
index's decimal digits, or one of `roundNamed`. The sparse phase's `rec-sparse-v`, `rec-sparse-zero` and `rec-sparse-tau`
are no round's commitment. -/
def roundCommitment (value : String) : Bool :=
  let index (s : String) := !s.isEmpty && s.all Char.isDigit
  roundNamed.contains value || roundIndexed.any fun pre => value.startsWith pre && index (value.drop pre.length).toString

/-- Whether every draw by `d`'s law draws unit `u`: a subset draw of the whole population, a `p = 1` draw, or a unit whose
stratum draws all of its units (`k_s = n_s`). -/
def drawsInFull (d : Draw.UnitDraw) (u : Nat) : Bool :=
  let full (s : Draw.Stratum) := s.units.any (fun (lo, hi) => lo ≤ u && u < hi) && s.k == s.size
  match d.law with
  | .subset k => k == d.population
  | .bernoulli a b => a == b
  | .stratified _ strata => strata.any full
  | .work _ strata => strata.any (full ·.toStratum)

/-- The first port of `own` and unit that reads a round's commitment (`roundCommitment`) by `own`'s reads, and that `d`
does not draw in full. -/
def fullMiss (own : Array Port) (d : Draw.UnitDraw) : Option (Port × Nat) :=
  own.findSome? fun p =>
    if roundCommitment p.value then
      ((List.range p.reads.size).find? fun u => p.reads[u]!.isSome && !drawsInFull d u).map (p, ·)
    else none

/-- U4 (PROTOCOL.md §16.10): a drawn statement draws in full every unit that reads a round's commitment by the verifier's
own reads `own` (`fullMiss`), so no stage's message is read off a sample. `draw` is the record's; a statement without a
draw proves its whole population. -/
def checkFull (own : Option (Array Port)) (draw : Option Json) : V Unit := do
  let (some own, some j) := (own, draw) | pure ()
  let d ← Draw.UnitDraw.ofJson j
  match fullMiss own d with
  | none => pure ()
  | some (p, u) =>
    throw s!"U4: unit {u} reads {p.value}, a round's commitment (port {p.name}), and the draw does not draw it in full"

/-! ### A recursive session's inner statement: its description, read from the developer's registration (§16.16) -/

/-- The circuit's first port named `name`. -/
def portOf (c : Circuit) (name : String) : Option Nat := (List.range c.ports.size).find? fun p => c.ports[p]!.name == name

/-- The statement's commit strings of `reg`'s registered rows: registered row `r` is row `r` of the table of the circuit's
port named `reg.name` (`checkDescription`: the table is the registration, in order). -/
def rowsOf (c : Circuit) (pub : Public) (reg : Port) (r : Nat) : ByteArray :=
  match portOf c reg.name, pub.tables with
  | some p, some t => tableRow pub t p r
  | _, _ => ByteArray.empty

/-- The path the statement's header gives registered row `r` of `reg` (`entry`), the one `rowsOf c pub reg r` opens the
registered root through. -/
def pathOf (pub : Public) (reg : Port) (r : Nat) : ByteArray :=
  match entry pub reg with
  | .ok (_, paths) => paths.getD r ByteArray.empty
  | .error _ => ByteArray.empty

/-- The first of `rows` table rows whose stated position is not its own index. -/
def badPosition (positions : Array Nat) (rows : Nat) : Option Nat :=
  (List.range rows).find? fun r => positions.getD r 0 != r

/-- The first of `n` instances whose ref names no row of port `p`'s `rows` table rows. -/
def badRef (t : Tables) (np p rows n : Nat) : Option Nat :=
  (List.range n).find? fun i => !decide (ref t np p i < rows)

/-- The first of port `p`'s `rows` table rows that none of `n` instances reads. -/
def unread (t : Tables) (np p rows n : Nat) : Option Nat :=
  (List.range rows).find? fun r => !(List.range n).any fun i => ref t np p i == r

/-- One port of an inner statement's description: a row of the circuit, of `reg`'s leaf schema (`portSchema`) and
width (`Port.fits`), whose table is `reg`'s registration in order (as many rows as its leaves, table row `r` at position
`r`, opening the registered root through its path), every instance reading a row of it, and every row of it read. -/
def checkDescriptionPort (c : Circuit) (pub : Public) (t : Tables) (reg : Port) : V Unit := do
  let np := c.ports.size
  let some p := portOf c reg.name
    | throw s!"registered reads: {reg.name} is not a row of the circuit (an input port, or an output port's row)"
  if portSchema c.ports[p]! != reg.schema then
    throw s!"registered reads: port {reg.name} is a {portSchema c.ports[p]!} row, and {reg.value}'s registered rows are {reg.schema} rows"
  if !reg.fits c.ports[p]! then
    throw s!"registered reads: port {reg.name}'s rows are {c.ports[p]!.bits} bits ({c.ports[p]!.words} u16 words), not {reg.value}'s registered rows of {reg.words.getD 0} {reg.wordBits}-bit leaves"
  let (positions, paths) ← entry pub reg
  let rows := t.counts.getD p 0
  if rows != reg.leaves then
    throw s!"registered reads: port {reg.name}: its table has {rows} rows, and {reg.value}'s registration {reg.leaves}: an inner statement's description is its registered rows, each once, in order"
  else if positions.size != rows || paths.size != rows then
    throw s!"registered reads: port {reg.name}: one position and one path per table row"
  else match badPosition positions rows with
  | some r =>
    throw s!"registered reads: port {reg.name}: table row {r} states position {positions.getD r 0}, and an inner statement's description table is its registration in order (row {r} at position {r})"
  | none =>
    match badRow reg pub t p rows positions paths with
    | some r =>
      throw s!"registered reads: port {reg.name}: table row {r} does not open {reg.value}'s registered root at position {r}"
    | none =>
      match badRef t np p rows pub.n with
      | some i =>
        throw s!"registered reads: port {reg.name}: instance {i}'s ref {ref t np p i} names no row of its {rows}-row table"
      | none =>
        match unread t np p rows pub.n with
        | some r =>
          throw s!"registered reads: port {reg.name}: none of the {pub.n} instances reads table row {r}, registered row {r} of {reg.value}'s {reg.leaves}: an inner statement reads its whole description"
        | none => pure ()

/-- **A recursive session's inner statement's description, read from the developer's registration** (PROTOCOL.md §16.16):
the header's `registered` names exactly the verifier's own ports (`own`, `--registered`), each a table of a shared-row file
that is the registration in order, every row of it opening the registered root at its index and read by an instance
(`checkDescriptionPort`; `FlockProofs.Registered`: `checkDescription_ok`). No position depends on which unit an instance
is, so unlike `check` the units need not be derived. -/
def checkDescription (c : Circuit) (pub : Public) (own : Array Port) : V Unit := do
  let .ok (.obj kvs) := pub.header.getObjVal? "registered"
    | throw "registered reads: the inner statement names no registered rows: its description committed afresh opens no registered root"
  for (name, _) in kvs.toArray do
    if !own.any (·.name == name) then throw s!"registered reads: the statement names port {name} registered, and the verifier's registered reads do not"
  let some t := pub.tables | throw "registered reads: a registered port is a table of a shared-row file"
  own.toList.forM (checkDescriptionPort c pub t)

/-- The first of the circuit's ports whose name an earlier port has. -/
def dupName (c : Circuit) : Option Nat :=
  (List.range c.ports.size).find? fun q => (List.range q).any fun q' => c.ports[q']!.name == c.ports[q]!.name

/-- **With registered reads, the circuit names each port once**: the verifier's registered reads, and every check of them,
name a port by its name, so a second port of a name is one no check reads. -/
def namesOnce (c : Circuit) : V Unit :=
  match dupName c with
  | some q => throw s!"registered reads: the circuit names port {c.ports[q]!.name} twice, and a registered read names one port"
  | none => pure ()

end Registered

/-- An hm96 statement's registered reads (`Registered.check`), and with the verifier's own, a circuit that names each port
once (`Registered.namesOnce`). A hidden-output statement's registered ports may be its output rows as well as its input
rows (§16.13): an output row's table is its instances' own rows, each opened against the registered root at the position
its unit reads, so the instance's output is that row of the registered value. -/
def checkRegistered (st : Stmt) (registered : Option (Array Registered.Port)) : V Unit :=
  match registered with
  | some _ => Registered.namesOnce st.c >>= fun _ => Registered.check st.c st.pub registered (Registered.derived st.partition)
  | none => Registered.check st.c st.pub none (Registered.derived st.partition)

/-- What a verdict reports of the rows its statements' registered ports opened: `{port: [[position, b ‖ c] ...]}`, each
table row's stated position and its hiding commitment `b ‖ c` (the commit-time opening's digest), in table order across
the statements; null without registered reads. A produced record holds every session's to its own
(`SampledProofs.Produced.shared`). -/
def registeredOpenings (sts : Array Stmt) (registered : Option (Array Registered.Port)) : Lean.Json :=
  match registered with
  | none => Lean.Json.null
  | some own => Lean.Json.mkObj (own.toList.map fun reg =>
      (reg.name, Lean.Json.arr (sts.foldl (fun acc st => acc ++ rowsOf st reg) #[])))
where
  rowsOf (st : Stmt) (reg : Registered.Port) : Array Lean.Json :=
    match Registered.portOf st.c reg.name, st.pub.tables, Registered.entry st.pub reg with
    | some p, some t, .ok (positions, _) =>
      (Array.range (t.counts.getD p 0)).map fun r =>
        Lean.Json.arr #[Lean.toJson (positions.getD r 0), Lean.Json.str (hexOf (Registered.tableRow st.pub t p r))]
    | _, _, _ => #[]

end Flock
