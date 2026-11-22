/-
`Flock.HmRow` (registered reads); `protocols/spot/README.md`, "Registered values": a statement's rows of a
registered value (a program's weights, constants and tables, committed once at registration under `hm96-sha512/row/v1`
leaves, or a served tree's rows under its port's `hm96-sha512/row/v2` or `hm96-sha512/row-seg/v1` leaves), opened against
the verifier's own registered roots at the positions its own program reads.
-/
module

public import Circuits.Lanes
public import Flock.Canon
public import Flock.HmRow

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Flock
namespace Registered

open Lean (Json)

/-- A file's format: theta's registration, each port a registered value's root, domain, leaves and row width, and no
position: every position a unit reads is derived from C and the partition (`OneCircuit.pos`). An object of the
file holds only its keys (`FILE_KEYS`, `PORT_KEYS`); a file of the earlier format, which stated positions, is refused
by its format's name. -/
def FORMAT : String := "verity/registered-reads/v2"

def REGISTERED_DOMAIN : String := "verity/one-stage/registered-domain/v1"
/-- The default leaf width, which the domain and the entry name only by leaving `word_bits` out. -/
def WORD_BITS : Nat := 16
def WIDTHS : List Nat := [16, 32]
def WIDTH_KEY : String := "word_bits"
/-- The key of a port whose rows are not `hm96-sha512/row/v1` rows: its leaves' schema, one of `SCHEMAS`. -/
def SCHEMA_KEY : String := "row_schema"
def SCHEMAS : List String := [HmRow.ROW_SCHEMA, HmRow.ROW_SCHEMA_V2, HmRow.ROW_SCHEMA_SEG]

/-- The keys a `FORMAT` file holds. -/
def FILE_KEYS : List String := ["format", "ports"]
/-- The keys a `FORMAT` file's port holds; `known` is `RegKnown.KNOWN_KEY`, `param` `PARAM_KEY`. No key says which tree
a row of in(S) reads: C does (`OneCircuit.io`, `treeReads`). -/
def PORT_KEYS : List String :=
  ["value", "root", "leaves", "words", WIDTH_KEY, "domain", SCHEMA_KEY, "known", "param"]

/-- Every key of the object `j` is one of `keys`. -/
def keysIn (j : Json) (keys : List String) : Bool :=
  match j with
  | .obj kvs => kvs.toArray.all fun (k, _) => keys.contains k
  | _ => false

private def tagged (frame : String) (H : ByteArray → ByteArray) (tag : String) (manifest : Json) : V ByteArray := do
  let t := tag.toUTF8
  let m := (← Flock.canonUtf8 manifest).toUTF8
  return H (frame.toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 m.size.toUInt64 ++ m)

/-- `registered_value.domain(program, value, words, leaves, word_bits, ...).binding` (`verity/commitments/registered_value.py`), `leaves`
the value's rows, whose domain is `CommitmentDomain(B, -2, RangeIndexedDomain(0, leaves), hash="sha512")`: `B` the 64 bytes of
`identity_digest_sha512("verity/one-stage/registered-domain/v1", {"program", "value", "schema", "words", "leaves"})` with
`"word_bits"` beside them when it is not 16, and `extra` after them (a per-window value's `"window"`, a produced value's
`"produced"`). -/
def bindingWith (program value : String) (words leaves wordBits : Nat) (extra : List (String × Json)) :
    V ByteArray := do
  if !WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  let manifest := Json.mkObj ([("program", Json.str program), ("value", Json.str value), ("schema", Json.str HmRow.ROW_SCHEMA),
    ("words", Lean.toJson words), ("leaves", Lean.toJson leaves)] ++
    (if wordBits == WORD_BITS then [] else [(WIDTH_KEY, Lean.toJson wordBits)]) ++ extra)
  tagged "verity/tagged-sha512/v1" Flock.Sha512.hash REGISTERED_DOMAIN manifest

/-- `registered_value.domain(...).domain_id`: `bindingWith`'s binding `B` as a frame-v3 domain over the value's `leaves`
rows (above). -/
def domainWith (program value : String) (words leaves wordBits : Nat) (extra : List (String × Json)) :
    V ByteArray := do
  let binding ← bindingWith program value words leaves wordBits extra
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
`readsFrom + i` reads `readsRel[i]`, and every other unit reads nothing (`readAt`; `reads`, the population-long array).
The file states no reads: the verifier's program gives them (`withPositions`). -/
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
  /-- the template parameter it reads (`PARAM_KEY`), empty when it is the port's name -/
  param : String := ""
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

/-- A file's ports, one per entry of its `ports` object in its order, an entry's name repeated as often as the object
holds it (`ofJson` refuses that). -/
def portsOf (j : Json) : V (Array Port) := do
  let ctx := "registered reads"
  let format ← J.str j "format" ctx
  if format != FORMAT then throw s!"{ctx}: not a {FORMAT} file"
  if !keysIn j FILE_KEYS then throw s!"{ctx}: a {FORMAT} file holds the keys {FILE_KEYS} and no other"
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM fun (name, e) => do
    if !keysIn e PORT_KEYS then
      throw s!"{ctx}: port {name}: a {FORMAT} port holds keys of {PORT_KEYS} and no other (no position: C and the partition give them)"
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
    let param ← if J.has e "param" then J.str e "param" ctx else pure ""
    return { name, value := ← J.str e "value" ctx, root, leaves, words, wordBits, domain, readsRel := #[], schema, param }

/-- The first port of `ps` whose name a later port has. -/
def dupName? : List Port → Option String
  | [] => none
  | p :: ps => if ps.any (·.name == p.name) then some p.name else dupName? ps

theorem dupName?_none {ps : List Port} (h : dupName? ps = none) :
    ∀ q ∈ ps, ∀ q' ∈ ps, q'.name = q.name → q' = q := by
  induction ps with
  | nil => intro q hq; cases hq
  | cons p ps ih =>
    simp only [dupName?] at h
    split at h
    · cases h
    · rename_i hany
      have hne : ∀ x ∈ ps, x.name ≠ p.name := fun x hx he =>
        hany (List.any_eq_true.mpr ⟨x, hx, by simp [he]⟩)
      intro q hq q' hq' hn
      rcases List.mem_cons.mp hq with hq0 | hq1 <;> rcases List.mem_cons.mp hq' with hq0' | hq1'
      · exact hq0'.trans hq0.symm
      · exact absurd (hn.trans (congrArg (·.name) hq0)) (hne _ hq1')
      · exact absurd (hn.symm.trans (congrArg (·.name) hq0')) (hne _ hq1)
      · exact ih h q hq1 q' hq1' hn

/-- **The verifier's registered reads**: a `FORMAT` file's ports (`portsOf`), each name naming one port
(`ofJson_names`). A verdict reports a port by its name (`finding`) and every check of a statement's reads finds a port by
its name, so a second port of a name would be one neither reads. `verity-verify-proof` refuses a file whose text names a key
twice (`JsonKeys.dupKey`) and `JsonParse.parse` keeps one of the two, so no file it reads gets here with one; this
refusal holds for any `Json` the verifier is given. -/
def ofJson (j : Json) : V (Array Port) := do
  let own ← portsOf j
  match dupName? own.toList with
  | some nm => throw s!"registered reads: port {nm} twice: a verdict reports one port per name, and each check reads a name's port"
  | none => pure own

theorem ofJson_names {j : Json} {own : Array Port} (h : ofJson j = .ok own) :
    ∀ q ∈ own, ∀ q' ∈ own, q'.name = q.name → q' = q := by
  obtain ⟨ps, -, h⟩ := HmNets.bind_ok h
  cases hd : dupName? ps.toList with
  | some nm => rw [hd] at h; cases h
  | none =>
    rw [hd] at h
    cases h
    intro q hq q' hq' hn
    exact dupName?_none hd q (Array.mem_toList_iff.mpr hq) q' (Array.mem_toList_iff.mpr hq') hn

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
output row, `HmOut`, whose table is its instances' own rows, instance `i`'s ref `i`), whose leaf schema is the
registration's (`portSchema`), whose rows are the registered rows (`Port.fits`), whose table rows all open the
registered root, and whose instances each read the row at the position the verifier's program reads for their unit. -/
def checkPort (c : Circuit) (pub : Public) (t : Tables) (indices : Array Nat) (reg : Port) : V Unit := do
  let np := c.ports.size
  let some p := (List.range np).find? (fun p => c.ports[p]!.name == reg.name)
    | throw s!"registered reads: {reg.name} is not a row of the circuit (an input port, or an output port's row)"
  if portSchema c.ports[p]! != reg.schema then
    throw s!"registered reads: port {reg.name} is a {portSchema c.ports[p]!} row, and {reg.value}'s registered rows are {reg.schema} rows"
  if !reg.fits c.ports[p]! then
    throw s!"registered reads: port {reg.name}'s rows are {c.ports[p]!.bits} bits ({c.ports[p]!.words} u16 words), not {reg.value}'s registered rows of {reg.words.getD 0} {reg.wordBits}-bit leaves; a reader of registered leaves at another width (the same bits cut again, low part first) is not read until the Program's Lean defines that cut"
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

/-- Definition `fn` of `P` is an input the prover does not choose: bound at registration (`RegisteredInput<w>_v1`) or
supplied by the verifier (`SuppliedInput<w>_v1`, PROTOCOL.md §5.4), whose port opens a root the verifier computes. -/
def regDef (P : Program.Prog) (fn : Nat) : Bool :=
  match P.defs[fn]? with
  | some f => f.prim && (f.binding == some Program.REGISTRATION || f.binding == some Program.SUPPLIED)
  | none => false

/-- The root's nodes of `P`. -/
def rootNodes (P : Program.Prog) : Array Program.Node := ((P.defs[P.root]?).map (·.body.nodes)).getD #[]

/-- Root node `k` of `P` is a registered input. -/
def regNode (P : Program.Prog) (k : Nat) : Bool := ((rootNodes P)[k]?).any (regDef P ·.fn)

def runDef (P : Program.Prog) (fn : Nat) : Bool :=
  match P.defs[fn]? with
  | some f => f.prim && f.binding == some Program.RUN
  | none => false

/-- Root node `k` of `P` is an input bound per run: no registered value, its rows a fresh input of the units reading them. -/
def runNode (P : Program.Prog) (k : Nat) : Bool := ((rootNodes P)[k]?).any (runDef P ·.fn)

/-- The one value holding run input node `w`'s rows. Every port that reads any of them names it, so they are committed
once, in one tree, each row a leaf, at the position C gives (`OneCircuit.pos`). -/
def freshValue (w : Nat) : String := s!"input/n{w}"

/-- Whether `v` is `freshValue w` for some node `w`: a name reserved for a run input node's tree. -/
def isFreshValue (v : String) : Bool :=
  let d := (v.drop 7).toString
  v.startsWith "input/n" && !d.isEmpty && d.all Char.isDigit && freshValue d.toNat! == v

/-- The refusal of a port `name` of role `role` registered as `v`, a run input node's tree's name (`isFreshValue`). -/
def RESERVED_WHY (role name v : String) : String :=
  s!"registered reads: {role} {name} is registered as {v}, a run input node's tree: that name is the node's alone"

/-- The first value two of `ps` name with another root, domain or row count: a value is one commitment. -/
def splitValue? : List Port → Option String
  | [] => none
  | p :: qs =>
    if qs.any fun q => q.value == p.value && (q.root != p.root || q.domain != p.domain || q.leaves != p.leaves) then
      some p.value
    else splitValue? qs

/-- A port's template parameter (`PARAM_KEY`) as its name and, for `name[i]`, row `i` of an array of rows. -/
def paramRow (param : String) : String × Option Nat :=
  match param.splitOn "[" with
  | [p, r] => match r.splitOn "]" with
    | [d, ""] => match d.toNat? with
      | some i => (p, some i)
      | none => (param, none)
    | _ => (param, none)
  | _ => (param, none)

end Registered

namespace OneCircuit

open Registered (regNode paramRow)

/-- Member `m` of root node `k` reads its parameter `param` (`paramRow`) at an input node's row, a registered input or a
run input (`runNode`): the node
the row (its leaf over the row's leaves) and the row's leaves, from the argument's part holding the member's row (`Program.laneWire`).
A form with no closed form here (an explicit part, a row across parts or lanes) is refused by name. -/
def regRow (P : Program.Prog) (k m : Nat) (param : String) : V (Nat × Nat × Nat) := do
  let ctx := s!"registered reads: parameter {param} of root node {k}"
  let node := (Registered.rootNodes P)[k]!
  let f := P.defs[node.fn]!
  let (p, row) := paramRow param
  let some i := (List.range f.params.size).find? (f.params[·]!.1 == p) | throw s!"{ctx}: {f.key} has no parameter {p}"
  let ty := f.params[i]!.2
  let (t0, len) ← match row, ty with
    | none, _ => pure (0, ty.leaves)
    | some r, .array n e => if r < n then pure (r * e.leaves, e.leaves) else throw s!"{ctx}: row {r} of {n}"
    | some _, _ => throw s!"{ctx}: a row of a parameter that is no array"
  let a := node.args[i]!
  let x ← match a.dims with
    | none => pure t0
    | some (d, Lb) =>
      if Lb == 0 || t0 % Lb + len > Lb then throw (notDerived s!"pos : {ctx}: a row across members")
      pure ((t0 / Lb * d) * Lb + t0 % Lb + m * Lb)
  let some j := (List.range a.seq.parts.size).find? fun j =>
      a.seq.starts[j]! ≤ x && x < a.seq.starts[j]! + a.seq.parts[j]!.count
    | throw s!"{ctx}: argument index {x} is in no part"
  let q := a.seq.parts[j]!
  let s0 := a.seq.starts[j]!
  if x + len > s0 + q.count then throw (notDerived s!"pos : {ctx}: a row across parts")
  let (run, lanes) := Program.partLanes q
  let (w, leaf, _, _) ← match Program.laneWire run lanes (x - s0) 0 len 1 with
    | .ok v => pure v
    | .error _ => throw (notDerived s!"pos : {ctx}: a row of no lane's run (an explicit part, or across lanes)")
  if !(regNode P w || Registered.runNode P w) then
    throw s!"{ctx}: member {m} reads root node {w}, which is no registered input and no run input"
  if len == 0 || leaf % len != 0 then throw s!"{ctx}: member {m} reads leaf {leaf} of node {w}, off its rows of {len}"
  return (w, leaf / len, len)

/-- The refusal of the members of root node `k` reading parameter `param` in a form with no closed form here (`what`)
where its argument reads an input node (`regRows?`). -/
def READ_FORM_WHY (k : Nat) (param what : String) : String :=
  s!"registered reads: parameter {param} of root node {k}: {what}, and its argument reads an input node: a parameter " ++
    "that reads an input reads one node, each row in one lane's run"

/-- `regRow` for the members `[m0, m0 + n)` of root node `k` at once (`Program.laneWire` over their rows, a step of a
member's leaves apart): the input node, registered or run, every one of them reads `param` at, with a row's leaves;
none where they read only root nodes that are no input (other blocks' outputs). Rows a lane's levels hold in no mixed
radix are read a member at a time. Refused by name (`READ_FORM_WHY`): members reading an input node and another node,
and, where the argument reads an input node anywhere, a row with no closed form here (across members, parts or lanes;
an explicit part). Its work is the argument's parts and a lane's levels, and a member at a time only where the
members' rows have no closed form. -/
def regRows? (P : Program.Prog) (k m0 n : Nat) (param : String) : V (Option (Nat × Nat)) := do
  let ctx := s!"registered reads: parameter {param} of root node {k}"
  let node := (Registered.rootNodes P)[k]!
  let f := P.defs[node.fn]!
  let (p, row) := paramRow param
  let some i := (List.range f.params.size).find? (f.params[·]!.1 == p) | throw s!"{ctx}: {f.key} has no parameter {p}"
  let ty := f.params[i]!.2
  let (t0, len) ← match row, ty with
    | none, _ => pure (0, ty.leaves)
    | some r, .array c e => if r < c then pure (r * e.leaves, e.leaves) else throw s!"{ctx}: row {r} of {c}"
    | some _, _ => throw s!"{ctx}: a row of a parameter that is no array"
  let a := node.args[i]!
  let input (w : Nat) : Bool := regNode P w || Registered.runNode P w
  let inputs := a.seq.parts.any fun q => (Registered.partNodes q).any input
  let edge (what : String) : V (Option (Nat × Nat)) := if inputs then throw (READ_FORM_WHY k param what) else pure none
  let L := match a.dims with | some (_, Lb) => Lb | none => 0
  if (a.dims.any fun (_, Lb) => Lb == 0 || t0 % Lb + len > Lb) then return ← edge "a row across members"
  let xOf (m : Nat) : Nat := match a.dims with
    | none => t0
    | some (d, Lb) => (t0 / Lb * d) * Lb + t0 % Lb + m * Lb
  let partOf (x : Nat) : Option Nat := (List.range a.seq.parts.size).find? fun j =>
    a.seq.starts[j]! ≤ x && x < a.seq.starts[j]! + a.seq.parts[j]!.count
  let wire (x c len : Nat) (L : Nat) : Option Nat := do
    let j ← partOf x
    let q := a.seq.parts[j]!
    guard (x + (c - 1) * L + len ≤ a.seq.starts[j]! + q.count)
    let (run, lanes) := Program.partLanes q
    match Program.laneWire run lanes (x - a.seq.starts[j]!) L len c with
    | .ok (w, _, _, _) => some w
    | .error _ => none
  if let some w := wire (xOf m0) (max n 1) len L then
    return if input w then some (w, len) else none
  let mut got : Array Nat := #[]
  for m in [0:max n 1] do
    match wire (xOf (m0 + m)) 1 len 0 with
    | some w => if !got.contains w then got := got.push w
    | none => return ← edge "a row of no lane's run (an explicit part, or across parts or lanes)"
  match got.toList with
  | [w] => return if input w then some (w, len) else none
  | ws => if ws.any input then throw (READ_FORM_WHY k param s!"its members read root nodes {ws}") else return none

/-- **theta's positions**: `pos : C -> Partition -> Block -> Port -> Position`, the row of the registered value
`reg` each unit of `indices` reads at `reg`'s parameter (`regRow`), a function of C (`P`): the registered input node's
leaf over the row's leaves, the leaf order of its tree. Every unit reads one node, whose rows are `reg.leaves`. Its
work is the statement's units, each found from the root's nodes and its argument's parts. The code's Core
`RegisteredCircuit.pos`, restricted to input nodes. A run input node `w` is read from its one tree `freshValue w`, and
a registered one under any other name. -/
def pos (P : Program.Prog) (indices : Array Nat) (reg : Registered.Port) : V Registered.Port := do
  let ns := HmLinks.nodes P
  let param := if reg.param.isEmpty then reg.name else reg.param
  let first := indices.foldl min (indices.getD 0 0)
  let mut rel : Array (Option Nat) := Array.replicate (indices.foldl max 0 + 1 - first) none
  let mut node : Option Nat := none
  for u in indices do
    let some (k, m) := HmLinks.unitAt ns u | throw s!"registered reads: port {reg.name}: unit {u} is no unit"
    let (w, r, len) ← regRow P k m param
    if node.any (· != w) then
      throw s!"registered reads: port {reg.name}: units read registered nodes {node.getD 0} and {w}, and a port is one value"
    if node.isNone then
      if regNode P w && reg.value.startsWith "input/" then
        throw s!"registered reads: port {reg.name}: C's node {w} is a registered input, read at its registered root, and the port names the fresh input {reg.value}"
      if Registered.runNode P w && reg.value != Registered.freshValue w then
        throw s!"registered reads: port {reg.name}: C's node {w} is a run input, its rows committed once, in {Registered.freshValue w}, and the port names {reg.value}"
      let wn := (Registered.rootNodes P)[w]!
      let rows := (P.defs[wn.fn]!).outLeaves * wn.members / len
      if rows != reg.leaves then
        throw s!"registered reads: port {reg.name}: {reg.value} holds {reg.leaves} rows, and C's input node {w} {rows}"
    node := some w
    rel := rel.set! (u - first) (some r)
  return { reg with readsRel := rel, readsFrom := first }

end OneCircuit

namespace Registered

open Lean (Json)

/-- The name of the one tree holding template `t`'s output field `j`, in every root, domain and file. -/
def fieldTree (t : String) (j : Nat) : String := s!"{t}/o{j}"

/-- The whole-output form `treeReads` extends to output fields, kept beside it: row `reg` of in(S)'s layout (`OneCircuit.io`, its first parameter leaf `lo`) at the statement's units `indices`,
from the verifier's program `P`: each unit reads the leaf of its writer (`HmLinks.writerOf`) in the writer
template's output tree (`HmLinks.treeLeaf`), and `reg` must be that tree (`fieldTree`), one for every unit: a row of
in(S) with a root of its own, or of another tree, is refused by name. A writer of more than one output field is
refused: the row reads its writer's whole output (`HmLinks.checkUnit`). Where no unit of `indices` has a writer, the
row is an ordinary input row and `reg` is returned unread. Its work is the statement's units, each found from the
root body's nodes. -/
def treeReadsWhole (P : Program.Prog) (indices : Array Nat) (lo : Nat) (reg : Port) : V Port := do
  let ns := HmLinks.nodes P
  let body := (P.defs[P.root]!).body
  let first := indices.foldl min (indices.getD 0 0)
  let mut rel : Array (Option Nat) := Array.replicate (indices.foldl max 0 + 1 - first) none
  let mut wired := 0
  for u in indices do
    let some v ← HmLinks.writerOf P ns u lo | continue
    let some (t, leaf) := HmLinks.treeLeaf P ns v
      | throw s!"registered reads: row {reg.name} of in(S): unit {u}'s writer {v} is outside the program's units"
    let some (k, _) := HmLinks.unitAt ns v | throw s!"registered reads: row {reg.name} of in(S): unit {v} is no unit"
    let fs := Program.fieldsOf (P.defs[body.nodes[k]!.fn]!).ret
    if fs.length != 1 then
      throw s!"registered reads: row {reg.name} of in(S) reads unit {v} of {t}, of {fs.length} output fields, and a row reads its writer's whole output, one field's tree"
    if isFreshValue reg.value then throw (RESERVED_WHY "row of in(S)" reg.name reg.value)
    if reg.value != fieldTree t 0 then
      throw s!"registered reads: row {reg.name} of in(S) is committed as {reg.value}, and C wires it from {fieldTree t 0}: a crossing wire is committed once, in its writer's output field's tree, and the reader's row opens that tree by path"
    rel := rel.set! (u - first) (some leaf)
    wired := wired + 1
  if wired == 0 then return reg
  if wired != indices.size then
    throw s!"registered reads: row {reg.name} of in(S): {wired} of the statement's {indices.size} units read it from a writer, and a row is one tree's"
  return { reg with readsRel := rel, readsFrom := first }

/-- The refusal of row `name` of in(S) whose unit `u` reads leaves of unit `v` (of template `t`, `n` output fields) that
are no one output field of it, whole and in order. -/
def FIELD_WHY (name : String) (u v : Nat) (t : String) (n : Nat) : String :=
  s!"registered reads: row {name} of in(S): unit {u} reads leaves of unit {v} of {t} that are no one of its {n} output fields, whole and in order: a row reads one output field's tree, at the positions C gives"

/-- Row `reg` of in(S)'s layout (`OneCircuit.io`, its parameter leaves `[lo, hi)`) at the statement's units `indices`,
from the verifier's program `P`: each unit's leaves are one output field `j` of its writer (`HmLinks.writerOf`,
`HmLinks.fieldRead`, from C's edges and the writer template's return type), read at the writer's leaf
(`HmLinks.treeLeaf`) of that field's tree, and `reg` must be that tree (`fieldTree t j`), the same field for every unit:
a row that is no one field whole (`FIELD_WHY`), of two fields across the units, with a root of its own, or of another
tree, is refused by name. A single-field writer's whole output is its field 0. Where no unit of `indices` has a writer,
the row is an ordinary input row and `reg` is returned unread. Its work is the statement's units, each found from the
root body's nodes. -/
def treeReads (P : Program.Prog) (indices : Array Nat) (lo hi : Nat) (reg : Port) : V Port := do
  let ns := HmLinks.nodes P
  let body := (P.defs[P.root]!).body
  let first := indices.foldl min (indices.getD 0 0)
  let mut rel : Array (Option Nat) := Array.replicate (indices.foldl max 0 + 1 - first) none
  let mut wired := 0
  let mut field : Option Nat := none
  for u in indices do
    let some v ← HmLinks.writerOf P ns u lo | continue
    let some (t, leaf) := HmLinks.treeLeaf P ns v
      | throw s!"registered reads: row {reg.name} of in(S): unit {u}'s writer {v} is outside the program's units"
    let some (k, m) := HmLinks.unitAt ns v | throw s!"registered reads: row {reg.name} of in(S): unit {v} is no unit"
    let some (ku, mu) := HmLinks.unitAt ns u | throw s!"registered reads: row {reg.name} of in(S): unit {u} is no unit"
    let ops ← Program.operands P P.rootScope (body.nodes[ku]!) mu
    let og ← HmLinks.outGates P k m
    let some j := HmLinks.fieldRead P og (ops.extract lo hi) k
      | throw (FIELD_WHY reg.name u v t (Program.fieldsOf (P.defs[body.nodes[k]!.fn]!).ret).length)
    if let some j' := field then
      if j' != j then
        throw s!"registered reads: row {reg.name} of in(S): the statement's units read fields {j'} and {j} of their writers, and a row is one tree's"
    field := some j
    if isFreshValue reg.value then throw (RESERVED_WHY "row of in(S)" reg.name reg.value)
    if reg.value != fieldTree t j then
      throw s!"registered reads: row {reg.name} of in(S) is committed as {reg.value}, and C wires it from {fieldTree t j}: a crossing wire is committed once, in its writer's output field's tree, and the reader's row opens that tree by path"
    rel := rel.set! (u - first) (some leaf)
    wired := wired + 1
  if wired == 0 then return reg
  if wired != indices.size then
    throw s!"registered reads: row {reg.name} of in(S): {wired} of the statement's {indices.size} units read it from a writer, and a row is one tree's"
  return { reg with readsRel := rel, readsFrom := first }

/-- `--linked F`'s rows: an object from an output port's name to the row of round 1's tree each unit of the
population writes there, a natural number or null (a unit of another template). -/
def linkedOf (j : Json) : V (Array (String × Array (Option Nat))) := do
  let .obj kvs := j | throw "linked: not an object from a port to its rows"
  kvs.toArray.mapM fun (p, v) => do
    let .arr xs := v | throw s!"linked: port {p}: not a list of rows"
    return (p, ← xs.mapM fun x => match x with
      | .null => pure none
      | x => match x.getNat? with
        | .ok n => pure (some n)
        | .error _ => throw s!"linked: port {p}: a row that is neither a natural number nor null")

/-- The refusal of linked output row `name` registered at its own template `t`'s field tree `fieldTree t j`: a linked
row is round 1's, read at round 1's tree. -/
def LINKED_OWN_WHY (name t : String) (j : Nat) : String :=
  s!"registered reads: output row {name} is linked to round 1's record, and it is registered as {fieldTree t j}, its template's own field tree: a linked row is read at round 1's tree, under round 1's name"

/-- The refusal of linked output row `name` whose rows are not one row of `value`'s leaves for each unit. -/
def LINKED_ROWS_WHY (name value : String) : String :=
  s!"registered reads: output row {name} is linked to round 1's {value}, and its rows are not one row of {value}'s leaves for each of the statement's units"

/-- `rel` from unit `first` on reads one row below `leaves` at each unit of `indices`, from the least of them, and
nothing past the greatest. -/
def readsCover (rel : Array (Option Nat)) (first : Nat) (indices : Array Nat) (leaves : Nat) : Bool :=
  first == indices.foldl min (indices.getD 0 0) && rel.size == indices.foldl max 0 + 1 - first &&
    indices.all fun u => match rel[u - first]? with
      | some (some r) => decide (r < leaves)
      | _ => false

/-- Output row `j` of the statement's circuit `c` (its port `HmOut.nIn c + j`) at the statement's units `indices`, from the
verifier's program `P`: the row is output field `j` of each unit of the circuit's template `t` (`HmRow.queryTemplate`),
the field's bits from the field's first in whole words, and that field is committed once, in its tree (`fieldTree t j`),
so `reg` must be that tree and each unit reads its own leaf in it by `HmLinks.treeLeaf`, the leaf `treeReads` opens for
a reader of the field. An output of several fields in fewer rows, a row of another width or offset, a port registered
as another value and a unit of another template are refused by name. A row `linked` names (by port, with the row of
round 1's tree each unit of the population writes) is read at round 1's tree, `reg`'s root, domain
and leaves, at those rows, and its own field tree is refused (`LINKED_OWN_WHY`), as are rows that are not one row of
`reg`'s leaves for each unit of `indices` (`LINKED_ROWS_WHY`). -/
def outReadsAt (c : Circuit) (P : Program.Prog) (indices : Array Nat) (j : Nat) (reg : Port)
    (linked : Array (String × Array (Option Nat)) := #[]) : V (Array (Option Nat) × Nat) := do
  if isFreshValue reg.value then throw (RESERVED_WHY "output row" reg.name reg.value)
  let some (t, _) := HmRow.queryTemplate c
    | throw s!"registered reads: output row {reg.name}: the circuit names no template of C (META program_digests)"
  let some f := (rootNodes P).findSome? fun x => (P.defs[x.fn]?).filter (·.key == t)
    | throw s!"registered reads: output row {reg.name}: C's root calls no {t}, the circuit's template"
  let fs := Program.fieldsOf f.ret
  if j ≥ c.outRowPorts.size then
    throw s!"registered reads: output row {reg.name}: the circuit has {c.outRowPorts.size} output rows, and this is row {j}"
  if fs.length != c.outRowPorts.size then
    throw s!"registered reads: output row {reg.name}: {t} returns {fs.length} output fields and the circuit has {c.outRowPorts.size} output rows: each field is committed in its own tree ({fieldTree t 0}, …), one output row a field"
  let bitsOf := fun (ty : Program.Ty) => (Program.Ty.widths ty).foldl (· + ·) 0
  let before := (fs.take j).foldl (fun n ty => n + bitsOf ty) 0
  let bits := bitsOf (fs.getD j default)
  let (_, _, w0, words) := HmOut.outGroup c j
  if 16 * w0 != before || words != (bits + 15) / 16 then
    throw s!"registered reads: output row {reg.name} is output words [{w0}, {w0 + words}), and {t}'s output field {j} is bits [{before}, {before + bits}): a row is one field's whole words"
  if let some (_, rows) := linked.find? (·.1 == reg.name) then
    if reg.value == fieldTree t j then throw (LINKED_OWN_WHY reg.name t j)
    let first := indices.foldl min (indices.getD 0 0)
    let rel := rows.extract first (indices.foldl max 0 + 1)
    if !readsCover rel first indices reg.leaves then throw (LINKED_ROWS_WHY reg.name reg.value)
    return (rel, first)
  if reg.value != fieldTree t j then
    throw s!"registered reads: output row {reg.name} is registered as {reg.value}, and it is {t}'s output field {j}, committed once in {fieldTree t j}: the row opens that tree's root, at its unit's leaf"
  let ns := HmLinks.nodes P
  let first := indices.foldl min (indices.getD 0 0)
  let mut rel : Array (Option Nat) := Array.replicate (indices.foldl max 0 + 1 - first) none
  for u in indices do
    let some (tu, leaf) := HmLinks.treeLeaf P ns u
      | throw s!"registered reads: output row {reg.name}: unit {u} is outside the program's units"
    if tu != t then throw s!"registered reads: output row {reg.name}: unit {u} is an instance of {tu}, not the circuit's template {t}"
    rel := rel.set! (u - first) (some leaf)
  return (rel, first)

/-- `reg` reading at `outReadsAt`'s positions from its first unit on, every other field its own. -/
def outReads (c : Circuit) (P : Program.Prog) (indices : Array Nat) (j : Nat) (reg : Port)
    (linked : Array (String × Array (Option Nat)) := #[]) : V Port := do
  let r ← outReadsAt c P indices j reg linked
  return { reg with readsRel := r.1, readsFrom := r.2 }

/-- The output row port `q` of `c` is: a circuit with hidden outputs declares its input rows (META `input_ports`, which
`HmOut.checkRows` holds to `HmOut.nIn`), and each port past them is an output row named after its run's first output
port, as `checkRows` holds; a circuit that declares none, or rows that `checkRows` refuses so (a count not the rows
before one output row each, or a port past it named otherwise), as a public-output statement's may, has no output row
among its ports, whatever its output ports. -/
def outRow? (c : Circuit) (q : Nat) : Option Nat :=
  match J.nat c.metaJ "input_ports" "circuit META" with
  | .ok k =>
    if k + c.outRowPorts.size == c.ports.size && k ≤ q
        && (c.outPorts[(HmOut.outGroup c (q - k)).1]?.map (·.1)) == some (c.ports.getD q default).name then
      some (q - k)
    else none
  | .error _ => none

/-- The verifier's registered reads `own` with every port's reads derived at the statement's circuit `c` and public file
`pub` from C, the verifier's program `program`: an output row (`outRow?`) by `outReads`, at round 1's rows where
`linked` names its port, a row of in(S)
(`OneCircuit.io`) by `treeReads`, a registered value's or a run input's by `OneCircuit.pos`; without the program every
port is refused, and so is a value named under two roots, domains or row counts (`splitValue?`). -/
def withPositions (c : Circuit) (pub : Public) (program : Option ByteArray) (own : Option (Array Port))
    (linked : Array (String × Array (Option Nat)) := #[]) : V (Option (Array Port)) := do
  let some own := own | return none
  let some b := program | throw "registered reads: a port's reads are derived from C, the verifier's program (--program)"
  if let some v := splitValue? own.toList then
    throw s!"registered reads: value {v} is registered under two roots, domains or row counts, and a value is one commitment every port reading it opens"
  let P ← Program.decode b
  let indices ← unitIndices pub
  let rows ← OneCircuit.io c P
  some <$> own.mapM fun reg =>
    match ((List.range c.ports.size).find? fun q => (c.ports.getD q default).name == reg.name).bind (outRow? c) with
    | some j => outReads c P indices j reg linked
    | none => ins c P indices rows reg
where
  ins (c : Circuit) (P : Program.Prog) (indices : Array Nat) (rows : Array (Nat × Nat × Nat)) (reg : Port) : V Port :=
    match rows.find? fun (p, _, _) => (c.ports.getD p default).name == reg.name with
    | some (_, lo, hi) => do
      let r ← treeReads P indices lo hi reg
      if r.readsRel.isEmpty then OneCircuit.pos P indices reg else pure r
    | none => OneCircuit.pos P indices reg

/-- **The derived port named `nm`**: the verifier's port of that name in `own` with its reads as `withPositions` derives
them at the statement's circuit `c` and public file `pub` from C, the verifier's `program`, the port the session's
registered check opens (`checkRegistered`); none where `own` is none or names no such port. A check of a port's reads
reads this port, never the file's, whose reads a `FORMAT` file never states. -/
def derivedPort (c : Circuit) (pub : Public) (program : Option ByteArray) (own : Option (Array Port)) (nm : String) :
    V (Option Port) := do
  let some own' ← withPositions c pub program own | return none
  return own'.find? (·.name == nm)

theorem derivedPort_ok {c : Circuit} {pub : Public} {program : Option ByteArray} {own : Option (Array Port)}
    {nm : String} {p : Port} (h : derivedPort c pub program own nm = .ok (some p)) :
    ∃ own', withPositions c pub program own = .ok (some own') ∧ own'.find? (·.name == nm) = some p := by
  unfold derivedPort at h
  cases hw : withPositions c pub program own with
  | error e => simp [hw, bind, Except.bind] at h
  | ok o =>
    cases o with
    | none => simp [hw, bind, Except.bind, pure, Except.pure] at h
    | some own' =>
      simp only [hw, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨own', rfl, h⟩

/-- The statement's registered reads against the verifier's own `own` (`--registered`). The header's `registered` must
name exactly `own`'s ports, each a port of the circuit and a table of a shared-row file: every table row's `b ‖ c`, as the
value's row at its stated position, opens the registered root through its path, and every instance reads, by its ref,
the position the verifier's program reads for its unit (`checkPort`; `FlockProofs.Registered`: `check_ok`); Without
`own`, a statement that names registered reads is refused: its rows would be checked against no root. With `own`, the
units must be `derived`: a unit's position is the verifier's, but which unit an instance is comes from its label, which
binds nothing unless the verifier derived it. -/
def check (c : Circuit) (pub : Public) (own : Option (Array Port)) (derived : Bool) : V Unit := do
  let hdr := pub.header.getObjVal? "registered"
  let some own := own
    | do
      if hdr.isOk then throw "registered reads: the statement reads registered values, and the verifier holds no registered roots (--registered)"
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
  d.callees.filter fun id => (Program.inputId? id).any fun (b, _) => b == Program.REGISTRATION || b == Program.SUPPLIED

/-- **A program that reads a registered value is verified only against registered roots** (`Flock.HmRow`): with the
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

/-- Every port of the circuit, each input port and each output row, is a port of `own`. -/
def complete (c : Circuit) (own : Array Port) : Bool := c.ports.all fun p => own.any (·.name == p.name)

/-- **Every port a statement's outputs read is registered** (PROTOCOL.md §16.10; `verity-verify-proof` runs it on every
statement whenever `--registered F` is given): every port of the circuit is a port of the verifier's own registered
reads (`checkComplete_ok`), so no output's cone reads a row the statement commits afresh and its public file states (a
lane's index, say), and `check` opens each of them at its registered root. It reads the circuit's ports, not its gates:
whatever an output's cone reads is a port's row. -/
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

/-- A root node's argument reads a registered input: a part of it reads a `RegisteredInput<w>_v1` root node. -/
def argReads (P : Program.Prog) (a : Program.Arg) : Bool := a.seq.parts.any fun q => (partNodes q).any (regNode P)

/-- **The template parameters that read a registered input** (`Flock.HmRow`), from the verifier's own program: for
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

/-- **Every registered input the statement reads has a port** (`Flock.HmRow`): with the verifier's own program and
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

/-- Whether every sample by `d`'s sampling design samples unit `u`: a subset sample of the whole population, a `p = 1` sample, or a unit whose
stratum samples all of its units (`k_s = n_s`). -/
def samplesInFull (d : Draw.UnitSample) (u : Nat) : Bool :=
  let full (s : Draw.Stratum) := s.units.any (fun (lo, hi) => lo ≤ u && u < hi) && s.k == s.size
  match d.samplingDesign with
  | .subset k => k == d.population
  | .bernoulli a b => a == b
  | .stratified _ strata => strata.any full
  | .work _ strata => strata.any (full ·.toStratum)

/-- The first port of `own` and unit that reads a round's commitment (`roundCommitment`) by `own`'s reads, and that `d`
does not sample in full. -/
def fullMiss (own : Array Port) (d : Draw.UnitSample) : Option (Port × Nat) :=
  own.findSome? fun p =>
    if roundCommitment p.value then
      ((List.range p.reads.size).find? fun u => p.reads[u]!.isSome && !samplesInFull d u).map (p, ·)
    else none

/-- U4 (`Flock.HmRow`): a sampled statement samples in full every unit that reads a round's commitment by the verifier's
own reads `own` (`fullMiss`), so no stage's message is read off a sample. `draw` is the record's; a statement without a
sample proves its whole population. -/
def checkFull (own : Option (Array Port)) (draw : Option Json) : V Unit := do
  let (some own, some j) := (own, draw) | pure ()
  let d ← Draw.UnitSample.ofJson j
  match fullMiss own d with
  | none => pure ()
  | some (p, u) =>
    throw s!"U4: unit {u} reads {p.value}, a round's commitment (port {p.name}), and the draw does not draw it in full"

/-! ### A recursive session's inner statement: its description, read from the developer's registration (`Tags.withPlainLeaves`) -/

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

/-- **A recursive session's inner statement's description, read from the developer's registration** (`Tags.withPlainLeaves`):
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
rows (`HmOut`): an output row's table is its instances' own rows, each opened against the registered root at the position
its unit reads, so the instance's output is that row of the registered value. -/
def checkRegistered (st : Stmt) (registered : Option (Array Registered.Port)) : V Unit :=
  match registered with
  | some _ => Registered.namesOnce st.c >>= fun _ => Registered.check st.c st.pub registered (Registered.derived st.partition)
  | none => Registered.check st.c st.pub none (Registered.derived st.partition)

/-- What a verdict reports of the rows its statements' registered ports opened: `{port: [[position, b ‖ c] ...]}`, each
table row's stated position and its hiding commitment `b ‖ c` (the commit-time opening's digest), in table order across
the statements; null without registered reads. A produced record holds every session's to its own
(`Spot.Produced.shared`). -/
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
