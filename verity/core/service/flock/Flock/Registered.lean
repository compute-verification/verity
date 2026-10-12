/-
PROTOCOL.md §16.10 (registered reads); `protocols/sampled_proofs/README.md`, "Registered values": a statement's rows of a
registered value (a program's weights, constants and tables, committed once at registration under `hm96-sha512/row/v1`
leaves), opened against the verifier's own registered roots at the positions its own program reads.
-/
import Flock.Canon
import Flock.HmRow

namespace Flock
namespace Registered

open Lean (Json)

def FORMAT : String := "verity/registered-reads/v0"

def REGISTERED_DOMAIN : String := "verity/one-stage/registered-domain/v0"
/-- The default leaf width, which the domain and the entry name only by leaving `word_bits` out. -/
def WORD_BITS : Nat := 16
def WIDTHS : List Nat := [16, 32]
def WIDTH_KEY : String := "word_bits"

private def tagged (frame : String) (H : ByteArray → ByteArray) (tag : String) (manifest : Json) : V ByteArray := do
  let t := tag.toUTF8
  let m := (← Flock.canon manifest).toUTF8
  return H (frame.toUTF8.push 0 ++ Flock.be32 t.size.toUInt32 ++ t ++ Flock.be64 m.size.toUInt64 ++ m)

/-- `registered.domain(program, value, words, leaves, word_bits).domain_id` (one-stage `registered.py`), `leaves` the
value's rows: `CommitmentDomain(B, -2, RangeIndexedDomain(0, leaves), hash="sha512").domain_id`, `B` the bytes of
`identity_digest("verity/one-stage/registered-domain/v0", {"program", "value", "schema", "words", "leaves"})` with
`"word_bits"` beside them when it is not 16. -/
def domain (program value : String) (words leaves wordBits : Nat) : V ByteArray := do
  if !WIDTHS.contains wordBits then throw s!"a registered value's leaves are 16 or 32 bits, not {wordBits}"
  let manifest := Json.mkObj ([("program", Json.str program), ("value", Json.str value), ("schema", Json.str HmRow.ROW_SCHEMA),
    ("words", Lean.toJson words), ("leaves", Lean.toJson leaves)] ++
    (if wordBits == WORD_BITS then [] else [(WIDTH_KEY, Lean.toJson wordBits)]))
  let binding ← tagged "veritor/tagged-sha256/v1" Flock.Sha256.hash REGISTERED_DOMAIN manifest
  let positions ← tagged "verity/tagged-sha512/v1" Flock.Sha512.hash "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", Lean.toJson (0 : Nat)), ("step", Lean.toJson (1 : Nat)), ("stop", Lean.toJson leaves)])
  return FV3.hash "domain" [binding, FV3.uint 0, positions, FV3.uint leaves] Flock.Sha512.hash

/-- A port of the verifier's own `--registered` file: the value its rows are rows of, the value's registered root, leaf
count, row width and frame-v3-sha512 domain, and the position every unit of the population reads (none: a unit of
another template). A row is `words` leaves of `wordBits` bits (16 or 32); a file that names neither (`words` none,
`wordBits` 16) has rows of 16-bit words, the port's own. -/
structure Port where
  name : String
  value : String
  root : ByteArray
  leaves : Nat
  words : Option Nat
  wordBits : Nat
  domain : ByteArray
  reads : Array (Option Nat)
  deriving Inhabited

def ofJson (j : Json) : V (Array Port) := do
  let ctx := "registered reads"
  if (← J.str j "format" ctx) != FORMAT then throw s!"{ctx}: not a {FORMAT} file"
  let .ok (.obj kvs) := j.getObjVal? "ports" | throw s!"{ctx}: ports is not an object"
  if kvs.toArray.isEmpty then throw s!"{ctx}: no port"
  kvs.toArray.mapM fun (name, e) => do
    let root ← unhex (← J.str e "root" ctx)
    let domain ← unhex (← J.str e "domain" ctx)
    let leaves ← J.nat e "leaves" ctx
    if root.size != 64 || domain.size != 64 || leaves == 0 then
      throw s!"{ctx}: port {name}: a 64-byte root and domain, and at least one leaf"
    let wordBits ← if J.has e "word_bits" then J.nat e "word_bits" ctx else pure 16
    let words ← if J.has e "words" then some <$> J.nat e "words" ctx else pure none
    if !(wordBits == 16 || wordBits == 32) || words == some 0 || (words.isNone && wordBits != 16) then
      throw s!"{ctx}: port {name}: leaves of 16 or 32 bits, at least one to a row, and a row's words when they are 32"
    let reads ← (← J.arr e "reads" ctx).mapM fun r => match r with
      | .null => pure none
      | r => some <$> J.asNat r s!"{ctx}: a read position"
    return { name, value := ← J.str e "value" ctx, root, leaves, words, wordBits, domain, reads }

/-- The u16 words of a registered row of `words` leaves of `wordBits` bits, as the statement lays out the port's row:
each leaf `wordBits / 16` words, low first, zero-padded to whole SHA-512 blocks of 64 words. -/
def rowWords (words wordBits : Nat) : Nat := (words * (wordBits / 16) + 63) / 64 * 64

/-- The circuit's port rows, of `n` u16 words, are `reg`'s registered rows. -/
def Port.fits (reg : Port) (n : Nat) : Bool := reg.words.all fun w => n == rowWords w reg.wordBits

/-- The frame-v3-sha512 node over `l` and `r` at index `idx` of level `d`. -/
def node (dom : ByteArray) (d idx : Nat) (l r : ByteArray) : ByteArray :=
  FV3.hash "node" [dom, FV3.uint d, FV3.uint idx, l, r] Sha512.hash

/-- The root a digest `cur` at index `idx` of level `d` climbs to through the siblings `sibs`, lowest first. -/
def climbL (dom : ByteArray) : ByteArray → Nat → Nat → List ByteArray → ByteArray
  | cur, _, _, [] => cur
  | cur, idx, d, sib :: rest =>
    climbL dom (if idx % 2 = 0 then node dom d (idx / 2) cur sib else node dom d (idx / 2) sib cur) (idx / 2) (d + 1) rest

/-- A path's `depth` siblings, 64 bytes each, leaf level first. -/
def sibs (path : ByteArray) (depth : Nat) : List ByteArray :=
  (List.range depth).map fun d => path.extract (64 * d) (64 * d + 64)

/-- The frame-v3-sha512 root a leaf at `i` climbs to through `path` (the siblings, leaf level first, 64 bytes each). -/
def climb (dom leaf : ByteArray) (i depth : Nat) (path : ByteArray) : ByteArray := climbL dom leaf i 0 (sibs path depth)

/-- The frame-v3-sha512 leaf at `pos` of a registered value's row whose `hm96-sha512/row/v1` leaf is `payload`. -/
def rowLeaf (dom : ByteArray) (pos : Nat) (payload : ByteArray) : ByteArray :=
  FV3.leaf dom pos HmRow.ROW_SCHEMA payload Sha512.hash

/-- A row whose `hm96-sha512/row/v1` leaf is `payload` opens `reg`'s registered root at `pos` through `path`: a position
among its leaves, one 64-byte sibling per level, and the climb ends at the root. -/
def opensLeaf (reg : Port) (payload : ByteArray) (pos : Nat) (path : ByteArray) : Bool :=
  decide (pos < reg.leaves) && decide (path.size = 64 * log2ceil reg.leaves) &&
    decide (climb reg.domain (rowLeaf reg.domain pos payload) pos (log2ceil reg.leaves) path = reg.root)

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
    !(decide (ref t np p i < rows) && decide (reg.reads[indices.getD i 0]? = some (some (positions.getD (ref t np p i) 0))))

/-- One port of the verifier's registered reads: a row of the circuit (an input port's, or a hidden-output statement's
output row, §16.13, whose table is its instances' own rows, instance `i`'s ref `i`), an `hm96-sha512/row/v1` row as the
registration commits one, whose rows are the registered rows (`Port.fits`), whose table rows all open the registered
root, and whose instances each read the row at the position the verifier's program reads for their unit. -/
def checkPort (c : Circuit) (pub : Public) (t : Tables) (indices : Array Nat) (reg : Port) : V Unit := do
  let np := c.ports.size
  let some p := (List.range np).find? (fun p => c.ports[p]!.name == reg.name)
    | throw s!"registered reads: {reg.name} is not a row of the circuit (an input port, or an output port's row)"
  if c.ports[p]!.v2 then
    throw s!"registered reads: port {reg.name} is a {if c.ports[p]!.segs.isSome then HmRow.ROW_SCHEMA_SEG else HmRow.ROW_SCHEMA_V2} row, and {reg.value}'s registered rows are {HmRow.ROW_SCHEMA} rows"
  if !reg.fits c.ports[p]!.words then
    throw s!"registered reads: port {reg.name}'s rows are {c.ports[p]!.words} u16 words, not {reg.value}'s registered rows of {reg.words.getD 0} {reg.wordBits}-bit leaves"
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
      match reg.reads[u]? with
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

/-- One port of an inner statement's description: a row of the circuit, an `hm96-sha512/row/v1` row of `reg`'s width
(`Port.fits`), whose table is `reg`'s registration in order (as many rows as its leaves, table row `r` at position `r`,
opening the registered root through its path), every instance reading a row of it, and every row of it read. -/
def checkDescriptionPort (c : Circuit) (pub : Public) (t : Tables) (reg : Port) : V Unit := do
  let np := c.ports.size
  let some p := portOf c reg.name
    | throw s!"registered reads: {reg.name} is not a row of the circuit (an input port, or an output port's row)"
  if c.ports[p]!.v2 then
    throw s!"registered reads: port {reg.name} is a {if c.ports[p]!.segs.isSome then HmRow.ROW_SCHEMA_SEG else HmRow.ROW_SCHEMA_V2} row, and {reg.value}'s registered rows are {HmRow.ROW_SCHEMA} rows"
  if !reg.fits c.ports[p]!.words then
    throw s!"registered reads: port {reg.name}'s rows are {c.ports[p]!.words} u16 words, not {reg.value}'s registered rows of {reg.words.getD 0} {reg.wordBits}-bit leaves"
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

end Registered

/-- An hm96 statement's registered reads (`Registered.check`). A hidden-output statement's registered ports may be its
output rows as well as its input rows (§16.13): an output row's table is its instances' own rows, each opened against the
registered root at the position its unit reads, so the instance's output is that row of the registered value. -/
def checkRegistered (st : Stmt) (registered : Option (Array Registered.Port)) : V Unit :=
  Registered.check st.c st.pub registered (Registered.derived st.partition)

end Flock
