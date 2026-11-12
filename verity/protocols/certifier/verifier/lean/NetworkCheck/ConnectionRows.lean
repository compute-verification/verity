module

public import NetworkCheck.Period
public import Flock.FV3
public import Identity.Tagged
meta import NetworkCheck.Period
meta import Flock.FV3
meta import Identity.Tagged

/-!
# The connection rows' head

What the network certifier signs, once, for an execution phase's connection rows (`connection_rows.py`), and the
verifier's read of its registration: the tables a schedule with lengths rows lists (`tables_of`: the lengths table, the
fails table, then each content class's request, response and end tables, `content_rows.tables_of`), each table's
frame-v3-sha512 domain (`domain`), the head and its digest (`Head.encode`, `Head.digest`) and the registration's read
(`check_registration`), which under the lengths rows' form 2 takes a class's rows as the registration's count of its
held rows: at most the releases of an execution phase, one count for a class's three tables, and the zero root for a
table of no row.

A schedule with lengths rows is `Tabled`: the period check's schedule (`Period.Schedule`, as `Period.scheduleOf` reads
it) beside the lengths rows' form and the content classes, which `Tabled.check` holds to what `warden.WardenParams` and
`content_rows.classes` take. Its digest is a parameter here, as the head's `parameters`.

Code: it imports the period check (`NetworkCheck.Period`) and what that imports, and nothing else.
-/

@[expose] public section

namespace NetworkCheck.ConnectionRows

open Lean (Json toJson)
open Flock (V)

/-! ## Tags and sizes -/

def domainTag : String := "verity-network-certifier/connection-rows-domain/v1"
def headTag : String := "verity-network-certifier/connection-rows/v1"
def registrationFormat : String := "verity-network-certifier/connection-rows-registration/v1"

def lengthsName : String := "lengths"
def failsName : String := "fails"
/-- A lengths slot's bytes: u32 N + 1 ‖ u32 bytes ‖ u32 n. -/
def lengthsSlot : Nat := 12
/-- A fails slot's bytes, and an end row's: u16 class ‖ u16 at. -/
def failsSlot : Nat := 4
def endBytes : Nat := failsSlot
/-- A count's and an id's bytes: a u32. -/
def u32Bytes : Nat := 4
/-- The words the lengths row's proof block returns before the row: its verdict and its failures. -/
def blockWords : Nat := 2
/-- The words the response tie's proof block returns before the row: its verdict. -/
def tieWords : Nat := 1
def digestBytes : Nat := 64
/-- The root of a table of no row. -/
def noRowsRoot : ByteArray := ⟨Array.replicate digestBytes 0⟩
/-- The widest public size class of a proof block's outputs, in bits (`Spot.Partition.UNIT_CLASS_MAX`). -/
def unitClassMax : Nat := 262144
/-- A class's least size, of prompt ids or of ids released. -/
def minSize : Nat := 2
/-- The most classes a schedule lists: the head counts its tables in a u16, three a class after the first two. -/
def mostClasses : Nat := (2 ^ 16 - 3) / 3
/-- The most slots a lengths row has: its proof block returns the whole row after its words (`check_slots`). -/
def mostSlots : Nat := (unitClassMax - blockWords * Period.outBits) / (8 * lengthsSlot)
/-- The most ids a response row has: its tie's proof block returns the whole row after its verdict, and a class's size
is a power of two (`content_rows.most_ids`). -/
def mostIds : Nat :=
  let fit := (unitClassMax - tieWords * Period.outBits) / (8 * u32Bytes) - 1
  if 1 ≤ fit then 1 <<< fit.log2 else 0

/-- A class's size: a power of two of at least `minSize` (`content_rows.check_size`). -/
def isSize (x : Nat) : Bool := minSize ≤ x && x &&& (x - 1) == 0

/-! ## The id request's length -/

/-- `active.ID_REQUEST_HEAD`. -/
def idRequestHead : String :=
  "POST /v1/completions HTTP/1.1\r\nhost: server\r\ncontent-type: application/json\r\ncontent-length:"
def idDigits : Nat := 10
def idNameBytes : Nat := 64
/-- A prompt id's place in an id request: `[` or `,`, then the id in `idDigits` characters. -/
def idField : Nat := 1 + idDigits
/-- The body's offset in an id request at offset 0: the head, the content-length, CRLF CRLF. -/
def bodyAt : Nat := idRequestHead.utf8ByteSize + idDigits + 4

/-- The body's pieces before its prompt ids (`active.id_request`), each a key and its value's width. -/
def bodyKeys : List (String × Nat) :=
  [("{\"max_tokens\":", idDigits), (",\"temperature\":", 8), (",\"top_p\":", 8), (",\"seed\":", idDigits),
   (",\"stop\":[],\"stream\":true", 0), (",\"model\":", idNameBytes), (",\"user\":", idNameBytes), (",\"prompt\":", 0)]

def bodyPieces : List Nat := bodyKeys.map fun (k, w) => k.utf8ByteSize + w

/-- The first prompt id's place in an id request none of whose earlier pieces moved. -/
def promptAt : Nat := bodyAt + bodyPieces.sum

/-- The bytes of `active.id_request` at offset 0 for a prompt of `P ≥ 1` ids in `B`-byte cells, whatever its fields:
the head, then each piece, with spaces before one that would straddle a cell's payload; none for a cell with no room
for the head or a piece. -/
def idRequestBytes (B P : Nat) : Option Nat :=
  let room := B - Period.cellHeader
  let pieces := bodyPieces ++ List.replicate P idField ++ [2]
  if room < bodyAt || pieces.any (room < ·) then none
  else some (pieces.foldl
    (fun a p => a + (if a / room != (a + p - 1) / room then (room - a % room) % room else 0) + p) bodyAt)

/-- `content_rows.request_room`: the bytes of the longest request a class of `P` prompt ids holds in `B`-byte cells. -/
def requestRoom (B P : Nat) : V Nat := do
  unless isSize P do throw s!"a class's size of prompt ids is a power of two of at least {minSize}, not {P}"
  match idRequestBytes B P with
  | some n => pure n
  | none => throw s!"a {B}-byte cell has no room for an id request's head or a field"

/-- `content_rows.request_bytes`: a request row's bytes, its count and `requestRoom` bytes to a whole 16-bit word. -/
def requestBytes (B P : Nat) : V Nat := do
  let size := u32Bytes + (← requestRoom B P)
  let word := Period.outBits / 8
  pure (size + (word - size % word) % word)

/-- `content_rows.response_bytes`: a response row's bytes in a class of `O` ids released, its count and its ids. -/
def responseBytes (O : Nat) : V Nat := do
  unless isSize O do throw s!"a class's size of ids released is a power of two of at least {minSize}, not {O}"
  unless O ≤ mostIds do throw s!"a response row is at most {mostIds} ids, not O={O}"
  pure (u32Bytes * (O + 1))

/-! ## A schedule with lengths rows -/

/-- A content class: its prompt ids P, its ids released O and, under the lengths rows' form 1, its tables' rows. -/
structure Class where
  promptIds : Nat
  ids : Nat
  rows : Option Nat := none

/-- A schedule with lengths rows: the schedule, the lengths rows' form (`WardenParams.lengths`, 1 or 2) and the content
classes (`WardenParams.content`), or none. -/
structure Tabled where
  schedule : Period.Schedule
  lengths : Nat
  content : Option (Array Class) := none

/-- Whether a class's tables hold a row a release filed in it and no other, so that their rows are a count: the
lengths rows' form 2 (`content_rows.counted`). -/
def Tabled.counted (t : Tabled) : Bool := t.lengths == 2

/-- K, a lengths row's slots: the most requests released in one slot stride (`WardenParams.stride_releases`),
`lanes · min(⌈every / (D + 2 − ingress_buckets)⌉, 1 + ⌈(every − 1) / (D + 1)⌉)`. -/
def strideReleases (s : Period.Schedule) : Nat :=
  let l := s.delay + 2 - s.ingressBuckets
  s.lanes * min ((s.every + l - 1) / l) (1 + (s.every - 1 + s.delay) / (s.delay + 1))

/-- R, a bound on the requests released in one execution phase (`WardenParams.phase_releases`):
`lanes · ⌈(T + ingress_buckets − 1) / (D + 1)⌉`. -/
def phaseReleases (s : Period.Schedule) : Nat :=
  s.lanes * ((s.window + s.ingressBuckets - 1 + s.delay) / (s.delay + 1))

/-- A lengths or fails table's rows: one a slot stride. -/
def strides (s : Period.Schedule) : Nat := s.window / s.every

/-- The most buckets from a connection's admission to its last egress cell's (`WardenParams.span`). -/
def span (s : Period.Schedule) : Nat := s.delay + (s.egressCells - 1) / s.slots

/-- What `warden.WardenParams` takes of a schedule with lengths rows beyond `Period.scheduleOf`'s, and
`content_rows.classes` of its classes. -/
def Tabled.check (t : Tabled) : V Unit := do
  let s := t.schedule
  let payload := s.cellBytes - Period.cellHeader
  unless t.lengths == 1 || t.lengths == 2 do throw s!"lengths is 1 or 2, the lengths rows' forms, not {t.lengths}"
  unless s.ingressBuckets ≤ s.delay do throw "with lengths rows ingress_buckets is at most the delay"
  unless span s ≤ s.window && s.egressCells ≤ 2 ^ 16 do
    throw "with lengths rows a connection ends within a window of its admission and its egress cells count in a u16"
  unless promptAt + idField ≤ payload do
    throw s!"at {s.cellBytes}-byte cells an id request's first prompt field isn't in its first cell"
  let K := strideReleases s
  unless 1 ≤ K && K ≤ mostSlots do throw s!"a lengths row is 1 <= K <= {mostSlots} slots, not K={K}"
  if let some classes := t.content then
    unless 0 < classes.size && classes.size ≤ mostClasses do throw s!"content is 1 to {mostClasses} classes"
    for (c, i) in classes.toList.zipIdx do
      match t.counted, c.rows with
      | true, some _ => throw s!"content class {i} is (P, O) under the lengths rows' form 2"
      | false, none => throw s!"content class {i} is (P, O, rows) under the lengths rows' form 1"
      | _, _ => pure ()
      let _ ← responseBytes c.ids
      let _ ← requestRoom s.cellBytes c.promptIds
      unless c.promptIds ≤ s.ingressCells * payload do
        throw s!"content class {i} has P={c.promptIds} prompt ids, more than a request's bytes"
      if let some rows := c.rows then
        unless 0 < rows && rows ≤ strides s * K do
          throw s!"content class {i} has 1 to {strides s * K} rows, the lengths table's slots, not {rows}"

/-! ## The tables -/

/-- A table the head lists: its name, its rows (under form 2 the most a class's table has) and a row's bytes. -/
structure Table where
  name : String
  rows : Nat
  rowBytes : Nat

/-- `connection_rows.tables_of`: the tables a schedule's head lists, in its order. -/
def Tabled.tables (t : Tabled) : V (Array Table) := do
  t.check
  let s := t.schedule
  let K := strideReleases s
  let mut out : Array Table :=
    #[⟨lengthsName, strides s, K * lengthsSlot⟩, ⟨failsName, strides s, K * failsSlot⟩]
  for (c, i) in (t.content.getD #[]).toList.zipIdx do
    let rows := if t.counted then phaseReleases s else c.rows.getD 0
    out := out.push ⟨s!"request.{i}", rows, ← requestBytes s.cellBytes c.promptIds⟩
    out := out.push ⟨s!"response.{i}", rows, ← responseBytes c.ids⟩
    out := out.push ⟨s!"end.{i}", rows, endBytes⟩
  pure out

/-- Of the table at place `k` of `Tabled.tables` (`connection_rows._held`): whether its rows are a count of held rows,
and whether it then has the count of the table listed before it, as a class's response and end tables have its request
table's. A class's tables are three from place 2. -/
def Tabled.held (t : Tabled) (k : Nat) : Bool × Bool :=
  let counted := t.counted && 2 ≤ k
  (counted, counted && (k - 2) % 3 != 0)

/-! ## The domain and the head -/

/-- `connection_rows.domain`'s id: the frame-v3-sha512 domain of one table of execution phase `w` of the link, in the
session, under the schedule whose digest is `sd`. -/
def domainId (sd session : ByteArray) (link : String) (w : Nat) (table : String) (rows rowBytes : Nat) :
    V ByteArray := do
  let binding ← Identity.identityDigestSha512 domainTag (Json.mkObj [
    ("parameters", .str (Flock.hexOf sd)), ("session", .str (Flock.hexOf session)), ("link", .str link),
    ("window", toJson w), ("table", .str table), ("rows", toJson rows), ("row_bytes", toJson rowBytes),
    ("row_schema", .str Period.rowSchema)])
  pure (Flock.FV3.Sha512.Domain.make binding (-1) 0 rows).id

/-- One table as the head lists it (`connection_rows.Entry`). -/
structure Entry where
  name : String
  rows : Nat
  rowBytes : Nat
  domain : ByteArray
  root : ByteArray

/-- `u16be len(name) ‖ name ‖ u32be rows ‖ u32be row bytes ‖ domain ‖ root`. -/
def Entry.encode (e : Entry) : V ByteArray := do
  unless e.domain.size == digestBytes && e.root.size == digestBytes do
    throw "a table's domain and root are 64 bytes"
  pure ((← text e.name "table") ++ (← uBE 4 e.rows "rows") ++ (← uBE 4 e.rowBytes "row bytes") ++ e.domain ++
    e.root)

/-- What the network certifier signs for an execution phase's connection rows (`connection_rows.Head`). -/
structure Head where
  link : String
  window : Nat
  parameters : ByteArray
  tables : Array Entry

/-- `u16be len(link) ‖ link ‖ u64be window ‖ parameters ‖ u16be tables ‖ each entry`, of a head that lists each of
its tables once, and at least one. -/
def Head.encode (h : Head) : V ByteArray := do
  unless h.parameters.size == digestBytes do throw "the parameters digest is 64 bytes"
  let names := h.tables.toList.map (·.name)
  unless !names.isEmpty && decide names.Nodup do throw "a head lists each of its tables once, and at least one"
  let start := (← text h.link "link") ++ (← uBE 8 h.window "window") ++ h.parameters ++
    (← uBE 2 h.tables.size "tables")
  pure ((← h.tables.mapM Entry.encode).foldl (· ++ ·) start)

/-- `Head.digest`: `tagged_sha512(HEAD_TAG, H)`. -/
def Head.digest (h : Head) : V ByteArray := do Identity.taggedSha512 headTag (← h.encode)

def Head.table (h : Head) (name : String) : Option Entry := h.tables.find? (·.name == name)

/-! ## The registration -/

def registrationKeys : List String :=
  ["certificate", "format", "link", "parameters", "row_schema", "tables", "window", "word_bits"]

def tableKeys : List String := ["domain", "layout", "name", "root", "rows", "value", "words"]

/-- What a table's registration says of its rows' order (`connection_rows._layout`), its keys in order: of the table at
place `k` of `Tabled.tables` at execution phase `w`. -/
def Tabled.layout (t : Tabled) (w k : Nat) : List (String × Nat) :=
  let s := t.schedule
  let first := w * s.window
  if k < 2 then
    [("every", s.every), ("first_bucket", first), ("slot_bytes", if k == 0 then lengthsSlot else failsSlot),
     ("slots", strideReleases s), ("strides", strides s)]
  else match (t.content.getD #[])[(k - 2) / 3]? with
    | some c => [("buckets", s.window), ("class", (k - 2) / 3), ("first_bucket", first), ("ids", c.ids),
                 ("prompt_ids", c.promptIds)]
    | none => []

/-- `connection_rows.check_registration` before its certificate: the head the registration states, under the schedule
whose digest is `sd`, with the certificate's key, digest and signature, or what fails. Under form 2 a class's rows are
the registration's count: at most `phaseReleases`, one for a class's three tables; and a table of no row has the zero
root. Each domain is derived at the rows the head then lists. -/
def Tabled.read (t : Tabled) (sd session : ByteArray) (link : String) (w : Nat) (doc : Json) :
    V (Head × ByteArray × ByteArray × ByteArray) := do
  let ctx := "registration"
  let listed ← t.tables
  Period.exactly doc registrationKeys ctx
  unless (← Flock.J.str doc "format" ctx) == registrationFormat && (← Flock.J.str doc "link" ctx) == link &&
      (← Flock.J.nat doc "window" ctx) == w && (← Flock.J.str doc "parameters" ctx) == Flock.hexOf sd &&
      (← Flock.J.nat doc "word_bits" ctx) == Period.outBits &&
      (← Flock.J.str doc "row_schema" ctx) == Period.rowSchema do
    throw s!"{ctx}: not the link's connection rows at window {w} under the schedule"
  let got ← Flock.J.arr doc "tables" ctx
  unless got.size == listed.size do throw s!"{ctx}: it lists the schedule's {listed.size} tables, and no other"
  let mut entries : Array Entry := #[]
  for ((j, x), k) in (got.toList.zip listed.toList).zipIdx do
    let what := s!"{ctx}: the {x.name} table"
    Period.exactly j tableKeys what
    unless (← Flock.J.str j "name" what) == x.name do throw s!"{what}: another name"
    let rows ← Flock.J.nat j "rows" what
    let (counted, shared) := t.held k
    if counted then
      unless rows ≤ x.rows do
        throw s!"{what}: rows {rows}, more than the {x.rows} an execution phase of the schedule releases"
      if shared then
        unless entries.back?.map (·.rows) == some rows do
          throw s!"{what}: rows {rows}, not its class's count: a class's three tables hold a row a release filed in it"
    else
      unless rows == x.rows do throw s!"{what}: rows {rows}, and the schedule's {x.rows}"
    let dom ← domainId sd session link w x.name rows x.rowBytes
    unless (← Flock.J.str j "value" what) == s!"{link}/{x.name}/{w}" &&
        (← Flock.J.nat j "words" what) == x.rowBytes * 8 / Period.outBits &&
        (← Flock.J.str j "domain" what) == Flock.hexOf dom do
      throw s!"{what}: not the schedule's value, words and domain"
    let layout := t.layout w k
    let l ← Flock.J.get j "layout" what
    Period.exactly l (layout.map (·.1)) s!"{what}: layout"
    for (key, v) in layout do
      unless (← Flock.J.nat l key s!"{what}: layout") == v do
        throw s!"{what}: layout: {key} is not the schedule's {v}"
    let root ← Period.hexAt j "root" what
    unless root.size == digestBytes do throw s!"{what}: a root is {digestBytes} bytes"
    unless rows != 0 || root == noRowsRoot do throw s!"{what}: a table of no row has the zero root"
    entries := entries.push ⟨x.name, rows, x.rowBytes, dom, root⟩
  let c ← Flock.J.get doc "certificate" ctx
  Period.exactly c ["digest", "key", "signature"] s!"{ctx}: certificate"
  let head : Head := { link, window := w, parameters := sd, tables := entries }
  pure (head, ← Period.hexAt c "key" ctx, ← Period.hexAt c "digest" ctx, ← Period.hexAt c "signature" ctx)

/-- `connection_rows.check_registration`: the head of the link's connection rows at execution phase `w`, which the
registration states and its certificate signs under the deployment key in the session, or the refusal. -/
def registered (p : Period.Params) (t : Tabled) (sd : ByteArray) (w : Nat) (doc : Json) : Period.Found Head := do
  let (head, key, digest, signature) ← match t.read sd p.session p.link w doc with
    | .ok x => pure x
    | .error e => Period.refuse e
  let .ok d := head.digest | Period.refuse "registration: the head doesn't encode"
  unless digest == d do Period.refuse "registration: the certificate doesn't sign its head"
  unless Period.certifiesOn p p.link .egress w
      { keyId := key, link := p.link, direction := .egress, window := w, digest, signature } do
    Period.refuse "registration: not under the deployment key in the session"
  pure head

/-! ## Pins against the Python -/

section Example

private def toySchedule : Period.Schedule :=
  { bucketMs := 100, cellBytes := 512, delay := 5, egressCells := 9, every := 2, ingressBuckets := 3,
    ingressCells := 4, lanes := 2, queue := 2, slots := 2, window := 16 }

/-- The toy schedule under form 2, and its `WardenParams.digest`. -/
private def held : Tabled :=
  { schedule := toySchedule, lengths := 2,
    content := some #[⟨4, 4, none⟩, ⟨16, 4, none⟩, ⟨128, 8, none⟩, ⟨128, 8, none⟩] }

private def heldDigest : String :=
  "fe09d8179188559f4744e18efbcc9371ecdf7bbaae4e2cc1fbe26e4df92299a4\
    a378a5a47d2b1de609f79f8c81b54362d5dbb2e19b488c143839714b5f972069"

/-- The toy schedule under form 1, each class with rows of its own. -/
private def own : Tabled :=
  { schedule := toySchedule, lengths := 1,
    content := some #[⟨4, 4, some 2⟩, ⟨16, 4, some 3⟩, ⟨128, 8, some 2⟩, ⟨128, 8, some 2⟩] }

private def session : ByteArray := ⟨(Array.range 32).map (·.toUInt8)⟩

private def listed (t : Tabled) : Option (List (String × Nat × Nat)) :=
  t.tables.toOption.map fun ts => ts.toList.map fun x => (x.name, x.rows, x.rowBytes)

private def sizes : List (String × Nat) :=
  [("lengths", 24), ("fails", 8), ("request.0", 418), ("response.0", 20), ("end.0", 4), ("request.1", 554),
   ("response.1", 20), ("end.1", 4), ("request.2", 1802), ("response.2", 36), ("end.2", 4), ("request.3", 1802),
   ("response.3", 36), ("end.3", 4)]

private def domainHex (table : String) (rows rowBytes : Nat) : Option String := do
  let sd ← (Flock.unhex heldDigest).toOption
  (domainId sd session "responses" 0 table rows rowBytes).toOption.map Flock.hexOf

/-- A head of the form 2 schedule: classes 0 and 2 hold two rows and classes 1 and 3 none; the table at place `k` has
the root of 64 bytes `k + 1`, or the zero root where it has no row. -/
private def head : V Head := do
  let sd ← Flock.unhex heldDigest
  let tables ← (← held.tables).toList.zipIdx.mapM fun (x, k) => do
    let rows := if k < 2 then x.rows else if (k - 2) / 3 % 2 == 0 then 2 else 0
    let root : ByteArray := if rows == 0 then noRowsRoot else ⟨Array.replicate digestBytes (k + 1).toUInt8⟩
    pure ({ name := x.name, rows, rowBytes := x.rowBytes, root,
            domain := ← domainId sd session "responses" 0 x.name rows x.rowBytes } : Entry)
  pure { link := "responses", window := 0, parameters := sd, tables := tables.toArray }

#guard idRequestHead.utf8ByteSize == 92 && bodyAt == 106 && bodyPieces == [24, 23, 17, 18, 24, 73, 72, 10] &&
  promptAt == 367 && bodyAt + 14 + idDigits == Period.maxTokensEnd
#guard mostIds == 4096 && mostClasses == 21844 && mostSlots == 2730
#guard isSize 2 && isSize 4096 && !isSize 0 && !isSize 1 && !isSize 3 && !isSize 12
-- `content_rows.request_room` at cells of 128, 256 and 4096 bytes, and at cells an id request doesn't fit
#guard [128, 256, 4096].map (fun B => [2, 4, 16, 128, 1024].map (idRequestBytes B)) ==
  [[some 463, some 489, some 630, some 1961, some 12627], [some 426, some 448, some 584, some 1841, some 11902],
   [some 391, some 413, some 545, some 1777, some 11641]]
#guard idRequestBytes 32 2 == none && idRequestBytes 64 2 == none
#guard (requestBytes 128 2).toOption == some 468 && (requestBytes 256 16).toOption == some 588 &&
  (requestBytes 4096 1024).toOption == some 11646 && (requestBytes 512 3).toOption == none
#guard (responseBytes 4).toOption == some 20 && (responseBytes 4096).toOption == some 16388 &&
  (responseBytes 8192).toOption == none && (responseBytes 6).toOption == none
-- K and R of the toy schedule, and `connection_rows.tables_of` under each form
#guard strideReleases toySchedule == 2 && phaseReleases toySchedule == 6 && strides toySchedule == 8
#guard listed own == some (sizes.map fun (n, b) =>
  (n, if n == "lengths" || n == "fails" then 8 else if n.endsWith ".1" then 3 else 2, b))
#guard listed held == some (sizes.map fun (n, b) => (n, if n == "lengths" || n == "fails" then 8 else 6, b))
-- what `WardenParams` refuses of a schedule with lengths rows, and `content_rows.classes` of its classes
#guard (({ held with lengths := 3 } : Tabled).check).toOption.isNone
#guard (({ held with content := some #[] } : Tabled).check).toOption.isNone
#guard (({ held with content := some #[⟨4, 4, some 2⟩] } : Tabled).check).toOption.isNone
#guard (({ own with content := some #[⟨4, 4, none⟩] } : Tabled).check).toOption.isNone
#guard (({ own with content := some #[⟨4, 4, some 17⟩] } : Tabled).check).toOption.isNone
#guard (({ own with content := some #[⟨4, 4, some 16⟩] } : Tabled).check).toOption.isSome
#guard (({ held with content := some #[⟨4, 8192, none⟩] } : Tabled).check).toOption.isNone
#guard (({ held with content := some #[⟨1024, 4, none⟩] } : Tabled).check).toOption.isSome
#guard (({ held with content := some #[⟨2048, 4, none⟩] } : Tabled).check).toOption.isNone
#guard (({ held with schedule := { toySchedule with cellBytes := 386 } } : Tabled).check).toOption.isNone
#guard (({ held with schedule := { toySchedule with cellBytes := 387 } } : Tabled).check).toOption.isSome
#guard (({ held with schedule := { toySchedule with ingressBuckets := 6 } } : Tabled).check).toOption.isNone
#guard (({ held with schedule := { toySchedule with egressCells := 25 } } : Tabled).check).toOption.isNone
#guard (({ held with schedule := { toySchedule with egressCells := 24 } } : Tabled).check).toOption.isSome
-- `connection_rows.domain(...).domain_id` of four tables of the form 2 schedule, one of no row
#guard domainHex "lengths" 8 24 == some
  "05bf64b22374df3145110cbd2c139c6dc9eb1143199b0e038377cd4037b1f412\
    f54e9abb5d63d890e7344fcf5fb8999570c8866ba7fc1b75aeb046e338df3569"
#guard domainHex "end.2" 2 4 == some
  "b38f5f40a9438b556b0c5e8f58c3fd9290ac080e7ed4e18c5a7b242b5fe05b71\
    f902f0793a50031e8a44e612f28ca2ef458be39826fbb10d2cee810324b33a93"
#guard domainHex "request.2" 2 1802 == some
  "925f015c6d32f9eee5ea5161663432b3cc9b38ebef783a614a4f10c227f466bb\
    5ed06b286850fd38c10fe72fec89a9b589e58d4d6dee8f12b2b7d774725ccd31"
#guard domainHex "response.1" 0 20 == some
  "4d6590d516239870ad37f01743ccccfe0889a422102513f2f978f998044ab339\
    862b3e6076cae65aedb4a86312a7541e0f81be956fae853f15d582b4742b0728"
-- `Head.encode` (its length and its SHA-512) and `Head.digest` of that head
#guard (head.bind Head.encode).toOption.map (·.size) == some 2125
#guard (head.bind Head.encode).toOption.map (fun b => Flock.hexOf (Flock.Sha512.hash b)) == some
  "4f2b1134d920d27227124e0ddfc26c9830f96c39ce1f75fa085aec6248aff5dc\
    e8b38a8cd78cb476cc9a2547bf55d124ce6b6de9bf6b5fcf14c94238c2f9b75b"
#guard (head.bind Head.digest).toOption.map Flock.hexOf == some
  "42e06e201e0aa95f640add398ae7bfa8e52b42746890306bcda039cca889aadc\
    2be1f1de6fa6356040606407609bdc7cc411b3d865761ff8004bc0455ee3444a"
-- a head that lists a table twice, or none, doesn't encode
#guard (head.bind fun h => ({ h with tables := h.tables.push h.tables[0]! } : Head).encode).toOption.isNone
#guard (head.bind fun h => ({ h with tables := #[] } : Head).encode).toOption.isNone
#guard held.held 0 == (false, false) && held.held 2 == (true, false) && held.held 3 == (true, true) &&
  held.held 4 == (true, true) && held.held 5 == (true, false) && own.held 3 == (false, false)

end Example

end NetworkCheck.ConnectionRows
