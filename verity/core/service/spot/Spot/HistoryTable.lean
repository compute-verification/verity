/-
History tables (`verity/ml/commitments/history.py`): a table of rows that other proof blocks output, read at hidden
positions by table reads (`EmbeddingRead_v1`). The verifier computes each table's root itself, from the writers' public
`b ‖ c` in the order its own program gives; the root the prover registers is only checked against it, so the prover
can't register rows of its choosing.

* `CAPACITY`: every table's tree is pinned at `2^17` ranks, whatever its length (`len`, its written rows), so its domain,
  depth and positions are public structure and no table's shape tells its length;
* `domainAt cap program table words`: `CommitmentDomain(B, -2, RangeIndexedDomain(0, cap), hash="sha512").domain_id`, `B`
  the 64 bytes of `identity_digest_sha512("verity/one-stage/history-table/v1", {"capacity", "program", "schema",
  "table", "words"})`: `program` the program's digest (hex), `table` the table's name in it, `words` its rows' 16-bit
  words and `schema` `hm96-sha512/row-seg/v1`, the format the reads open (`EmbeddingRead_v1`);
* `placeholder words`: the `b ‖ c` of the row of `words` zero words under the zero salt, its inner digest
  `sha512/row-seg/v1`'s. It is known content, so a rank from `len` up opens only as that row;
* `leavesAt`, `rootAt`: rank `r`'s frame leaf (`FV3.Sha512.leafAs` at `hm96-sha512/row-seg/v1`) over the tree leaf of
  `bcs[r]` below `bcs.size`, else of the placeholder's, and the frame-v3 root over all `cap` of them (`FV3.root`, which
  pads nothing at a power of two);
* `Run`, `order`: the program's description of who writes each position, checked at the size of that description and
  expanded to one `Source` a position: a run of positions `start …` either written by units `base + Σ d_k · step_k` of
  the table's writer template at its output (`d` the digits of the position's offset in `radix`, the first the slowest,
  as `Spot.Wires.Wire.writerAt` reads a wire), or carried as rows `row …` of a value the verifier holds (a window edge's
  rows of an earlier window). The runs must tile `[0, len)`, every unit must lie in its template and every held row in
  its value; a run of another template or output is refused, so a position's writer is the table's own;
* `checkAt cap program table words bcs registered`: the refusals, in order: a capacity that is not a power of two of at
  least 2, a row of no words or of `2^64` bits or more, more rows than the capacity, a `b ‖ c` that is not 128 bytes, a
  registered root that is not 64 bytes, then a registered root other than `rootAt`'s. `check` is it at `CAPACITY`;
* `memberStrings`, `tableStrings`: the `bcs` `checkAt` reads, from the writer's members' public bodies (each unit's
  `b ‖ c` of its rows, or a shared-row body's tables and refs) at the positions `order` gives, and from the strings the
  verifier holds of the table's earlier windows for its held runs;
* `tableRoots`, `rootsOf`: each table's registered root held to `checkAt` over those strings, from members whose public
  bodies hash to the digests the window's leaf layer binds; `resolve` puts the root and domain id it computed for table
  `H` in place of a supplied value's sources `{"history": "H/root"}` and `{"history": "H/domain"}`, so a reader's
  `TableRead` takes the root the verifier computed, never one the prover or the verifier's Python supplies.

What a read that opens against this root returns is stated by `Hash.Guarantees.HistoryReadBinding`: the row behind
the `b ‖ c` at its index, or SHA-512 collides. That guarantee is stated over `Hash.HistoryTable.root`, and
`Properties.Spot.HistoryRootIsModel` says a root `checkAt` accepts is that root, at SHA-512 under the table's domain.
-/
module

public import Flock.FV3
public import Flock.Hm96
public import Flock.HmRow
public import Flock.Canon
public import Spot.Json
public import Spot.Wires

@[expose] public section

namespace Spot.HistoryTable

open Lean (Json)
open Flock (V)

/-- Every table's ranks: its tree's leaves, written or placeholder. -/
def CAPACITY : Nat := 2 ^ 17
def DOMAIN_TAG : String := "verity/one-stage/history-table/v1"
/-- The format the table's rows are committed and read in. -/
def SCHEMA : String := Flock.HmRow.ROW_SCHEMA_SEG
/-- A `b ‖ c`'s bytes, and a digest's. -/
def BC_BYTES : Nat := 128
def DIGEST_BYTES : Nat := 64
/-- The placeholder's salt: 192 zero bytes. -/
def SALT_BYTES : Nat := 192
/-- The owner `-2` of a domain the verifier derives. -/
def OWNER : Int := -2

/-- A power of two of at least 2: a capacity whose tree pads nothing. -/
def capOk (cap : Nat) : Bool := 2 ≤ cap && cap &&& (cap - 1) == 0

/-- A row of `words` 16-bit words is at least one word and its bits fit a row leaf's u64 length. -/
def wordsOk (words : Nat) : Bool := 0 < words && 16 * words < 2 ^ 64

/-- `domainAt`'s binding: `identity_digest_sha512(DOMAIN_TAG, {"capacity", "program", "schema", "table", "words"})`. -/
def binding (cap : Nat) (program table : String) (words : Nat) : V ByteArray := do
  let manifest := Json.mkObj [("capacity", Lean.toJson cap), ("program", Json.str program), ("schema", Json.str SCHEMA),
    ("table", Json.str table), ("words", Lean.toJson words)]
  return Flock.FV3.Sha512.taggedSha512 DOMAIN_TAG (← Flock.canonUtf8 manifest).toUTF8

/-- The table's frame-v3-sha512 domain id at capacity `cap`. -/
def domainAt (cap : Nat) (program table : String) (words : Nat) : V ByteArray := do
  return (Flock.FV3.Sha512.Domain.make (← binding cap program table words) OWNER 0 cap).id

/-- The table's domain id at `CAPACITY`. -/
def domain (program table : String) (words : Nat) : V ByteArray := domainAt CAPACITY program table words

/-- The placeholder's `b ‖ c`: `words` zero words as a `sha512/row-seg/v1` bit row, under the zero salt. -/
def placeholder (words : Nat) : ByteArray :=
  Flock.Hm96.Default512.commitString
    (Flock.HmRow.segDigest (16 * words) (ByteArray.mk (Array.replicate (2 * words) 0)))
    (ByteArray.mk (Array.replicate SALT_BYTES 0))

/-- Rank `r`'s frame leaf: `bcs[r]`'s tree leaf below `bcs.size`, else the placeholder `ph`'s, at `r`. -/
def leafAt (dom : ByteArray) (bcs : Array ByteArray) (phLeaf : ByteArray) (r : Nat) : ByteArray :=
  Flock.FV3.Sha512.leafAs SCHEMA dom r (if h : r < bcs.size then Flock.Hm96.Default512.treeLeaf bcs[r] else phLeaf)

/-- The `cap` frame leaves of a table whose written rows' `b ‖ c` are `bcs` and whose placeholder's is `ph`. -/
def leavesAt (cap : Nat) (dom : ByteArray) (bcs : Array ByteArray) (ph : ByteArray) : Array ByteArray :=
  let phLeaf := Flock.Hm96.Default512.treeLeaf ph
  (Array.range cap).map (leafAt dom bcs phLeaf)

/-- The table's root at capacity `cap`: the frame-v3 root over `leavesAt`. -/
def rootAt (cap : Nat) (dom : ByteArray) (bcs : Array ByteArray) (ph : ByteArray) : ByteArray :=
  Flock.FV3.root dom (leavesAt cap dom bcs ph)

/-! ## Who writes each position -/

/-- Where a run's rows come from: units of the table's writer template at its output (`writer`, `out`), unit
`base + Σ d_k · step_k` for the offset whose digits in `radix` are `d` (the first the slowest), or rows `row …` of a value
the verifier holds (`held`). -/
inductive From where
  | unit (writer out : String) (radix : List Nat) (base : Int) (step : List Int)
  | held (value : String) (row : Nat)
  deriving Repr, BEq

/-- A run of the table's positions `start … start + count - 1` and where they come from. -/
structure Run where
  start : Nat
  count : Nat
  src : From
  deriving Repr, BEq

/-- A position's row: unit `u`'s output row of the writer template, or row `row` of the held value `value`. -/
inductive Source where
  | unit (u : Nat)
  | held (value : String) (row : Nat)
  deriving Repr, BEq, Inhabited

/-- The writer unit of offset `i` of a unit run: `base + Σ d_k · step_k` (`Spot.Wires.accum`, fastest digit first). -/
def unitAt (radix : List Nat) (base : Int) (step : List Int) (i : Nat) : Int :=
  base + Wires.accum (radix.zip step).reverse i

/-- Why a run is refused, checked at the size of its description: its rows' count, its digits, its template and output,
the span of its units (`Spot.Wires.lowOf`, `highOf`) against the writer template's `units`, or its held rows against the
value's (`heldRows`). -/
def runWhy (writer out : String) (units : Nat) (heldRows : String → Option Nat) (r : Run) : Option String :=
  match r.src with
  | .unit w o radix base step =>
    if w != writer || o != out then some s!"a run of positions from {r.start} written by {w}'s {o}, not the table's writer"
    else if radix.length != step.length || radix.any (· == 0) then some s!"a run from {r.start}: its radix and steps"
    else if Wires.prodR (radix.zip step).reverse != r.count then
      some s!"a run from {r.start}: its radix covers no {r.count} positions"
    else
      let ps := (radix.zip step).reverse
      if base + Wires.lowOf ps < 0 || (units : Int) ≤ base + Wires.highOf ps then
        some s!"a run from {r.start}: a writer unit outside {writer}'s {units}"
      else none
  | .held v row => match heldRows v with
    | none => some s!"a run from {r.start}: {v} is no value the verifier holds"
    | some n => if n < row + r.count then some s!"a run from {r.start}: past {v}'s {n} rows" else none

/-- **Each position's row, from the program's runs**: the runs, ordered by their first position, must tile `[0, len)`
with no run empty, and each must pass `runWhy`; then position `p`'s source, one a position. -/
def order (runs : Array Run) (len : Nat) (writer out : String) (units : Nat) (heldRows : String → Option Nat) :
    V (Array Source) := do
  let sorted := runs.qsort (·.start < ·.start)
  let mut at_ := 0
  for r in sorted do
    if r.count == 0 then throw s!"a run from {r.start} of no positions"
    if r.start != at_ then throw s!"the runs leave position {at_} unwritten or write it twice"
    if let some why := runWhy writer out units heldRows r then throw why
    at_ := r.start + r.count
  if at_ != len then throw s!"the runs write {at_} positions, not the table's {len}"
  let mut out : Array Source := Array.emptyWithCapacity len
  for r in sorted do
    match r.src with
    | .unit _ _ radix base step =>
      for i in [0:r.count] do out := out.push (.unit (unitAt radix base step i).toNat)
    | .held v row =>
      for i in [0:r.count] do out := out.push (.held v (row + i))
  return out

/-! ## The check -/

/-- **The registered root is the table's root over its writers' rows**, at capacity `cap`: `checkAt`'s refusals in
order (the module docstring), else `rootAt` over the domain `domainAt` names, the rows `bcs` and the placeholder of
`words` words must be `registered`. -/
def checkAt (cap : Nat) (program table : String) (words : Nat) (bcs : Array ByteArray) (registered : ByteArray) :
    V Unit := do
  if !capOk cap then throw s!"a table's capacity is a power of two of at least 2, not {cap}"
  if !wordsOk words then throw s!"a table's rows are at least one word and under 2^64 bits, not {words} words"
  if cap < bcs.size then throw s!"{bcs.size} rows past the table's capacity of {cap}"
  if bcs.any (·.size != BC_BYTES) then throw s!"a row's b || c is {BC_BYTES} bytes"
  if registered.size != DIGEST_BYTES then throw s!"a registered root is {DIGEST_BYTES} bytes"
  let dom ← domainAt cap program table words
  if rootAt cap dom bcs (placeholder words) != registered then
    throw "the registered root is not the table's root over its writers' rows"

/-- `checkAt` at `CAPACITY`. -/
def check (program table : String) (words : Nat) (bcs : Array ByteArray) (registered : ByteArray) : V Unit :=
  checkAt CAPACITY program table words bcs registered

/-! ## The rows' strings, from the writers' public bodies -/

/-- One member of the table's writer template: its first unit `base` and its `n` units, its rows' names in order
(`ports`, inputs then outputs), each shared-row table's rows (`tables`, none for a body of one row a unit a port) and its
public body, the bytes the leaf layer binds. -/
structure Member where
  base : Nat
  n : Nat
  ports : Array String
  tables : Option (Array (String × Nat)) := none
  body : ByteArray

/-- **Each of a member's units' `b ‖ c` of its row `port`**, from its public body as the circuit prover writes it: every
unit's `b ‖ c`, ports in order, unit-major; or, with shared-row tables, each port's table rows (`n` for a port no table
names), port-major, then each unit's ref into each port's table (u32 little-endian, unit-major), a unit's string the
row its ref names. Refused where the member has no row `port`, the body is not that length, or a ref is past its
table. -/
def memberStrings (m : Member) (port : String) : V (Array ByteArray) := do
  let some j := m.ports.findIdx? (· == port) | throw s!"the member at unit {m.base} has no row {port}"
  let np := m.ports.size
  match m.tables with
  | none =>
    if m.body.size != m.n * np * BC_BYTES then
      throw s!"the member at unit {m.base}: its public body is not {m.n} units' {np} strings"
    return (Array.range m.n).map fun i => m.body.extract ((i * np + j) * BC_BYTES) ((i * np + j + 1) * BC_BYTES)
  | some ts =>
    let counts := m.ports.map fun q => ((ts.find? (·.1 == q)).map (·.2)).getD m.n
    let refsAt := BC_BYTES * counts.foldl (· + ·) 0
    if m.body.size != refsAt + 4 * m.n * np then
      throw s!"the member at unit {m.base}: its public body is not its tables' strings and {m.n} units' refs"
    let off := BC_BYTES * (counts.extract 0 j).foldl (· + ·) 0
    let rows := counts[j]!
    (Array.range m.n).mapM fun i => do
      let r := (Flock.u32OfLE m.body (refsAt + 4 * (i * np + j))).toNat
      if rows ≤ r then throw s!"the member at unit {m.base}: unit {m.base + i}'s ref is past its {port} table"
      return m.body.extract (off + r * BC_BYTES) (off + (r + 1) * BC_BYTES)

/-- **A table's rows' strings in position order** (`order`'s sources): a unit run's from the writer's members, which
must hold its units `[first, first + units)` in order, each its units' `out` strings (`memberStrings`); a held run's
from the strings the verifier holds of the table's earlier windows (`held`, by value). -/
def tableStrings (runs : Array Run) (len : Nat) (writer out : String) (first units : Nat) (members : Array Member)
    (held : Array (String × Array ByteArray)) : V (Array ByteArray) := do
  let heldOf := fun v => (held.find? (·.1 == v)).map (·.2)
  let srcs ← order runs len writer out units fun v => (heldOf v).map (·.size)
  let mut mine : Array ByteArray := #[]
  if srcs.any (· matches .unit _) then
    let mut at_ := first
    for m in members.qsort (·.base < ·.base) do
      if m.base != at_ then throw s!"{writer}'s members leave unit {at_} without a public body"
      mine := mine ++ (← memberStrings m out)
      at_ := at_ + m.n
    if at_ != first + units then throw s!"{writer}'s members hold {at_ - first} of its {units} units"
  srcs.mapM fun
    | .unit u => match mine[u]? with
      | some s => pure s
      | none => throw s!"{writer}'s unit {u} has no string"
    | .held v r => match (heldOf v).bind (·[r]?) with
      | some s => pure s
      | none => throw s!"{v} holds no row {r}"

/-- A history table the verifier's program declares (`interface.History`): the program its domain names, its writer
template and output, its rows' words, its runs and its capacity. -/
structure Spec where
  program : String
  writer : String
  out : String
  words : Nat
  runs : Array Run
  capacity : Nat := CAPACITY

/-- A table's length: its runs' rows. -/
def Spec.length (h : Spec) : Nat := h.runs.foldl (· + ·.count) 0

/-- **The leaf layer** (`rows.leaf_layer`): SHA-512 (hex) of the canonical JSON object of each member's public body's
SHA-512 (hex) by its key. -/
def leafLayer (digests : Array (String × String)) : V String :=
  canonSha512 (Json.mkObj (digests.toList.map fun (k, d) => (k, Json.str d)))

/-- **Each table's root, recomputed from the writers' public bodies the leaf layer binds**: every member's body digest
(`digests`, by key) must make the leaf layer `leaf`, and each table's writer members' bodies must have the digests it
names; then table `name`'s strings (`tableStrings`: its writer's units `[first, first + units)`, and `held name`, the
strings the verifier holds of its earlier windows) and `checkAt` against the root the window registers for it
(`registered`). Each table's `(name, root, domain)`, the root and domain its reads open against. -/
def tableRoots (leaf : String) (digests : Array (String × String))
    (tables : Array (String × Spec × Nat × Nat × Array (String × Member)))
    (held : String → Array (String × Array ByteArray)) (registered : String → Option ByteArray) :
    V (Array (String × ByteArray × ByteArray)) := do
  if (digests.map (·.1)).toList.eraseDups.length != digests.size then throw "a member's key twice among the digests"
  if (← leafLayer digests) != leaf then throw "the members' public bodies' digests are not the leaf layer"
  if (tables.map (·.1)).toList.eraseDups.length != tables.size then throw "a history table named twice"
  tables.mapM fun (name, h, first, units, ms) => do
    for (k, m) in ms do
      match digests.find? (·.1 == k) with
      | some (_, d) =>
        if Flock.hexOf (Flock.Sha512.hash m.body) != d then
          throw s!"history {name}: {k}'s public body is not the one the leaf layer binds"
      | none => throw s!"history {name}: {k} is no member the leaf layer binds"
    let bcs ← tableStrings h.runs h.length h.writer h.out first units (ms.map (·.2)) (held name)
    let some root := registered name | throw s!"history {name}: the window registers no root for it"
    checkAt h.capacity h.program name h.words bcs root
    return (name, root, ← domainAt h.capacity h.program name h.words)

/-! ## The difftest's pieces -/

/-- A JSON list of lowercase hex strings, as bytes. -/
def hexes (j : Json) (what : String) : V (Array ByteArray) := do
  let .arr xs := j | throw s!"{what} is not a list"
  xs.mapM fun
    | .str s => Flock.unhex s
    | _ => throw s!"{what}: not a hex string"

def natField (j : Json) (k : String) : V Nat :=
  match nat? (getD j k) with
  | some n => pure n
  | none => throw s!"{k} is not a natural number"

def strField (j : Json) (k : String) : V String :=
  match getD j k with
  | .str s => pure s
  | _ => throw s!"{k} is not a string"

/-- `{"capacity", "program", "table", "words", "bcs"}` to `{"domain", "placeholder", "root"}` (hex): the table's domain,
placeholder and root at that capacity, or `checkAt`'s refusals but the last (no registered root). -/
def rootPiece (j : Json) : V Json := do
  let cap ← natField j "capacity"
  let words ← natField j "words"
  let program ← strField j "program"
  let table ← strField j "table"
  let bcs ← hexes (getD j "bcs") "bcs"
  if !capOk cap then throw s!"a table's capacity is a power of two of at least 2, not {cap}"
  if !wordsOk words then throw s!"a table's rows are at least one word and under 2^64 bits, not {words} words"
  if cap < bcs.size then throw s!"{bcs.size} rows past the table's capacity of {cap}"
  if bcs.any (·.size != BC_BYTES) then throw s!"a row's b || c is {BC_BYTES} bytes"
  let dom ← domainAt cap program table words
  let ph := placeholder words
  return Json.mkObj [("domain", Json.str (Flock.hexOf dom)), ("placeholder", Json.str (Flock.hexOf ph)),
    ("root", Json.str (Flock.hexOf (rootAt cap dom bcs ph)))]

/-- `rootPiece`'s input with `"registered"` (hex) to `{"ok": true}`, or `checkAt`'s refusal. -/
def checkPiece (j : Json) : V Json := do
  let some (.str r) := field? j "registered" | throw "registered is not a hex string"
  let registered ← Flock.unhex r
  checkAt (← natField j "capacity") (← strField j "program") (← strField j "table") (← natField j "words")
    (← hexes (getD j "bcs") "bcs") registered
  return Json.mkObj [("ok", Json.bool true)]

def intList (j : Json) (what : String) : V (List Int) := do
  let .arr xs := j | throw s!"{what} is not a list"
  xs.toList.mapM fun x => match int? x with
    | some i => pure i
    | none => throw s!"{what}: not an integer"

def natList (j : Json) (what : String) : V (List Nat) := do
  (← intList j what).mapM fun i => if 0 ≤ i then pure i.toNat else throw s!"{what}: not a natural number"

/-- A run: `{"start", "count", "writer", "out", "radix", "base", "step"}` or `{"start", "count", "held", "row"}`. -/
def runOf (j : Json) : V Run := do
  let start ← natField j "start"
  let count ← natField j "count"
  match keys? j with
  | some ks =>
    if ks == sortStrings #["base", "count", "out", "radix", "start", "step", "writer"] then
      let some base := int? (getD j "base") | throw "base is not an integer"
      let src := From.unit (← strField j "writer") (← strField j "out") (← natList (getD j "radix") "radix") base
        (← intList (getD j "step") "step")
      return { start, count, src }
    else if ks == sortStrings #["count", "held", "row", "start"] then
      return { start, count, src := .held (← strField j "held") (← natField j "row") }
    else throw "a run is {start, count, writer, out, radix, base, step} or {start, count, held, row}"
  | none => throw "a run is not an object"

/-- `{"runs", "length", "writer", "out", "units", "held": {value: rows}}` to `{"sources": [{"unit"} | {"held", "row"}]}`:
`order`'s sources, or its refusal. -/
def orderPiece (j : Json) : V Json := do
  let .arr rs := getD j "runs" | throw "runs is not a list"
  let runs ← rs.mapM runOf
  let held ← match getD j "held" with
    | .obj _ => (entries (getD j "held")).mapM fun (v, n) => match nat? n with
      | some k => pure (v, k)
      | none => throw s!"held: {v}'s rows are not a natural number"
    | _ => throw "held is not an object"
  let srcs ← order runs (← natField j "length") (← strField j "writer") (← strField j "out") (← natField j "units")
    fun v => (held.find? (·.1 == v)).map (·.2)
  return Json.mkObj [("sources", Json.arr (srcs.map fun
    | .unit u => Json.mkObj [("unit", Lean.toJson u)]
    | .held v r => Json.mkObj [("held", Json.str v), ("row", Lean.toJson r)]))]

/-- A member: `{"base", "n", "ports", "body"}` (hex), with `"tables": {port: rows}` for a shared-row body. -/
def memberOf (j : Json) : V Member := do
  let .arr ps := getD j "ports" | throw "a member's ports are not a list"
  let ports ← ps.mapM fun | .str s => pure s | _ => throw "a member's port is not a string"
  let tables ← match field? j "tables" with
    | none => pure none
    | some t@(.obj _) => some <$> (entries t).mapM fun (p, n) => match nat? n with
      | some k => pure (p, k)
      | none => throw s!"tables: {p}'s rows are not a natural number"
    | some _ => throw "a member's tables are not an object"
  let .str b := getD j "body" | throw "a member's body is not a hex string"
  return { base := ← natField j "base", n := ← natField j "n", ports, tables, body := ← Flock.unhex b }

/-- `{"runs", "length", "writer", "out", "first", "units", "members", "held": {value: [hex]}}` to `{"bcs": [hex]}`:
`tableStrings`' strings, or its refusal. -/
def stringsPiece (j : Json) : V Json := do
  let .arr rs := getD j "runs" | throw "runs is not a list"
  let .arr ms := getD j "members" | throw "members is not a list"
  let held ← match getD j "held" with
    | h@(.obj _) => (entries h).mapM fun (v, ss) => return (v, ← hexes ss s!"held {v}")
    | _ => throw "held is not an object"
  let bcs ← tableStrings (← rs.mapM runOf) (← natField j "length") (← strField j "writer") (← strField j "out")
    (← natField j "first") (← natField j "units") (← ms.mapM memberOf) held
  return Json.mkObj [("bcs", Json.arr (bcs.map (Json.str ∘ Flock.hexOf)))]

/-- An object of hex strings, as `(key, bytes)`. -/
def hexObj (j : Json) (what : String) : V (Array (String × ByteArray)) := do
  let .obj _ := j | throw s!"{what} is not an object"
  (entries j).mapM fun (k, v) => match v with
    | .str s => return (k, ← Flock.unhex s)
    | _ => throw s!"{what}: {k} is not a hex string"

/-- `tableRoots` over its JSON form under the leaf layer `leaf`: `{"digests": {key: hex}, "tables": [{"name", "program",
"writer", "out", "words", "runs", "capacity", "first", "units", "members": [member with "key"]}], "held": {name: {value:
[hex]}}, "registered": {name: hex}}`. -/
def rootsOf (leaf : String) (j : Json) : V (Array (String × ByteArray × ByteArray)) := do
  let .obj _ := getD j "digests" | throw "digests is not an object"
  let digests ← (entries (getD j "digests")).mapM fun (k, d) => match d with
    | .str s => pure (k, s)
    | _ => throw s!"digests: {k} is not a string"
  let .arr ts := getD j "tables" | throw "tables is not a list"
  let tables ← ts.mapM fun t => do
    let .arr rs := getD t "runs" | throw "a table's runs are not a list"
    let .arr ms := getD t "members" | throw "a table's members are not a list"
    let spec : Spec := { program := ← strField t "program", writer := ← strField t "writer", out := ← strField t "out",
                         words := ← natField t "words", runs := ← rs.mapM runOf, capacity := ← natField t "capacity" }
    let members ← ms.mapM fun m => return (← strField m "key", ← memberOf m)
    return (← strField t "name", spec, ← natField t "first", ← natField t "units", members)
  let held ← match getD j "held" with
    | h@(.obj _) => (entries h).mapM fun (n, vs) => do
      let .obj _ := vs | throw s!"held: {n} is not an object"
      return (n, ← (entries vs).mapM fun (v, ss) => return (v, ← hexes ss s!"held {n} {v}"))
    | _ => throw "held is not an object"
  let registered ← hexObj (getD j "registered") "registered"
  tableRoots leaf digests tables (fun n => ((held.find? (·.1 == n)).map (·.2)).getD #[])
    fun n => (registered.find? (·.1 == n)).map (·.2)

/-- The supplied values `supplied` (`Supplied.sources`' form) with each source `{"history": "H/root"}` or
`{"history": "H/domain"}` the bytes `{"bytes": hex}` of the root or domain id `roots` holds for table `H`; an error on
a table `roots` holds none of, or another field. Every other source is left as it is, for `Supplied.sources` to read or
refuse. -/
def resolve (roots : Array (String × ByteArray × ByteArray)) (supplied : Json) : V Json := do
  let .obj _ := supplied | return supplied
  let source : Json → V Json := fun s => match s with
    | .obj o => match o.toArray with
      | #[("history", .str n)] =>
        match roots.find? (n == ·.1 ++ "/root"), roots.find? (n == ·.1 ++ "/domain") with
        | some (_, r, _), _ => pure (Json.mkObj [("bytes", Json.str (Flock.hexOf r))])
        | none, some (_, _, d) => pure (Json.mkObj [("bytes", Json.str (Flock.hexOf d))])
        | none, none => throw s!"history {n}: the verifier computed no table's root or domain of that name"
      | _ => pure s
    | _ => pure s
  let row : Json → V Json := fun r => do
    match r with
    | .arr ss => return Json.arr (← ss.mapM source)
    | r => pure r
  let values ← (entries supplied).mapM fun (v, rows) => do
    match rows with
    | .arr rs => return (v, Json.arr (← rs.mapM row))
    | rows => pure (v, rows)
  return Json.mkObj values.toList

/-- `{"leaf_layer", "digests": {key: hex}, "tables": [{"name", "program", "writer", "out", "words", "runs", "capacity",
"first", "units", "members": [member with "key"]}], "held": {name: {value: [hex]}}, "registered": {name: hex}}` to
`{"roots": {name: {"root", "domain"}}}`: `tableRoots`', or its refusal. -/
def rootsPiece (j : Json) : V Json := do
  let .str leaf := getD j "leaf_layer" | throw "leaf_layer is not a string"
  let got ← rootsOf leaf j
  return Json.mkObj [("roots", Json.mkObj (got.toList.map fun (n, r, d) =>
    (n, Json.mkObj [("domain", Json.str (Flock.hexOf d)), ("root", Json.str (Flock.hexOf r))])))]

end Spot.HistoryTable
