/-
History tables (`one_stage/history.py`): a table of rows that other proof blocks output, read at hidden positions by
table reads (`EmbeddingRead_v1`). The verifier computes each table's root itself, from the writers' public `b ‖ c` in
the order its own program gives; the root the prover registers is only checked against it, so the prover can't
register rows of its choosing.

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
  registered root that is not 64 bytes, then a registered root other than `rootAt`'s. `check` is it at `CAPACITY`.
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

end Spot.HistoryTable
