import PouwWindow.Grid
import Flock.Hash
import Flock.Hm96
import Flock.HmRow
import PouwCommit.Frame

/-!
# The entry row (`window.py`, "The entry row")

One entry of the sampling block's root as its committed 336-byte row holds it:
six u64s little-endian (kind, lo, hi, n_a, n_b, a zero pad), the side roots A and B and the pass's -h3 root (64 bytes
each), tile0, cols, m and a zero u64, then the call's record's SHA-512. An `Entry` here holds what a row can hold: u64s
and 64-byte digests (`Entry.fromRow?`); Python's dataclass holds any values, and `Entry.row` refuses the rest, so the
functions of an entry agree on the entries that encode. `layoutProblem` holds an entry to the layout; `tileIn?`, `tileAt?`
and `tileRead?` map a drawn index to its rows and its word in the pass's -h3 leaf; `entryDigest`, `entryBc` and `entryLeaf`
commit the row as an `hm96-sha512/row/v2` row.
-/

namespace PouwWindow

open Flock

/-- `T_BITS`: a tile index's bits. -/
def tBits : Nat := 40

/-- A side of an entry: A holds a call's kept activation rows, B its kept weight rows. -/
inductive Side where
  | a
  | b
  deriving DecidableEq, Repr

/-- `DEPTHS`: the side trees' public depths. -/
def Side.depth : Side → Nat
  | .a => 13
  | .b => 17

def Side.name : Side → String
  | .a => "a"
  | .b => "b"

def Side.ofName? : String → Option Side
  | "a" => some .a
  | "b" => some .b
  | _ => none

def sides : List Side := [.a, .b]

/-- `PORT`: the window root's port. -/
def port : String := "entries"
/-- `ENTRY_SCHEMA`: `hm96-sha512/row/v2`. -/
def entrySchema : String := HmRow.ROW_SCHEMA_V2
def entryBytes : Nat := 336
def entryWords : Nat := entryBytes / 2
def entryBits : Nat := 8 * entryBytes
/-- `TILE`: the pass's tiles are 64 × 64. -/
def tile : Nat := 64
/-- `ROW_STEP`: a short last tile row is whole 16-row ticket subtiles. -/
def rowStep : Nat := 16

/-- An entry's kind; its code is `KIND_CODE`'s. -/
inductive Kind where
  | none
  | call
  | filler
  deriving DecidableEq, Repr, Inhabited

def Kind.code : Kind → Nat
  | .none => 0
  | .call => 1
  | .filler => 2

def Kind.name : Kind → String
  | .none => "none"
  | .call => "call"
  | .filler => "filler"

def Kind.ofCode? : Nat → Option Kind
  | 0 => some .none
  | 1 => some .call
  | 2 => some .filler
  | _ => Option.none

def Kind.ofName? : String → Option Kind
  | "none" => some .none
  | "call" => some .call
  | "filler" => some .filler
  | _ => Option.none

def zeroDigest : ByteArray := ByteArray.mk (Array.replicate 64 0)

/-- `Entry`: one entry as its row holds it. -/
structure Entry where
  kind : Kind
  lo : Nat := 0
  hi : Nat := 0
  nA : Nat := 0
  nB : Nat := 0
  rootA : ByteArray := zeroDigest
  rootB : ByteArray := zeroDigest
  rootH3 : ByteArray := zeroDigest
  record : ByteArray := zeroDigest
  tile0 : Nat := 0
  cols : Nat := 0
  m : Nat := 0
  deriving Inhabited

namespace Entry

def nSide (e : Entry) : Side → Nat
  | .a => e.nA
  | .b => e.nB

def root (e : Entry) : Side → ByteArray
  | .a => e.rootA
  | .b => e.rootB

/-- `entry == Entry("none")`: the zero row. -/
def isZero (e : Entry) : Bool :=
  e.kind == .none && e.lo == 0 && e.hi == 0 && e.nA == 0 && e.nB == 0 && e.rootA == zeroDigest &&
    e.rootB == zeroDigest && e.rootH3 == zeroDigest && e.record == zeroDigest && e.tile0 == 0 && e.cols == 0 && e.m == 0

/-- `k` bytes of `n`, little-endian. -/
def le (k n : Nat) : ByteArray := ByteArray.mk ((Array.range k).map fun i => (n / 256 ^ i % 256).toUInt8)

/-- `k` bytes of `b` from `o`, little-endian. -/
def readLe (b : ByteArray) (o k : Nat) : Nat := (List.range k).foldl (fun acc i => acc + (b.get! (o + i)).toNat * 256 ^ i) 0

/-- The row's bytes (for an entry whose integers are u64s and whose digests are 64 bytes). -/
def row (e : Entry) : ByteArray :=
  le 8 e.kind.code ++ le 8 e.lo ++ le 8 e.hi ++ le 8 e.nA ++ le 8 e.nB ++ le 8 0 ++ e.rootA ++ e.rootB ++ e.rootH3 ++
    le 8 e.tile0 ++ le 8 e.cols ++ le 8 e.m ++ le 8 0 ++ e.record

/-- `Entry.row`'s refusals: the roots and record 64-byte digests, the integers u64s. -/
def check (e : Entry) : Except String Unit := do
  if [e.rootA, e.rootB, e.rootH3, e.record].any (·.size != 64) then
    throw "an entry's roots and record are 64-byte digests"
  if [e.lo, e.hi, e.nA, e.nB, e.tile0, e.cols, e.m].any (· ≥ 2 ^ 64) then
    throw "an entry's integers are u64s: argument out of range"

/-- `Entry.from_row`. -/
def fromRow? (row : ByteArray) : Except String Entry := do
  if row.size != entryBytes then throw s!"an entry row is {entryBytes} bytes, not {row.size}"
  let code := readLe row 0 8
  let some kind := Kind.ofCode? code | throw s!"kind {code} is none of ('call', 'filler', 'none') ([0, 1, 2])"
  if readLe row 40 8 != 0 then throw "its bytes 40..48 are not zero"
  if readLe row 264 8 != 0 then throw "its bytes 264..272 are not zero"
  pure { kind, lo := readLe row 8 8, hi := readLe row 16 8, nA := readLe row 24 8, nB := readLe row 32 8,
         rootA := row.extract 48 112, rootB := row.extract 112 176, rootH3 := row.extract 176 240,
         tile0 := readLe row 240 8, cols := readLe row 248 8, m := readLe row 256 8, record := row.extract 272 336 }

/-- `Entry.words`: the row as its 168 u16 words, little-endian. -/
def words (e : Entry) : List Nat := (List.range entryWords).map fun i => readLe e.row (2 * i) 2

/-- `Entry.from_words`. -/
def fromWords? (ws : List Int) : Except String Entry := do
  if ws.length != entryWords || ws.any (fun w => w < 0 || w ≥ 2 ^ 16) then
    throw s!"an entry row is {entryWords} u16 words"
  fromRow? (ws.foldl (fun acc w => acc ++ le 2 w.toNat) .empty)

/-- `pass_tiles`: a call's tiles in its pass, ⌈m/64⌉ tile rows of `cols` tiles. -/
def passTiles (e : Entry) : Nat := (e.m + tile - 1) / tile * e.cols

/-- `tiles`: an entry's tiles, its sides' rows' products. -/
def tiles (e : Entry) : Nat := if e.kind == .none then 0 else e.nA * e.nB

private def anyNonzero (b : ByteArray) : Bool := b.toList.any (· != 0)

/-- `layout_problem`: why an entry's values are not the layout's, or `none`. -/
def layoutProblem (e : Entry) : Option String :=
  if !(e.lo < 2 ^ tBits && e.hi < 2 ^ tBits) then some s!"its range [{e.lo}, {e.hi}) is not of {tBits}-bit indices"
  else if e.kind == .none then if e.isZero then none else some "an empty entry is not the zero row"
  else match sides.find? (fun s => !(1 ≤ e.nSide s && e.nSide s ≤ 2 ^ s.depth)) with
  | some s => some s!"side {s.name} holds {e.nSide s} rows, not 1 to 2^{s.depth}"
  | none =>
    if e.kind == .filler then
      if anyNonzero e.rootH3 then some "the filler's -h3 root is not zero"
      else if anyNonzero e.record then some "the filler's record is not zero"
      else if e.tile0 != 0 || e.cols != 0 || e.m != 0 then some "the filler's tile0, cols and m are not zero"
      else none
    else if !(1 ≤ e.m && e.m ≤ 2 ^ Side.a.depth && e.m % rowStep == 0 && e.nA ≤ e.m) then
      some s!"its m {e.m} is not a multiple of {rowStep} from n_a = {e.nA} to 2^{Side.a.depth}"
    else if !(1 ≤ e.cols && e.cols ≤ 2 ^ (Side.b.depth - 6) && e.nB ≤ tile * e.cols) then
      some s!"its cols {e.cols} is not 1 to 2^{Side.b.depth - 6} with 64·cols >= n_b = {e.nB}"
    else if e.tile0 + e.passTiles > 2 ^ tBits then
      some s!"its pass tiles [{e.tile0}, {e.tile0 + e.passTiles}) are not of {tBits}-bit indices"
    else none

/-- `tile_in`: (A row, B row) of index `t` in the entry's range, row-major. -/
def tileIn? (e : Entry) (t : Int) : Except String (Nat × Nat) :=
  if e.kind == .none || !(e.lo ≤ t && t < e.hi) || e.nB < 1 then .error s!"index {t} is not in the entry's range"
  else
    let u := (t - e.lo).toNat
    .ok (u / e.nB, u % e.nB)

/-- `tile_read`: (p, at), the -h3 leaf a call's tile (I, J) reads and its word's place in it. `at` is Python's
`(I mod 64) | (J mod 64) << 6 | (rows/16 − 1) << 12`: the low part is below 2^12, so the or is a sum, and it is negative
(−4096 plus the low part) when the tile row is shorter than 16, which only an entry off the layout has. -/
def tileRead? (e : Entry) (i j : Int) : Except String (Nat × Int) :=
  if e.kind != .call || !(0 ≤ i && i < e.m && 0 ≤ j && j < tile * e.cols) then
    .error s!"({i}, {j}) is not a word of the call's m x 64·cols"
  else
    let i := i.toNat
    let j := j.toNat
    let rows := min tile (e.m - tile * (i / tile))
    .ok (e.tile0 + i / tile * e.cols + j / tile,
      ((i % tile + j % tile * 64 : Nat) : Int) + ((rows / rowStep : Nat) - 1 : Int) * 4096)

/-- `entry_digest`: x, the row's `sha512/row/v2` digest. -/
def digest (e : Entry) : ByteArray := Sha512.hash (HmRow.rowPrefixV2 entryBits ++ e.row)

/-- `entry_bc`: b ‖ c, hm96-sha512 over the row's digest under `salt` (192 bytes, as hm96 refuses any other). -/
def bc? (e : Entry) (salt : ByteArray) : Except String ByteArray :=
  if salt.size != HmRow.SALT_BYTES then .error s!"an hm96-sha512 salt is {HmRow.SALT_BYTES} bytes"
  else .ok (Hm96.Default512.commitString e.digest salt)

end Entry

/-- `tile_at`: (entry, A row, B row) of drawn index `t`: the first entry, not empty, whose range holds it. -/
def tileAt? (entries : List Entry) (t : Int) : Except String (Nat × Nat × Nat) :=
  match entries.zipIdx.find? (fun (x, _) => x.kind != .none && x.lo ≤ t && t < x.hi) with
  | some (x, e) => do
    let (i, j) ← x.tileIn? t
    pure (e, i, j)
  | none => .error s!"index {t} is in no entry's range"

/-- `entry_leaf`: entry `e`'s frame-v3-sha512 leaf in the window root over its b ‖ c (128 bytes, as hm96's `tree_leaf`
refuses any other). -/
def entryLeaf? (dom : PouwCommit.Domain) (e : Nat) (bc : ByteArray) : Except String ByteArray :=
  if bc.size != HmRow.BC_BYTES then .error s!"an hm96-sha512 commit string b || c is {HmRow.BC_BYTES} bytes"
  else .ok (dom.leaf e entrySchema (Hm96.Default512.treeLeaf bc))

end PouwWindow
