/-
PROTOCOL.md §16.1: sparse F2 matrices, and Flock's BLAKE3 compression circuit as a pinned file (`flock-r1cs-rows/v1`)
whose statement digest `D_b3` the verifier recomputes and compares with the pinned value.
-/
import Lean.Data.Json
import Flock.Canon
import Flock.Hash

namespace Flock

open Lean (Json)

@[inline] def rd32 (b : ByteArray) (o : Nat) : Nat :=
  (b.get! o).toNat ||| ((b.get! (o + 1)).toNat <<< 8) ||| ((b.get! (o + 2)).toNat <<< 16) ||| ((b.get! (o + 3)).toNat <<< 24)

/-- `rd32` as compiled code computes it (`rd32Fast_eq`): the shifts in `UInt32`, where compiled `Nat.shiftLeft` makes a
GMP integer each time. -/
@[inline] def rd32Fast (b : ByteArray) (o : Nat) : Nat :=
  ((b.get! o).toUInt32 ||| ((b.get! (o + 1)).toUInt32 <<< 8) ||| ((b.get! (o + 2)).toUInt32 <<< 16) |||
    ((b.get! (o + 3)).toUInt32 <<< 24)).toNat

theorem rd32Fast_shl (x : UInt8) (k : UInt32) (hk : k.toNat + 8 ≤ 32) :
    (x.toUInt32 <<< k).toNat = x.toNat <<< k.toNat := by
  have hk32 : k.toNat % 32 = k.toNat := Nat.mod_eq_of_lt (by omega)
  rw [UInt32.toNat_shiftLeft, UInt8.toNat_toUInt32, hk32, Nat.shiftLeft_eq]
  apply Nat.mod_eq_of_lt
  calc x.toNat * 2 ^ k.toNat < 2 ^ 8 * 2 ^ k.toNat := Nat.mul_lt_mul_of_pos_right x.toNat_lt (Nat.two_pow_pos _)
    _ = 2 ^ (8 + k.toNat) := (Nat.pow_add 2 8 k.toNat).symm
    _ ≤ 2 ^ 32 := Nat.pow_le_pow_right (by decide) (by omega)

theorem rd32Fast_eq (b : ByteArray) (o : Nat) : rd32Fast b o = rd32 b o := by
  simp only [rd32Fast, rd32, UInt32.toNat_or, UInt8.toNat_toUInt32,
    rd32Fast_shl _ 8 (by decide), rd32Fast_shl _ 16 (by decide), rd32Fast_shl _ 24 (by decide)]
  rfl

/-- Compiled code runs `rd32Fast`; the kernel and every proof see `rd32`. `@[csimp]` rewrites only code compiled after
it, so it stays directly below `rd32`. -/
@[csimp] theorem rd32_eq_rd32Fast : @rd32 = @rd32Fast :=
  funext fun b => funext fun o => (rd32Fast_eq b o).symm

def findByte (b : ByteArray) (x : UInt8) (from_ : Nat := 0) : Option Nat := Id.run do
  for i in [from_:b.size] do
    if b.get! i == x then return some i
  return none

def pushLe64 (b : ByteArray) (x : Nat) : ByteArray := Id.run do
  let mut out := b
  for i in [0:8] do out := out.push ((x >>> (8 * i)) % 256).toUInt8
  return out

def pushLe32 (b : ByteArray) (x : Nat) : ByteArray := Id.run do
  let mut out := b
  for i in [0:4] do out := out.push ((x >>> (8 * i)) % 256).toUInt8
  return out

/-- A sparse F2 matrix with `rows` rows over `width` columns: row `i`'s columns are the le32 words
`data[start[i] ..][0 .. len[i]]`. Rows past `rows` are empty. -/
structure Sparse where
  rows : Nat
  width : Nat
  data : ByteArray
  start : Array Nat
  len : Array Nat
  deriving Inhabited

namespace Sparse

def row (m : Sparse) (i : Nat) : Array Nat :=
  (Array.range (m.len.getD i 0)).map fun t => rd32 m.data (m.start.getD i 0 + 4 * t)

/-- Rows given explicitly; every column must be below `width`. -/
def ofRows (rows : Array (Array Nat)) (width : Nat) : V Sparse := do
  let mut data := ByteArray.empty
  let mut start := #[]
  let mut len := #[]
  for r in rows do
    start := start.push data.size
    len := len.push r.size
    for c in r do
      if c ≥ width then throw s!"sparse matrix: column {c} outside {width}"
      data := pushLe32 data c
  return ⟨rows.size, width, data, start, len⟩

/-- Column `t` of row `i`. -/
@[inline] def col (m : Sparse) (i t : Nat) : Nat := rd32 m.data (m.start[i]! + 4 * t)

/-- Entries `t, …, t + fuel − 1` of row `i`, each adding `ei` to its column. -/
def foldRow (m : Sparse) (i : Nat) (ei : F128) : Nat → Nat → Array F128 → Array F128
  | 0, _, acc => acc
  | fuel + 1, t, acc => foldRow m i ei fuel (t + 1) (acc.modify (m.col i t) (· + ei))

/-- Rows `i, …, i + fuel − 1`. -/
def foldRows (m : Sparse) (e : Array F128) : Nat → Nat → Array F128 → Array F128
  | 0, _, acc => acc
  | fuel + 1, i, acc =>
    let ei := e[i]!
    foldRows m e fuel (i + 1) (if ei.isZero then acc else m.foldRow i ei m.len[i]! 0 acc)

/-- `Xᵀ e`: `out[c] = Σ_{i : c ∈ row i} e[i]` for `c < width`. -/
def foldT (m : Sparse) (e : Array F128) : Array F128 := m.foldRows e (min m.rows e.size) 0 (Array.replicate m.width 0)

/-- `Mat(X)` of the circuit digest (§16.1), appended to `buf`, flushing into `h` every 64 KiB. -/
def absorbMat (m : Sparse) (rows : Nat) (h0 : Blake3.Hasher) (buf0 : ByteArray) : Blake3.Hasher × ByteArray := Id.run do
  let mut h := h0
  let mut buf := pushLe64 (pushLe64 buf0 rows) m.width
  for i in [0:rows] do
    let n := m.len.getD i 0
    buf := pushLe64 buf n
    let s := m.start.getD i 0
    for t in [0:n] do buf := pushLe64 buf (rd32 m.data (s + 4 * t))
    if buf.size ≥ 65536 then
      h := h.update buf
      buf := ByteArray.empty
  return (h, buf)

end Sparse

/-- What the verifier pins about a slot circuit file: its shape, the header values it must state, the input rows Δ
rewires (`A = [r]`, `B = [zConst]`), and its digest. -/
structure CompSpec where
  name : String
  /-- the slot is `2^kLog` bits -/
  kLog : Nat
  /-- the block R1CS it was digested as -/
  mDigest : Nat
  usefulBits : Nat
  zConst : Nat
  /-- further header fields that must hold these values (the layout offsets the leaf's Δ and regions use) -/
  header : List (String × Nat)
  inputRows : List Nat
  pinnedDigest : String
  deriving Inhabited

/-- A pinned slot circuit, `C = I`. -/
structure Comp where
  spec : CompSpec
  a : Sparse
  b : Sparse
  /-- recomputed from the rows at load -/
  digest : ByteArray
  deriving Inhabited

namespace Comp

/-- `D = BLAKE3("flock-r1cs-stmt-v2" ‖ le64 m ‖ le64 k_log ‖ le64 k_skip ‖ le64 useful_bits ‖ u8 1 ‖ le64 pin ‖ u8 0
‖ Mat(A) ‖ Mat(B) ‖ Mat(I))`, upstream's circuit digest (§16.1). -/
def digestOf (sp : CompSpec) (a b : Sparse) : ByteArray := Id.run do
  let n := 2 ^ sp.kLog
  let mut buf := "flock-r1cs-stmt-v2".toUTF8
  for x in [sp.mDigest, sp.kLog, 6, sp.usefulBits] do buf := pushLe64 buf x
  buf := pushLe64 (buf.push 1) sp.zConst |>.push 0
  let (h1, b1) := a.absorbMat n Blake3.Hasher.plain buf
  let (h2, b2) := b.absorbMat n h1 b1
  let mut h := h2
  buf := pushLe64 (pushLe64 b2 n) n
  for i in [0:n] do
    buf := pushLe64 (pushLe64 buf 1) i
    if buf.size ≥ 65536 then
      h := h.update buf
      buf := ByteArray.empty
  return (h.update buf).finalize

private def readRows (file : ByteArray) (pos0 : Nat) (rows width : Nat) : V (Sparse × Nat) := do
  let mut pos := pos0
  let mut start := Array.mkEmpty rows
  let mut len := Array.mkEmpty rows
  for _ in [0:rows] do
    if pos + 4 > file.size then throw "slot circuit: truncated"
    let k := rd32 file pos
    if pos + 4 + 4 * k > file.size then throw "slot circuit: truncated"
    for t in [0:k] do
      if rd32 file (pos + 4 + 4 * t) ≥ width then throw "slot circuit: a column outside the slot"
    start := start.push (pos + 4)
    len := len.push k
    pos := pos + 4 + 4 * k
  return (⟨rows, width, file, start, len⟩, pos)

/-- Load a pinned `flock-r1cs-rows/v1` file (a JSON header line, then A's rows and B's rows, each `le32 count` then `le32`
columns): its header must state the pinned shape, its rows must decode exactly, its input rows must be the ones Δ rewires,
and its digest must be the pinned one. -/
def load (sp : CompSpec) (file : ByteArray) : V Comp := do
  let ctx := s!"slot circuit {sp.name}"
  let some nl := findByte file 10 | throw s!"{ctx}: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw s!"{ctx}: header is not UTF-8"
  let h ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"{ctx}: header: {e}"
  if (← J.str h "format" ctx) != "flock-r1cs-rows/v1" then throw s!"{ctx}: not flock-r1cs-rows/v1"
  let want : List (String × Nat) := [("m", sp.mDigest), ("k_log", sp.kLog), ("k_skip", 6), ("useful_bits", sp.usefulBits),
    ("const_pin", sp.zConst), ("layout", 0), ("rows", 2 ^ sp.kLog), ("cols", 2 ^ sp.kLog)] ++ sp.header
  for (k, v) in want do
    if (← J.nat h k ctx) != v then throw s!"{ctx}: {k} is not {v}"
  let n := 2 ^ sp.kLog
  let (a, p1) ← readRows file (nl + 1) n n
  let (b, p2) ← readRows file p1 n n
  if p2 != file.size then throw s!"{ctx}: trailing bytes"
  for r in sp.inputRows do
    if a.row r != #[r] || b.row r != #[sp.zConst] then throw s!"{ctx}: row {r} is not an input row"
  let digest := digestOf sp a b
  if hexOf digest != sp.pinnedDigest then throw s!"{ctx}: its digest is not the pinned one"
  return ⟨sp, a, b, digest⟩

end Comp

end Flock
