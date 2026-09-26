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

/-- `Xᵀ e`: `out[c] = Σ_{i : c ∈ row i} e[i]` for `c < width`. -/
def foldT (m : Sparse) (e : Array F128) : Array F128 := Id.run do
  let mut acc : Array F128 := Array.replicate m.width 0
  for i in [0:min m.rows e.size] do
    let ei := e[i]!
    if ei.isZero then continue
    let s := m.start[i]!
    for t in [0:m.len[i]!] do
      acc := acc.modify (rd32 m.data (s + 4 * t)) (· + ei)
  return acc

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

/-- The compression circuit: one `2^14`-bit slot, `C = I`. -/
structure Comp where
  a : Sparse
  b : Sparse
  /-- `D_b3`, recomputed from the rows at load -/
  digest : ByteArray
  deriving Inhabited

namespace Comp

def kLog : Nat := 14
def zConst : Nat := 11706
def cvBase : Nat := 0
def outLo : Nat := 256
def mBase : Nat := 512
def tLo : Nat := 1024
def flagsBase : Nat := 1120
def usefulBits : Nat := 11707
/-- The block R1CS it was digested as: `m = 17` (8 blocks of `2^14`). -/
def mDigest : Nat := 17
/-- `D_b3`, upstream's digest of this circuit (b684b12), bound by the circuit statement's digest. -/
def pinnedDigest : String := "3f96431c023dbc354e51d101ab575b2be8cbd33bfcac5e9961d65e8e36d66be3"

/-- `D = BLAKE3("flock-r1cs-stmt-v2" ‖ le64 m ‖ le64 k_log ‖ le64 k_skip ‖ le64 useful_bits ‖ u8 1 ‖ le64 pin ‖ u8 0
‖ Mat(A) ‖ Mat(B) ‖ Mat(I))`. -/
def digestOf (a b : Sparse) : ByteArray := Id.run do
  let n := 2 ^ kLog
  let mut buf := "flock-r1cs-stmt-v2".toUTF8
  for x in [mDigest, kLog, 6, usefulBits] do buf := pushLe64 buf x
  buf := pushLe64 (buf.push 1) zConst |>.push 0
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
    if pos + 4 > file.size then throw "compression circuit: truncated"
    let k := rd32 file pos
    if pos + 4 + 4 * k > file.size then throw "compression circuit: truncated"
    for t in [0:k] do
      if rd32 file (pos + 4 + 4 * t) ≥ width then throw "compression circuit: a column outside the slot"
    start := start.push (pos + 4)
    len := len.push k
    pos := pos + 4 + 4 * k
  return (⟨rows, width, file, start, len⟩, pos)

/-- Load the pinned file: its header must state the pinned shape, its rows must decode exactly, its input rows must be
the ones Δ rewires (`A = [r]`, `B = [z]`), and its digest must be the pinned `D_b3`. -/
def load (file : ByteArray) : V Comp := do
  let some nl := findByte file 10 | throw "compression circuit: no header line"
  let text ← match String.fromUTF8? (file.extract 0 nl) with
    | some s => pure s
    | none => throw "compression circuit: header is not UTF-8"
  let h ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"compression circuit: header: {e}"
  let ctx := "compression circuit"
  if (← J.str h "format" ctx) != "flock-r1cs-rows/v1" then throw "compression circuit: not flock-r1cs-rows/v1"
  let want : List (String × Nat) := [("m", mDigest), ("k_log", kLog), ("k_skip", 6), ("useful_bits", usefulBits),
    ("const_pin", zConst), ("layout", 0), ("rows", 2 ^ kLog), ("cols", 2 ^ kLog), ("cv_base", cvBase),
    ("out_lo_base", outLo), ("m_base", mBase), ("t_lo_base", tLo), ("flags_base", flagsBase)]
  for (k, v) in want do
    if (← J.nat h k ctx) != v then throw s!"compression circuit: {k} is not {v}"
  let n := 2 ^ kLog
  let (a, p1) ← readRows file (nl + 1) n n
  let (b, p2) ← readRows file p1 n n
  if p2 != file.size then throw "compression circuit: trailing bytes"
  for r in (List.range 256).map (cvBase + ·) ++ (List.range 512).map (mBase + ·) ++ (List.range (flagsBase + 32 - tLo)).map (tLo + ·) do
    if a.row r != #[r] || b.row r != #[zConst] then throw s!"compression circuit: row {r} is not an input row"
  let digest := digestOf a b
  if hexOf digest != pinnedDigest then throw "compression circuit: its digest is not the pinned D_b3"
  return ⟨a, b, digest⟩

end Comp

end Flock
