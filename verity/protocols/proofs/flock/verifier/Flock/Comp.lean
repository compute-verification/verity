/-
`Flock.Statement`: sparse F2 matrices, and Flock's BLAKE3 compression circuit as a pinned file (`flock-r1cs-rows/v1`)
whose statement digest `D_b3` the verifier recomputes and compares with the pinned value.
-/
module

public import Flock.Canon
public import Flock.Hash
public import Flock.JsonParse

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

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

/-- `rd32` at `o`, its bytes read with `uget`. -/
@[inline] def rd32U (data : ByteArray) (o : USize) (h : o.toNat + 4 ≤ data.size) (hs : o.toNat + 4 ≤ USize.size) :
    Nat :=
  ((data.uget o (by omega)).toUInt32 |||
    ((data.uget (o + USize.ofNat 1) (by rw [toNat_add_ofNat o 1 (by omega)]; omega)).toUInt32 <<< 8) |||
    ((data.uget (o + USize.ofNat 2) (by rw [toNat_add_ofNat o 2 (by omega)]; omega)).toUInt32 <<< 16) |||
    ((data.uget (o + USize.ofNat 3) (by rw [toNat_add_ofNat o 3 (by omega)]; omega)).toUInt32 <<< 24)).toNat

theorem rd32U_eq (data : ByteArray) (o : USize) (h : o.toNat + 4 ≤ data.size) (hs : o.toNat + 4 ≤ USize.size) :
    rd32U data o h hs = rd32 data o.toNat := by
  rw [← rd32Fast_eq]
  unfold rd32U rd32Fast
  rw [uget_eq_get! data o o.toNat rfl,
    uget_eq_get! data _ (o.toNat + 1) (toNat_add_ofNat o 1 (by omega)),
    uget_eq_get! data _ (o.toNat + 2) (toNat_add_ofNat o 2 (by omega)),
    uget_eq_get! data _ (o.toNat + 3) (toNat_add_ofNat o 3 (by omega))]

theorem toNat_toUSize (s : Nat) (h : s < USize.size) : s.toUSize.toNat = s := by
  have hsz : USize.size = 2 ^ System.Platform.numBits := rfl
  rw [Nat.toUSize, USize.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]

/-- `f` over the `k` entries from byte `o` of `data`, read with `rd32U`. -/
@[specialize] def rowFoldU {β : Type} (data : ByteArray) (f : β → Nat → β) :
    (k : Nat) → (o : USize) → o.toNat + 4 * k ≤ data.size → o.toNat + 4 * k < USize.size → β → β
  | 0, _, _, _, acc => acc
  | k + 1, o, h, hs, acc =>
    rowFoldU data f k (o + USize.ofNat 4) (by rw [toNat_add_ofNat o 4 (by omega)]; omega)
      (by rw [toNat_add_ofNat o 4 (by omega)]; omega) (f acc (rd32U data o (by omega) (by omega)))

theorem rowFoldU_eq {β : Type} (data : ByteArray) (s : Nat) (f : β → Nat → β) :
    ∀ (k t : Nat) (o : USize), o.toNat = s + 4 * t → (h : o.toNat + 4 * k ≤ data.size) →
      (hs : o.toNat + 4 * k < USize.size) → ∀ acc : β,
      rowFoldU data f k o h hs acc = (List.range' t k).foldl (fun acc u => f acc (rd32 data (s + 4 * u))) acc
  | 0, _, _, _, _, _, _ => rfl
  | k + 1, t, o, ho, h, hs, acc => by
    rw [rowFoldU, List.range'_succ, List.foldl_cons, rd32U_eq, ho,
      rowFoldU_eq data s f k (t + 1) _ (by rw [toNat_add_ofNat o 4 (by omega), ho]; omega)]

/-- `f` over the `k` numbers from `i`, without building the list. -/
@[specialize] def foldFrom {β : Type} (f : β → Nat → β) : Nat → Nat → β → β
  | 0, _, acc => acc
  | k + 1, i, acc => foldFrom f k (i + 1) (f acc i)

theorem foldFrom_eq {β : Type} (f : β → Nat → β) :
    ∀ (k i : Nat) (acc : β), foldFrom f k i acc = (List.range' i k).foldl f acc
  | 0, _, _ => rfl
  | k + 1, i, acc => by rw [foldFrom, List.range'_succ, List.foldl_cons, foldFrom_eq f k (i + 1)]

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

theorem pushLe32_eq (b : ByteArray) (x : Nat) :
    pushLe32 b x = (((b.push (x % 256).toUInt8).push ((x >>> 8) % 256).toUInt8).push ((x >>> 16) % 256).toUInt8).push
      ((x >>> 24) % 256).toUInt8 := by
  simp only [pushLe32, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size, List.range', Id.run]
  rfl

theorem toUInt8_shiftRight (x : Nat) (k : UInt32) (hx : x ≤ 4294967295) (hk : k.toNat < 32) :
    (x.toUInt32 >>> k).toUInt8 = ((x >>> k.toNat) % 256).toUInt8 := by
  apply UInt8.toNat_inj.mp
  rw [UInt32.toNat_toUInt8, UInt32.toNat_shiftRight, Nat.mod_eq_of_lt hk]
  have h1 : x.toUInt32.toNat = x := by
    simp [Nat.toUInt32, UInt32.toNat_ofNat']
    omega
  rw [h1]
  simp [Nat.toUInt8, UInt8.toNat_ofNat']

/-- `pushLe32` as compiled code computes it (`pushLe32_eq_fast`): the bytes cut from a `UInt32` when `x` fits one. The
bound is `≤ 4294967295`, not `< 2 ^ 32`: compiled code parses a literal past `UInt32` from its decimal string each
call. -/
def pushLe32Fast (b : ByteArray) (x : Nat) : ByteArray :=
  if x ≤ 4294967295 then
    let w := x.toUInt32
    (((b.push w.toUInt8).push (w >>> 8).toUInt8).push (w >>> 16).toUInt8).push (w >>> 24).toUInt8
  else pushLe32 b x

/-- Compiled code runs `pushLe32Fast`; the kernel and every proof see `pushLe32`. `@[csimp]` rewrites only code compiled
after it, so it stays directly below `pushLe32`. -/
@[csimp] theorem pushLe32_eq_fast : @pushLe32 = @pushLe32Fast := by
  funext b x
  unfold pushLe32Fast
  split
  · rename_i hx
    have h0 := toUInt8_shiftRight x 0 hx (by decide)
    rw [UInt32.shiftRight_zero] at h0
    dsimp only
    rw [pushLe32_eq, h0, toUInt8_shiftRight x 8 hx (by decide), toUInt8_shiftRight x 16 hx (by decide),
      toUInt8_shiftRight x 24 hx (by decide)]
    rfl
  · rfl

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

/-- `(m.row i).foldl f init` (`rowFold_eq`) without building the row: with `rowFoldU` where the row lies in `m.data` and
its offsets fit a `USize`. -/
@[inline] def rowFold {β : Type} (m : Sparse) (i : Nat) (f : β → Nat → β) (init : β) : β :=
  let s := m.start.getD i 0
  let k := m.len.getD i 0
  if h : s + 4 * k ≤ m.data.size ∧ (s + 4 * k).toUSize.toNat = s + 4 * k then
    have := USize.toNat_lt_size (s + 4 * k).toUSize
    have hs := toNat_toUSize s (by omega)
    rowFoldU m.data f k s.toUSize (by omega) (by omega) init
  else (m.row i).foldl f init

theorem rowFold_eq {β : Type} (m : Sparse) (i : Nat) (f : β → Nat → β) (init : β) :
    m.rowFold i f init = (m.row i).foldl f init := by
  unfold rowFold
  dsimp only
  split
  · rename_i h
    have := USize.toNat_lt_size (m.start.getD i 0 + 4 * m.len.getD i 0).toUSize
    rw [rowFoldU_eq m.data (m.start.getD i 0) f _ 0 _ (by rw [toNat_toUSize _ (by omega)]; omega)]
    simp only [row, ← Array.foldl_toList, Array.toList_map, List.foldl_map, Array.toList_range, List.range_eq_range']
  · rfl

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

/-- `Mat(X)` of the circuit digest (`Flock.Statement`), appended to `buf`, flushing into `h` every 64 KiB. -/
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
‖ Mat(A) ‖ Mat(B) ‖ Mat(I))`, upstream's circuit digest (`Flock.Statement`). -/
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
  let h ← match JsonParse.parse text with
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
