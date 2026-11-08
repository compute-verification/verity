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

/-- `ofRows` with `data`'s buffer sized up front (`ofRows_eq_fast`): an empty array is the same value whatever its
capacity. -/
def ofRowsFast (rows : Array (Array Nat)) (width : Nat) : V Sparse := do
  let mut data := ByteArray.emptyWithCapacity (4 * rows.foldl (· + ·.size) 0)
  let mut start := Array.mkEmpty rows.size
  let mut len := Array.mkEmpty rows.size
  for r in rows do
    start := start.push data.size
    len := len.push r.size
    for c in r do
      if c ≥ width then throw s!"sparse matrix: column {c} outside {width}"
      data := pushLe32 data c
  return ⟨rows.size, width, data, start, len⟩

/-- Compiled code runs `ofRowsFast`; the kernel and every proof see `ofRows`. `@[csimp]` rewrites only code compiled
after it, so it stays directly below `ofRows`. -/
@[csimp] theorem ofRows_eq_fast : @ofRows = @ofRowsFast := rfl

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

/-- `n` zero bytes, doubling, within `fuel` halvings. -/
def zerosF : Nat → Nat → ByteArray
  | 0, _ => ByteArray.empty
  | fuel + 1, n =>
    if n == 0 then ByteArray.empty
    else
      let h := zerosF fuel (n / 2)
      if n % 2 == 0 then h ++ h else (h ++ h).push 0

/-- `n` zero bytes. -/
def zeros (n : Nat) : ByteArray := zerosF (n.log2 + 1) n

/-- `b` with byte `i` set to `x`; `b` when `i` is out of bounds. -/
@[inline] def setU (b : ByteArray) (i : USize) (x : UInt8) : ByteArray :=
  if h : i.toNat < b.size then b.uset i x h else b

/-- Byte `i` of `b` as a word; `0` when `i` is out of bounds. -/
@[inline] def getU (b : ByteArray) (i : USize) : UInt64 :=
  if h : i.toNat < b.size then (b.uget i h).toUInt64 else 0

/-- Bytes `o, …, o + 7` of `b`, little-endian. -/
@[inline] def rd64 (b : ByteArray) (o : USize) : UInt64 :=
  getU b o ^^^ getU b (o + 1) <<< 8 ^^^ getU b (o + 2) <<< 16 ^^^ getU b (o + 3) <<< 24 ^^^ getU b (o + 4) <<< 32 ^^^
    getU b (o + 5) <<< 40 ^^^ getU b (o + 6) <<< 48 ^^^ getU b (o + 7) <<< 56

/-- `b` with bytes `o, …, o + 7` set to `w`'s, little-endian. -/
@[inline] def wr64 (b : ByteArray) (o : USize) (w : UInt64) : ByteArray :=
  let b := setU b o w.toUInt8
  let b := setU b (o + 1) (w >>> 8).toUInt8
  let b := setU b (o + 2) (w >>> 16).toUInt8
  let b := setU b (o + 3) (w >>> 24).toUInt8
  let b := setU b (o + 4) (w >>> 32).toUInt8
  let b := setU b (o + 5) (w >>> 40).toUInt8
  let b := setU b (o + 6) (w >>> 48).toUInt8
  setU b (o + 7) (w >>> 56).toUInt8

/-- Column `c` of an unboxed accumulator: bytes `16c, …, 16c + 15`, `lo` then `hi`. -/
@[inline] def colB (b : ByteArray) (c : Nat) : F128 :=
  let o := (16 * c).toUSize
  ⟨rd64 b o, rd64 b (o + 8)⟩

theorem setU_eq (b : ByteArray) (i : USize) (x : UInt8) : setU b i x = b.set! i.toNat x := by
  unfold setU
  split
  · rename_i h
    obtain ⟨bs⟩ := b
    simp only [ByteArray.uset, ByteArray.set!, Array.uset, Array.set!_eq_setIfInBounds, Array.setIfInBounds,
      show i.toNat < bs.size from h, dite_true]
  · rename_i h
    obtain ⟨bs⟩ := b
    simp only [ByteArray.set!, Array.set!_eq_setIfInBounds, Array.setIfInBounds, show ¬ i.toNat < bs.size from h,
      dite_false]

theorem getU_eq (b : ByteArray) (i : USize) : getU b i = b[i.toNat]!.toUInt64 := by
  unfold getU
  split
  · rename_i h
    rw [uget_eq_get! b i i.toNat rfl h]
    obtain ⟨bs⟩ := b
    simp only [ByteArray.get!]
    rw [getElem!_pos (ByteArray.mk bs) i.toNat h, getElem!_pos bs i.toNat h]
    rfl
  · rename_i h
    rw [getElem!_neg b i.toNat h]
    rfl

theorem getU_pos (b : ByteArray) (i : USize) (h : i.toNat < b.size) : getU b i = (b.uget i h).toUInt64 := dite_eq_left h

theorem uset_eq_setU (b : ByteArray) (i : USize) (x : UInt8) (h : i.toNat < b.size) : b.uset i x h = setU b i x :=
  (dite_eq_left h).symm

theorem size_uset (b : ByteArray) (i : USize) (x : UInt8) (h : i.toNat < b.size) : (b.uset i x h).size = b.size := by
  rw [uset_eq_setU, setU_eq, ByteArray.size_set!]

theorem size_setU (b : ByteArray) (i : USize) (x : UInt8) : (setU b i x).size = b.size := by
  rw [setU_eq, ByteArray.size_set!]

theorem size_wr64 (b : ByteArray) (o : USize) (w : UInt64) : (wr64 b o w).size = b.size := by
  simp only [wr64, size_setU]

theorem toNat_add_lit (o : USize) (k : Nat) (h : o.toNat + k < USize.size) :
    (o + (OfNat.ofNat k : USize)).toNat = o.toNat + k :=
  toNat_add_ofNat o k h

theorem toNat_toUSize (n : Nat) (h : n < USize.size) : n.toUSize.toNat = n := by
  have hsz : USize.size = 2 ^ System.Platform.numBits := rfl
  simp only [Nat.toUSize, USize.toNat_ofNat']
  exact Nat.mod_eq_of_lt (by omega)

/-- `rd64` with no bounds check: the caller proves bytes `o, …, o + 7` inside `b` (`rd64P_eq`). -/
@[inline] def rd64P (b : ByteArray) (o : USize) (h : o.toNat + 8 ≤ b.size) (hs : o.toNat + 8 ≤ USize.size) : UInt64 :=
  (b.uget o (by omega)).toUInt64 ^^^
    (b.uget (o + 1) (by rw [toNat_add_lit o 1 (by omega)]; omega)).toUInt64 <<< 8 ^^^
    (b.uget (o + 2) (by rw [toNat_add_lit o 2 (by omega)]; omega)).toUInt64 <<< 16 ^^^
    (b.uget (o + 3) (by rw [toNat_add_lit o 3 (by omega)]; omega)).toUInt64 <<< 24 ^^^
    (b.uget (o + 4) (by rw [toNat_add_lit o 4 (by omega)]; omega)).toUInt64 <<< 32 ^^^
    (b.uget (o + 5) (by rw [toNat_add_lit o 5 (by omega)]; omega)).toUInt64 <<< 40 ^^^
    (b.uget (o + 6) (by rw [toNat_add_lit o 6 (by omega)]; omega)).toUInt64 <<< 48 ^^^
    (b.uget (o + 7) (by rw [toNat_add_lit o 7 (by omega)]; omega)).toUInt64 <<< 56

/-- `wr64` with no bounds check: the caller proves bytes `o, …, o + 7` inside `b` (`wr64P_eq`). -/
@[inline] def wr64P (b : ByteArray) (o : USize) (w : UInt64) (h : o.toNat + 8 ≤ b.size)
    (hs : o.toNat + 8 ≤ USize.size) : ByteArray :=
  let b1 := b.uset o w.toUInt8 (by omega)
  have h1 : b1.size = b.size := size_uset ..
  let b2 := b1.uset (o + 1) (w >>> 8).toUInt8 (by rw [toNat_add_lit o 1 (by omega)]; omega)
  have h2 : b2.size = b.size := (size_uset ..).trans h1
  let b3 := b2.uset (o + 2) (w >>> 16).toUInt8 (by rw [toNat_add_lit o 2 (by omega)]; omega)
  have h3 : b3.size = b.size := (size_uset ..).trans h2
  let b4 := b3.uset (o + 3) (w >>> 24).toUInt8 (by rw [toNat_add_lit o 3 (by omega)]; omega)
  have h4 : b4.size = b.size := (size_uset ..).trans h3
  let b5 := b4.uset (o + 4) (w >>> 32).toUInt8 (by rw [toNat_add_lit o 4 (by omega)]; omega)
  have h5 : b5.size = b.size := (size_uset ..).trans h4
  let b6 := b5.uset (o + 5) (w >>> 40).toUInt8 (by rw [toNat_add_lit o 5 (by omega)]; omega)
  have h6 : b6.size = b.size := (size_uset ..).trans h5
  let b7 := b6.uset (o + 6) (w >>> 48).toUInt8 (by rw [toNat_add_lit o 6 (by omega)]; omega)
  have h7 : b7.size = b.size := (size_uset ..).trans h6
  b7.uset (o + 7) (w >>> 56).toUInt8 (by rw [toNat_add_lit o 7 (by omega)]; omega)

theorem rd64P_eq (b : ByteArray) (o : USize) (h : o.toNat + 8 ≤ b.size) (hs : o.toNat + 8 ≤ USize.size) :
    rd64P b o h hs = rd64 b o := by
  simp only [rd64P, rd64]
  rw [getU_pos b o (by omega), getU_pos b (o + 1) (by rw [toNat_add_lit o 1 (by omega)]; omega),
    getU_pos b (o + 2) (by rw [toNat_add_lit o 2 (by omega)]; omega),
    getU_pos b (o + 3) (by rw [toNat_add_lit o 3 (by omega)]; omega),
    getU_pos b (o + 4) (by rw [toNat_add_lit o 4 (by omega)]; omega),
    getU_pos b (o + 5) (by rw [toNat_add_lit o 5 (by omega)]; omega),
    getU_pos b (o + 6) (by rw [toNat_add_lit o 6 (by omega)]; omega),
    getU_pos b (o + 7) (by rw [toNat_add_lit o 7 (by omega)]; omega)]

theorem wr64P_eq (b : ByteArray) (o : USize) (w : UInt64) (h : o.toNat + 8 ≤ b.size) (hs : o.toNat + 8 ≤ USize.size) :
    wr64P b o w h hs = wr64 b o w := by
  unfold wr64P wr64
  -- outermost first, so no rewritten term sits under a later write's bound proof
  simp only [↓uset_eq_setU]

theorem size_wr64P (b : ByteArray) (o : USize) (w : UInt64) (h : o.toNat + 8 ≤ b.size) (hs : o.toNat + 8 ≤ USize.size) :
    (wr64P b o w h hs).size = b.size := by
  rw [wr64P_eq, size_wr64]

/-- `b` with column `c` (`colB`) increased by `x`, its bounds checked once (`addB_eq`). -/
@[inline] def addB (b : ByteArray) (c : Nat) (x : F128) (hc : 16 * c + 16 < USize.size) : ByteArray :=
  if h : 16 * c + 16 ≤ b.size then
    have ho : (16 * c).toUSize.toNat = 16 * c := toNat_toUSize _ (by omega)
    have ho8 : ((16 * c).toUSize + 8).toNat = 16 * c + 8 := by rw [toNat_add_lit _ 8 (by omega), ho]
    let lo := rd64P b (16 * c).toUSize (by omega) (by omega) ^^^ x.lo
    let hi := rd64P b ((16 * c).toUSize + 8) (by omega) (by omega) ^^^ x.hi
    wr64P (wr64P b (16 * c).toUSize lo (by omega) (by omega)) ((16 * c).toUSize + 8) hi
      (by rw [size_wr64P]; omega) (by omega)
  else b

/-- `foldRow` on an unboxed accumulator (`colB`). -/
def foldRowB (m : Sparse) (i : Nat) (ei : F128) (hw : 16 * m.width + 16 < USize.size) :
    Nat → Nat → ByteArray → ByteArray
  | 0, _, acc => acc
  | fuel + 1, t, acc =>
    let c := m.col i t
    foldRowB m i ei hw fuel (t + 1) (if h : c < m.width then addB acc c ei (by omega) else acc)

/-- `foldRows` on an unboxed accumulator (`colB`). -/
def foldRowsB (m : Sparse) (e : Array F128) (hw : 16 * m.width + 16 < USize.size) : Nat → Nat → ByteArray → ByteArray
  | 0, _, acc => acc
  | fuel + 1, i, acc =>
    let ei := e[i]!
    foldRowsB m e hw fuel (i + 1) (if ei.isZero then acc else m.foldRowB i ei hw m.len[i]! 0 acc)

/-- `foldT` with its accumulator unboxed (`foldT_eq_fast`): no `F128` is allocated until the result. -/
def foldTFast (m : Sparse) (e : Array F128) : Array F128 :=
  if hw : 16 * m.width + 16 < USize.size then
    let acc := m.foldRowsB e hw (min m.rows e.size) 0 (zeros (16 * m.width))
    (Array.range m.width).map fun c =>
      let v := colB acc c
      if v.lo == 0 && v.hi == 0 then 0 else v
  else m.foldT e

theorem zerosF_spec : ∀ (fuel n : Nat), n < 2 ^ fuel →
    (zerosF fuel n).size = n ∧ ∀ p : Nat, (zerosF fuel n)[p]! = 0
  | 0, n, h => by
    have : n = 0 := by simp at h; omega
    subst this
    exact ⟨rfl, fun p => by simp [zerosF]; rfl⟩
  | fuel + 1, n, h => by
    unfold zerosF
    by_cases h0 : n = 0
    · subst h0
      exact ⟨rfl, fun p => by simp; rfl⟩
    · have ih := zerosF_spec fuel (n / 2) (by rw [Nat.pow_succ] at h; omega)
      have hz : ∀ p : Nat, (zerosF fuel (n / 2) ++ zerosF fuel (n / 2))[p]! = 0 := by
        intro p
        by_cases hp : p < (zerosF fuel (n / 2) ++ zerosF fuel (n / 2)).size
        · rw [getElem!_pos _ p hp]
          by_cases hl : p < (zerosF fuel (n / 2)).size
          · rw [ByteArray.getElem_append_left hl, ← getElem!_pos _ p hl, ih.2]
          · rw [ByteArray.getElem_append_right (by omega), ← getElem!_pos _ _ (by rw [ByteArray.size_append] at hp; omega),
              ih.2]
        · rw [getElem!_neg _ p hp]; rfl
      simp only [h0, beq_iff_eq, ↓reduceIte]
      split
      · exact ⟨by rw [ByteArray.size_append, ih.1]; omega, hz⟩
      · refine ⟨by rw [ByteArray.size_push, ByteArray.size_append, ih.1]; omega, fun p => ?_⟩
        rw [ByteArray.getElem!_push]
        split
        · rfl
        · exact hz p

theorem zeros_spec (n : Nat) : (zeros n).size = n ∧ ∀ p : Nat, (zeros n)[p]! = 0 :=
  zerosF_spec _ n (Nat.lt_log2_self)

theorem getElem!_setU (b : ByteArray) (i : USize) (x : UInt8) (p : Nat) (h : i.toNat < b.size) :
    (setU b i x)[p]! = if i.toNat = p then x else b[p]! := by
  rw [setU_eq, ByteArray.getElem!_set! b i.toNat x p h]

/-- `rd64` reads bytes `o, …, o + 7` by position. -/
theorem rd64_eq (b : ByteArray) (o : USize) (h : o.toNat + 8 ≤ USize.size) :
    rd64 b o = b[o.toNat]!.toUInt64 ^^^ b[o.toNat + 1]!.toUInt64 <<< 8 ^^^ b[o.toNat + 2]!.toUInt64 <<< 16 ^^^
      b[o.toNat + 3]!.toUInt64 <<< 24 ^^^ b[o.toNat + 4]!.toUInt64 <<< 32 ^^^ b[o.toNat + 5]!.toUInt64 <<< 40 ^^^
      b[o.toNat + 6]!.toUInt64 <<< 48 ^^^ b[o.toNat + 7]!.toUInt64 <<< 56 := by
  simp only [rd64, getU_eq, toNat_add_lit o 1 (by omega), toNat_add_lit o 2 (by omega), toNat_add_lit o 3 (by omega),
    toNat_add_lit o 4 (by omega), toNat_add_lit o 5 (by omega), toNat_add_lit o 6 (by omega),
    toNat_add_lit o 7 (by omega)]

/-- `wr64`'s bytes by position, inside the array. -/
theorem wr64_get (b : ByteArray) (o : USize) (w : UInt64) (h : o.toNat + 8 ≤ USize.size) (hb : o.toNat + 8 ≤ b.size)
    (p : Nat) :
    (wr64 b o w)[p]! = if p = o.toNat then w.toUInt8 else if p = o.toNat + 1 then (w >>> 8).toUInt8
      else if p = o.toNat + 2 then (w >>> 16).toUInt8 else if p = o.toNat + 3 then (w >>> 24).toUInt8
      else if p = o.toNat + 4 then (w >>> 32).toUInt8 else if p = o.toNat + 5 then (w >>> 40).toUInt8
      else if p = o.toNat + 6 then (w >>> 48).toUInt8 else if p = o.toNat + 7 then (w >>> 56).toUInt8 else b[p]! := by
  simp only [wr64]
  rw [getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 7 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 6 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 5 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 4 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 3 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 2 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by simp only [size_setU]; rw [toNat_add_lit o 1 (by omega)]; omega),
    getElem!_setU _ _ _ _ (by omega)]
  simp only [toNat_add_lit o 1 (by omega), toNat_add_lit o 2 (by omega), toNat_add_lit o 3 (by omega),
    toNat_add_lit o 4 (by omega), toNat_add_lit o 5 (by omega), toNat_add_lit o 6 (by omega),
    toNat_add_lit o 7 (by omega)]
  by_cases h0 : p = o.toNat <;> by_cases h1 : p = o.toNat + 1 <;> by_cases h2 : p = o.toNat + 2 <;>
    by_cases h3 : p = o.toNat + 3 <;> by_cases h4 : p = o.toNat + 4 <;> by_cases h5 : p = o.toNat + 5 <;>
    by_cases h6 : p = o.toNat + 6 <;> by_cases h7 : p = o.toNat + 7 <;>
    simp only [h0, h1, h2, h3, h4, h5, h6, h7, ↓reduceIte] <;> first | omega | (simp only [eq_comm] at *; simp_all)

theorem mod256_shl (w : UInt64) (k : Nat) :
    ((w >>> (8 * k).toUInt64).toUInt8.toUInt64 <<< (8 * k).toUInt64) = F128.bslot w k := by
  rw [UInt64.toUInt64_toUInt8]
  unfold F128.bslot
  congr 1
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_mod, UInt64.toNat_and]
  exact (Nat.and_two_pow_sub_one_eq_mod _ 8).symm

theorem ite_ne {α : Type} {p q : Nat} (h : p ≠ q) (a b : α) : (if p = q then a else b) = b := ite_eq_right h

theorem rd64_wr64 (b : ByteArray) (o : USize) (w : UInt64) (h : o.toNat + 8 ≤ USize.size)
    (hb : o.toNat + 8 ≤ b.size) : rd64 (wr64 b o w) o = w := by
  rw [rd64_eq _ o h]
  simp (disch := omega) only [wr64_get b o w h hb, ite_ne, ↓reduceIte]
  have e0 := mod256_shl w 0; have e1 := mod256_shl w 1
  have e2 := mod256_shl w 2; have e3 := mod256_shl w 3
  have e4 := mod256_shl w 4; have e5 := mod256_shl w 5
  have e6 := mod256_shl w 6; have e7 := mod256_shl w 7
  simp only [Nat.mul_zero, Nat.mul_one, Nat.reduceMul] at e0 e1 e2 e3 e4 e5 e6 e7
  rw [show Nat.toUInt64 0 = 0 from rfl, UInt64.shiftRight_zero, UInt64.shiftLeft_zero] at e0
  rw [show Nat.toUInt64 8 = 8 from rfl] at e1; rw [show Nat.toUInt64 16 = 16 from rfl] at e2
  rw [show Nat.toUInt64 24 = 24 from rfl] at e3; rw [show Nat.toUInt64 32 = 32 from rfl] at e4
  rw [show Nat.toUInt64 40 = 40 from rfl] at e5; rw [show Nat.toUInt64 48 = 48 from rfl] at e6
  rw [show Nat.toUInt64 56 = 56 from rfl] at e7
  rw [e0, e1, e2, e3, e4, e5, e6, e7]
  exact (F128.u64_bytes w).symm

theorem rd64_wr64_ne (b : ByteArray) (o o' : USize) (w : UInt64) (h : o.toNat + 8 ≤ USize.size)
    (h' : o'.toNat + 8 ≤ USize.size) (hb : o.toNat + 8 ≤ b.size) (hd : o.toNat + 8 ≤ o'.toNat ∨ o'.toNat + 8 ≤ o.toNat) :
    rd64 (wr64 b o w) o' = rd64 b o' := by
  rw [rd64_eq _ o' h', rd64_eq _ o' h']
  simp only [wr64_get b o w h hb]
  simp (disch := omega) only [ite_ne]

/-- An accumulator and its unboxed form agree: `colB` gives each of the `w` columns. -/
def Agree (w : Nat) (acc : Array F128) (bs : ByteArray) : Prop :=
  acc.size = w ∧ bs.size = 16 * w ∧ ∀ c < w, colB bs c = acc[c]!

theorem toNat_col (c w : Nat) (hc : c < w) (hw : 16 * w + 16 < USize.size) : (16 * c).toUSize.toNat = 16 * c :=
  toNat_toUSize _ (by omega)

theorem addB_eq (b : ByteArray) (c : Nat) (x : F128) (hc : 16 * c + 16 < USize.size) (h : 16 * c + 16 ≤ b.size) :
    addB b c x hc = wr64 (wr64 b (16 * c).toUSize (rd64 b (16 * c).toUSize ^^^ x.lo)) ((16 * c).toUSize + 8)
      (rd64 b ((16 * c).toUSize + 8) ^^^ x.hi) := by
  unfold addB
  rw [dite_eq_left h]
  simp only [↓wr64P_eq, ↓rd64P_eq]

theorem agree_addB (w : Nat) (acc : Array F128) (bs : ByteArray) (c : Nat) (x : F128) (hw : 16 * w + 16 < USize.size)
    (hc : c < w) (hcs : 16 * c + 16 < USize.size) (ha : Agree w acc bs) :
    Agree w (acc.modify c (· + x)) (addB bs c x hcs) := by
  obtain ⟨hs, hb, hv⟩ := ha
  rw [addB_eq bs c x hcs (by omega)]
  have ho := toNat_col c w hc hw
  have ho8 : ((16 * c).toUSize + 8).toNat = 16 * c + 8 := by rw [toNat_add_lit _ 8 (by omega), ho]
  refine ⟨by rw [Array.size_modify, hs], by simp only [size_wr64, hb], fun c' hc' => ?_⟩
  have ho' := toNat_col c' w hc' hw
  have ho8' : ((16 * c').toUSize + 8).toNat = 16 * c' + 8 := by rw [toNat_add_lit _ 8 (by omega), ho']
  have hsz : ∀ v, (wr64 bs (16 * c).toUSize v).size = 16 * w := fun v => by rw [size_wr64, hb]
  by_cases hcc : c = c'
  · subst hcc
    rw [getElem!_pos _ c (by rw [Array.size_modify]; omega), Array.getElem_modify_self, ← getElem!_pos acc c (by omega),
      ← hv c hc]
    simp only [colB]
    rw [rd64_wr64_ne _ _ _ _ (by omega) (by omega) (by rw [hsz]; omega) (by omega),
      rd64_wr64 _ _ _ (by omega) (by rw [ho]; omega), rd64_wr64 _ _ _ (by omega) (by rw [hsz]; omega)]
    rfl
  · rw [getElem!_pos _ c' (by rw [Array.size_modify]; omega), Array.getElem_modify_of_ne hcc,
      ← getElem!_pos acc c' (by omega), ← hv c' hc']
    simp only [colB]
    have d1 : 16 * c + 8 + 8 ≤ 16 * c' ∨ 16 * c' + 8 ≤ 16 * c + 8 := by omega
    have d2 : 16 * c + 8 ≤ 16 * c' ∨ 16 * c' + 8 ≤ 16 * c := by omega
    have d3 : 16 * c + 8 + 8 ≤ 16 * c' + 8 ∨ 16 * c' + 8 + 8 ≤ 16 * c + 8 := by omega
    have d4 : 16 * c + 8 ≤ 16 * c' + 8 ∨ 16 * c' + 8 + 8 ≤ 16 * c := by omega
    rw [rd64_wr64_ne _ _ _ _ (by omega) (by omega) (by rw [hsz]; omega) (by rw [ho8, ho']; exact d1),
      rd64_wr64_ne _ _ _ _ (by omega) (by omega) (by rw [ho]; omega) (by rw [ho, ho']; exact d2),
      rd64_wr64_ne _ _ _ _ (by omega) (by omega) (by rw [hsz]; omega) (by rw [ho8, ho8']; exact d3),
      rd64_wr64_ne _ _ _ _ (by omega) (by omega) (by rw [ho]; omega) (by rw [ho, ho8']; exact d4)]

theorem agree_foldRow (m : Sparse) (i : Nat) (ei : F128) (hw : 16 * m.width + 16 < USize.size) :
    ∀ fuel t acc bs, Agree m.width acc bs → Agree m.width (m.foldRow i ei fuel t acc) (m.foldRowB i ei hw fuel t bs)
  | 0, _, _, _, h => h
  | fuel + 1, t, acc, bs, h => by
    simp only [foldRow, foldRowB]
    apply agree_foldRow m i ei hw fuel (t + 1)
    split
    · exact agree_addB _ _ _ _ _ hw (by assumption) _ h
    · rename_i hc
      have : acc.modify (m.col i t) (· + ei) = acc := by
        apply Array.ext (by rw [Array.size_modify])
        intro j h1 h2
        rw [Array.getElem_modify_of_ne (by obtain ⟨hs, _⟩ := h; omega)]
      rw [this]; exact h

theorem agree_foldRows (m : Sparse) (e : Array F128) (hw : 16 * m.width + 16 < USize.size) :
    ∀ fuel i acc bs, Agree m.width acc bs → Agree m.width (m.foldRows e fuel i acc) (m.foldRowsB e hw fuel i bs)
  | 0, _, _, _, h => h
  | fuel + 1, i, acc, bs, h => by
    simp only [foldRows, foldRowsB]
    apply agree_foldRows m e hw fuel (i + 1)
    split
    · exact h
    · exact agree_foldRow m i _ hw _ _ _ _ h

/-- Compiled code runs `foldTFast`; the kernel and every proof see `foldT`. `@[csimp]` rewrites only code compiled after
it, so it stays above every caller of `foldT` but `foldTFast`'s own fallback. -/
@[csimp] theorem foldT_eq_fast : @foldT = @foldTFast := by
  funext m e
  unfold foldTFast
  split
  · rename_i hw
    have z := zeros_spec (16 * m.width)
    have h0 : Agree m.width (Array.replicate m.width 0) (zeros (16 * m.width)) := by
      refine ⟨Array.size_replicate .., z.1, fun c hc => ?_⟩
      rw [getElem!_pos _ c (by simp; omega), Array.getElem_replicate]
      have ho := toNat_col c _ hc hw
      have ho8 : ((16 * c).toUSize + 8).toNat = 16 * c + 8 := by rw [toNat_add_lit _ 8 (by omega), ho]
      simp only [colB]
      rw [rd64_eq _ _ (by omega), rd64_eq _ _ (by omega)]
      simp only [z.2]
      rfl
    obtain ⟨hs, _, hv⟩ := agree_foldRows m e hw (min m.rows e.size) 0 _ _ h0
    apply Array.ext (by simp [foldT, hs])
    intro c h1 h2
    simp only [Array.getElem_map, Array.getElem_range]
    have hc : c < m.width := by simpa using h2
    have := hv c hc
    rw [getElem!_pos _ c h1] at this
    unfold foldT
    rw [← this]
    split
    · rename_i hz
      simp only [Bool.and_eq_true, beq_iff_eq] at hz
      cases hcol : colB (m.foldRowsB e hw (min m.rows e.size) 0 (zeros (16 * m.width))) c
      rw [hcol] at hz
      simp only at hz
      rw [hz.1, hz.2]
      rfl
    · rfl
  · rfl

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
