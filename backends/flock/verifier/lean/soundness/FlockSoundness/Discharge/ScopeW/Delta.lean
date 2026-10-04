import FlockSoundness.Discharge.ScopeW.Scope
import FlockSoundness.Discharge.Hidden.Wide.RowBc

set_option Elab.async false

/-!
# Δ at the widened scope's rows: frozen statements

`HmRow.delta` without a typed template, pair by pair, for rows of any shape (`delta_flat_eqW`, no shape hypothesis):
the slot constants, each row's copies (`rowCopiesW`: a segmented row's by `rowCopiesSeg`, every other row's by
`Hidden.rowCopiesOf`), the units' leaf and cut bits and the wires' bits. A segmented row's copies follow
`HmRow.chainStart` and `HmRow.msgBit` compression by compression (its `hm96` slot last): each chain start's `cv` its
midstate (`startMid`), each constant message bit from the pin or the forced zero, each digest bit from a segment's last
output (`msgCopy`).

What `HmRow.msgBit` and `HmRow.chainStart` give at each shape (N1):

* a row of whole blocks (`RowWhole`: `RowV1` with `ParseFacts`, or `RowV2Whole`): row bits before the `hm96` slot, the
  padding block in it (`msgBit_whole_row`, `msgBit_whole_pad`);
* `RowV2Short`: the `hm96` slot's block is the row's bytes, then its padding (`msgBit_short_row`, `msgBit_short_tail`);
* `RowSegWhole`: each segment's chain starts at `segFirst` (`chainStart_seg`, `segFirst_succ`, `segs_whole`), holds the
  segment's bits, then a padding block (`msgBit_seg_row`, `msgBit_seg_pad`); the outer message's chain holds the
  segments' digests (`msgBit_seg_digest`) and ends in the `hm96` slot (`seg_chunks`) with its padding
  (`msgBit_seg_tail`).

The final block in the `hm96` slot (`RowV2Short`'s, and a segmented row's outer one when `s` is odd) is data bytes then
padding bytes: `tailBit` reads the padding's bits, `tailBit_words` puts them in the block's words, and `fill_short`,
`fill_seg` say the two parts fill the block; `blockBit_col` and `blockBit_of_col` invert a block column.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm

/-! ## Δ's copies at a segmented row -/

/-- A row's segments, `(bits, compressions)` each; none for a row of no segments. -/
abbrev segsOf (p : Port) : Array (ℕ × ℕ) := p.segs.getD #[]

/-- A segmented row's segment `j`'s first compression (`j = s`: the outer message's): the compressions before it. -/
def segFirst (p : Port) (j : ℕ) : ℕ := ((segsOf p).extract 0 j).foldl (· + ·.2) 0

/-- The midstate at chain start `j` (`HmRow.chainStart`): segment `j`'s, or past the segments the row prefix's (the
outer message's, or a row of no segments' one chain's). -/
def startMid (p : Port) (j : ℕ) : Array UInt64 :=
  if j < (segsOf p).size then HmRow.segMidstate p j else HmRow.midstateOf p

/-- VU `g`'s row `p`: compression `k`'s `cv` input's first column; past its compressions, its `hm96` slot's. -/
def cvAt (c : Circuit) (g p k : ℕ) : ℕ :=
  if k < ((HmRow.rowSlots c.ports).2[p]!).size then shaBase c g p k + shaIn c (2 * shaSub c p k)
  else c.slotCol c.hmNet g p + hmIn c 0

/-- VU `g`'s row `p`: compression `k`'s `msg` input's first column; past its compressions, its `hm96` slot's `pad`. -/
def msgAt (c : Circuit) (g p k : ℕ) : ℕ :=
  if k < ((HmRow.rowSlots c.ports).2[p]!).size then shaBase c g p k + shaIn c (2 * shaSub c p k + 1)
  else c.slotCol c.hmNet g p + hmIn c 1

/-- VU `g`'s row `p`: compression `k`'s output's first column. -/
def outAt (c : Circuit) (g p k : ℕ) : ℕ := shaBase c g p k + shaOut c (shaSub c p k)

/-- Δ's copy at message column `x` for the bit `HmRow.msgBit` gives there: a constant from the pin or the forced zero, a
digest bit from column `col` of compression `kk`'s output (`out kk + col`), none at a row bit. -/
def msgCopy (c : Circuit) (pin x : ℕ) (out : ℕ → ℕ) : HmRow.MsgBit → List (ℕ × ℕ)
  | .row _ => []
  | .const v => [(x, if v then pin else zeroCol c)]
  | .digest kk col => [(x, out kk + col)]

/-- Δ's copies into VU `g`'s segmented row `p`, compression by compression, its `hm96` slot last: a chain start's `cv`
its midstate, then each message bit's copy. -/
def rowCopiesSeg (c : Circuit) (pin g p : ℕ) : List (ℕ × ℕ) :=
  (List.range (((HmRow.rowSlots c.ports).2[p]!).size + 1)).flatMap fun k =>
    ((HmRow.chainStart c.ports[p]! k).toList.flatMap fun j =>
      (List.range 512).map fun t => (cvAt c g p k + t, constSrc c pin (startMid c.ports[p]! j) t)) ++
    (List.range 1024).flatMap fun t => msgCopy c pin (msgAt c g p k + t) (outAt c g p) (HmRow.msgBit c.ports[p]! k t)

/-- Δ's copies into the rows, for rows of any shape. -/
def rowCopiesW (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p =>
    if c.ports[p]!.segs.isSome then rowCopiesSeg c pin g p else Hidden.rowCopiesOf c pin g p

/-- Δ's pairs without a typed template, for rows of any shape. -/
def flatPairsW (c : Circuit) (pin : ℕ) : List (ℕ × ℕ) :=
  constPairs c pin ++ (rowCopiesW c pin).flatMap pair2 ++ Flat.leafPairs c ++ Flat.cutPairs c ++ Flat.wirePairs c

/-- **Δ without a typed template, pair by pair, for rows of any shape**: the slot constants' pairs, each row's copies,
the units' leaf and cut bits and the wires' bits. -/
theorem delta_flat_eqW (c : Circuit) (pin : ℕ) (ht : c.typed = none) :
    (HmRow.delta c pin).1.toList = flatPairsW c pin := by
  sorry

theorem cvAt_lt {c : Circuit} {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) :
    cvAt c g p k = shaBase c g p k + shaIn c (2 * shaSub c p k) := by
  simp only [cvAt, rowSlots_size hp, hk, ↓reduceIte]

theorem cvAt_hm {c : Circuit} {g p : ℕ} (hp : p < c.ports.size) :
    cvAt c g p c.ports[p]!.chunks = c.slotCol c.hmNet g p + hmIn c 0 := by
  simp only [cvAt, rowSlots_size hp, Nat.lt_irrefl, ↓reduceIte]

theorem msgAt_lt {c : Circuit} {g p k : ℕ} (hp : p < c.ports.size) (hk : k < c.ports[p]!.chunks) :
    msgAt c g p k = shaBase c g p k + shaIn c (2 * shaSub c p k + 1) := by
  simp only [msgAt, rowSlots_size hp, hk, ↓reduceIte]

theorem msgAt_hm {c : Circuit} {g p : ℕ} (hp : p < c.ports.size) :
    msgAt c g p c.ports[p]!.chunks = c.slotCol c.hmNet g p + hmIn c 1 := by
  simp only [msgAt, rowSlots_size hp, Nat.lt_irrefl, ↓reduceIte]

/-- **Row `p`'s message byte `byte`, bit `b`** (`Typed.msgCol`, Δ's leaves' source): column `64 (o / 8) + 56 − 8 (o % 8) + b`
of compression `k`'s message (`msgAt`), `(k, o) = HmRow.byteAt`. -/
theorem msgCol_eqW (c : Circuit) (g p byte b : ℕ) :
    Typed.msgCol c g p byte b = msgAt c g p (HmRow.byteAt c.ports[p]! byte).1 +
      (64 * ((HmRow.byteAt c.ports[p]! byte).2 / 8) + 56 - 8 * ((HmRow.byteAt c.ports[p]! byte).2 % 8) + b) := by
  sorry

theorem byteAt_unseg {p : Port} (hs : p.segs = none) (byte : ℕ) : HmRow.byteAt p byte = (byte / 128, byte % 128) := by
  simp [HmRow.byteAt, hs]

/-! ## Rows of whole blocks -/

/-- An unsegmented row of `16 · words` bits in `chunks ≥ 1` whole blocks: a `sha512/row/v1` row (`rowWhole_of_v1`) or a
`sha512/row/v2` row of whole blocks (`rowWhole_of_v2`). -/
def RowWhole (p : Port) : Prop := p.segs = none ∧ p.bits = 16 * p.words ∧ 0 < p.chunks ∧ 2 * p.words = 128 * p.chunks

theorem rowWhole_of_v2 {p : Port} (h : RowV2Whole p) : RowWhole p := ⟨h.2.1, h.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2⟩

theorem rowWhole_of_v1 {tags : Tags} {c : Circuit} (P : ParseFacts tags c) {p : Port} (hp : p ∈ c.ports.toList)
    (h : RowV1 p) : RowWhole p :=
  ⟨P.v1Segs p hp h.1, h.2.1, h.2.2, P.portWords p hp h.1⟩

/-- **A whole row's compressions hold its bits**: compression `k`'s message bit `t` is the row's bit
`8 (128 k + byte) + bit` of the block column's byte and bit. -/
theorem msgBit_whole_row {p : Port} (h : RowWhole p) {k t : ℕ} (hk : k < p.chunks) (ht : t < 1024) :
    HmRow.msgBit p k t = .row (8 * (128 * k + (HmRow.blockBit t).1) + (HmRow.blockBit t).2) := by
  sorry

/-- **A whole row's `hm96` slot holds its padding block** (`HmRow.padWords`). -/
theorem msgBit_whole_pad {p : Port} (h : RowWhole p) {t : ℕ} (ht : t < 1024) :
    HmRow.msgBit p p.chunks t = .const (wordsBit (HmRow.padWords p.words) t) := by
  sorry

/-- An unsegmented row's one chain starts at its first compression (its `hm96` slot, for a row of none). -/
theorem chainStart_unseg {p : Port} (hs : p.segs = none) (k : ℕ) :
    HmRow.chainStart p k = if k = 0 then some 0 else none := by
  unfold HmRow.chainStart
  rw [hs]
  by_cases hk : k = 0 <;> simp [hk]

/-! ## The final block in the `hm96` slot -/

/-- Bit `t` (a block column) of a message block whose bytes from byte `r` on are `tl`'s: bit `(blockBit t).2` of `tl`'s
byte `(blockBit t).1 − r`. -/
def tailBit (tl : ByteArray) (r t : ℕ) : Bool :=
  ((tl.get! ((HmRow.blockBit t).1 - r)).toNat >>> (HmRow.blockBit t).2) % 2 == 1

/-- A block column is its byte's and bit's message column. -/
theorem blockBit_col {t : ℕ} (ht : t < 1024) :
    64 * ((HmRow.blockBit t).1 / 8) + 56 - 8 * ((HmRow.blockBit t).1 % 8) + (HmRow.blockBit t).2 = t := by
  sorry

/-- Byte `o`'s bit `b`'s message column is the block column of `(o, b)`. -/
theorem blockBit_of_col {o b : ℕ} (ho : o < 128) (hb : b < 8) :
    HmRow.blockBit (64 * (o / 8) + 56 - 8 * (o % 8) + b) = (o, b) := by
  sorry

/-- SHA-512's padding after `total` bytes: `0x80`, zeros to 112 mod 128, 16 length bytes. -/
theorem padTail_size (total : ℕ) : (HmRow.padTail total).size = (239 - total % 128) % 128 + 17 := by
  sorry

/-- **A block of `pre`'s bytes then `tl`'s**: at a column of `tl`'s, the block's word bit is `tl`'s bit. -/
theorem tailBit_words {pre tl : ByteArray} {t : ℕ} (hsz : pre.size + tl.size = 128) (ht : t < 1024)
    (hr : pre.size ≤ (HmRow.blockBit t).1) :
    tailBit tl pre.size t = wordsBit (HmRow.wordsBE (pre ++ tl) 16) t := by
  sorry

theorem tailOf_short {p : Port} (h : RowV2Short p) : HmRow.tailOf p = HmRow.padTail (128 + 2 * p.words) := by
  sorry

/-- **A short row's bytes and padding fill its `hm96` slot's block.** -/
theorem fill_short {p : Port} (h : RowV2Short p) : 2 * p.words + (HmRow.tailOf p).size = 128 := by
  sorry

theorem tailOf_seg {p : Port} (h : RowSegWhole p) :
    HmRow.tailOf p = HmRow.padTail (128 + 64 * (segsOf p).size) := by
  sorry

/-- **A segmented row's outer message's last digest bytes and padding fill its `hm96` slot's block**: `64 s mod 128`
digest bytes (`64` for `s` odd, none for `s` even), then the padding. -/
theorem fill_seg {p : Port} (h : RowSegWhole p) : 64 * (segsOf p).size % 128 + (HmRow.tailOf p).size = 128 := by
  sorry

/-! ## Short rows -/

/-- **A short row's `hm96` slot holds its bits**, at the block's first `2 · words` bytes. -/
theorem msgBit_short_row {p : Port} (h : RowV2Short p) {t : ℕ} (ht : t < 1024)
    (hr : (HmRow.blockBit t).1 < 2 * p.words) :
    HmRow.msgBit p p.chunks t = .row (8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2) := by
  sorry

/-- **A short row's `hm96` slot holds its padding** past its bytes. -/
theorem msgBit_short_tail {p : Port} (h : RowV2Short p) {t : ℕ} (ht : t < 1024)
    (hr : 2 * p.words ≤ (HmRow.blockBit t).1) :
    HmRow.msgBit p p.chunks t = .const (tailBit (HmRow.tailOf p) (2 * p.words) t) := by
  sorry

/-! ## Segmented rows -/

theorem segFirst_zero (p : Port) : segFirst p 0 = 0 := by
  simp [segFirst]

/-- A segmented row's byte `byte`: block `byte % 1024 / 128` of segment `byte / 1024`. -/
theorem byteAt_seg {p : Port} {segs : Array (ℕ × ℕ)} (hs : p.segs = some segs) (byte : ℕ) :
    HmRow.byteAt p byte = (segFirst p (byte / 1024) + byte % 1024 / 128, byte % 128) := by
  simp [HmRow.byteAt, segFirst, segsOf, hs]

theorem segFirst_succ (p : Port) {j : ℕ} (hj : j < (segsOf p).size) :
    segFirst p (j + 1) = segFirst p j + ((segsOf p)[j]!).2 := by
  sorry

/-- **A whole-block segmented row's segments**: segment `j`'s bits a positive multiple of 1024, its compressions its
data blocks and a padding block. -/
theorem segs_whole {p : Port} (h : RowSegWhole p) {j : ℕ} (hj : j < (segsOf p).size) :
    0 < ((segsOf p)[j]!).1 ∧ ((segsOf p)[j]!).1 % 1024 = 0 ∧ ((segsOf p)[j]!).2 = ((segsOf p)[j]!).1 / 1024 + 1 := by
  sorry

/-- **The outer message ends in the `hm96` slot**: its compressions run from `segFirst p s` to the row's `chunks`. -/
theorem seg_chunks {p : Port} (h : RowSegWhole p) :
    p.chunks = segFirst p (segsOf p).size + 64 * (segsOf p).size / 128 := by
  sorry

/-- **Each segment's chain, and the outer message's, starts at its first compression.** -/
theorem chainStart_seg {p : Port} (h : RowSegWhole p) {j : ℕ} (hj : j ≤ (segsOf p).size) :
    HmRow.chainStart p (segFirst p j) = some j := by
  sorry

/-- **No other compression starts a chain**, for a row of any shape. -/
theorem chainStart_none {p : Port} {k : ℕ} (hk : ∀ j ≤ (segsOf p).size, segFirst p j ≠ k) :
    HmRow.chainStart p k = none := by
  sorry

/-- **A segment's data blocks hold its bits**: block `i` of segment `j`, bit `t`, is the row's bit
`8192 j + 1024 i + 8 · byte + bit`. -/
theorem msgBit_seg_row {p : Port} (h : RowSegWhole p) {j i t : ℕ} (hj : j < (segsOf p).size)
    (hi : i < ((segsOf p)[j]!).1 / 1024) (ht : t < 1024) :
    HmRow.msgBit p (segFirst p j + i) t =
      .row (HmRow.ROW_SEG_BITS * j + 1024 * i + 8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2) := by
  sorry

/-- **A segment's last block is a padding block of its own.** -/
theorem msgBit_seg_pad {p : Port} (h : RowSegWhole p) {j t : ℕ} (hj : j < (segsOf p).size) (ht : t < 1024) :
    HmRow.msgBit p (segFirst p j + ((segsOf p)[j]!).1 / 1024) t =
      .const (wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf p)[j]!).1 / 8)) 16) t) := by
  sorry

/-- **The outer message's byte `e < 64 s` is a digest byte**: bit `b` of byte `e % 64` of segment `e / 64`'s digest,
column `64 (q / 8) + 56 − 8 (q % 8) + b` (`q = e % 64`) of that segment's last compression's output. -/
theorem msgBit_seg_digest {p : Port} (h : RowSegWhole p) {e b : ℕ} (he : e < 64 * (segsOf p).size) (hb : b < 8) :
    HmRow.msgBit p (segFirst p (segsOf p).size + e / 128) (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b) =
      .digest (segFirst p (e / 64 + 1) - 1) (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b) := by
  sorry

/-- **The `hm96` slot's block past the digests is the outer message's padding.** -/
theorem msgBit_seg_tail {p : Port} (h : RowSegWhole p) {t : ℕ} (ht : t < 1024)
    (hr : 64 * (segsOf p).size % 128 ≤ (HmRow.blockBit t).1) :
    HmRow.msgBit p p.chunks t = .const (tailBit (HmRow.tailOf p) (64 * (segsOf p).size % 128) t) := by
  sorry

end FlockSoundness.Discharge.ScopeW
