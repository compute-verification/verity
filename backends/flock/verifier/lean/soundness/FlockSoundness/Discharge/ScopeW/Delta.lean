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

/-- Compression `k`'s `cv` and `msg` columns as `HmRow.delta` reads them at a segmented row. -/
theorem cvmsg_eq (c : Circuit) (g p k : ℕ) :
    (if k < ((HmRow.rowSlots c.ports).2[p]!).size then
      (c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
          (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2]!.1,
        c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1 +
          (portCols (c.nets.getD c.shaNet default).2.inGroups)[2 * ((HmRow.rowSlots c.ports).2[p]!)[k]!.2 + 1]!.1)
    else (c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[0]!.1,
      c.slotCol c.hmNet g p + (portCols (c.nets.getD c.hmNet default).2.inGroups)[1]!.1)) =
      (cvAt c g p k, msgAt c g p k) := by
  unfold cvAt msgAt shaBase shaIn shaSub hmIn
  split <;> simp only [Hidden.getElem!_eq_getD']

/-- **Δ without a typed template, pair by pair, for rows of any shape**: the slot constants' pairs, each row's copies,
the units' leaf and cut bits and the wires' bits. -/
theorem delta_flat_eqW (c : Circuit) (pin : ℕ) (ht : c.typed = none) :
    (HmRow.delta c pin).1.toList = flatPairsW c pin := by
  unfold HmRow.delta
  simp only
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsW c pin)
    (P := fun d => d.toList = constPairs c pin) ?_ ?_
  · rw [← Array.forIn_toList]
    refine (forIn_id_append (slotPairs c pin) _ (fun r _ d => ?_) #[]).trans ?_
    rotate_left
    · simp [constPairs]
    · split
      · rename_i i hk
        refine ⟨_, rfl, ?_⟩
        simp only [slotPairs, hk]
        rw [Std.Legacy.Range.forIn_eq_forIn_range', forIn_id_append (fun s =>
          if (r.slot s + (c.nets.getD i default).2.constPos != pin) = true then
            [(r.slot s + (c.nets.getD i default).2.constPos, r.slot s + (c.nets.getD i default).2.constPos),
              (r.slot s + (c.nets.getD i default).2.constPos, pin)] else []) _ (fun s _ d => ?_) d]
        · simp [List.range_eq_range']
        · split
          · exact ⟨_, rfl, by simp⟩
          · exact ⟨_, rfl, by simp⟩
      · rename_i hk
        refine ⟨_, rfl, ?_⟩
        unfold slotPairs
        split
        · rename_i i hi; exact absurd hi (hk i)
        · simp
  intro d hd
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsW c pin)
    (P := fun d' => d'.toList = d.toList ++ (rowCopiesW c pin).flatMap pair2) ?_ ?_
  · rw [rowCopiesW, List.flatMap_assoc]
    refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d
    rw [List.flatMap_assoc]
    refine forIn_range_append _ _ (fun p hp d => ?_) d
    rcases hsg : c.ports[p]!.segs with _ | segs
    · simp only [Option.isSome_none, Bool.false_eq_true, ↓reduceIte, ← Hidden.rowPairsOf_eq]
      refine ⟨_, rfl, ?_⟩
      rw [rowPairsOf, ← List.append_assoc, ← List.append_assoc]
      refine Eq.trans (forIn_range'_append (fun k => (List.range 1024).flatMap fun t => bitPairs c pin
          (msgColOf c g p k + t) (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t)) _ _
          (fun k _ _ d => ⟨_, rfl, ?_⟩) _) ?_
      · refine forIn_range_append _ _ (fun t _ d => ?_) d
        split
        · rename_i v hv
          have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = some v := hv
          refine ⟨_, rfl, ?_⟩
          rw [hv']
          simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
          rfl
        · rename_i hv
          have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) k t = none :=
            Option.eq_none_iff_forall_ne_some.mpr hv
          refine ⟨_, rfl, ?_⟩
          rw [hv']
          simp only [bitPairs, List.append_nil]
      refine (List.append_left_inj _).mpr ?_
      refine Eq.trans (forIn_range_append (fun t => bitPairs c pin (padCol c g p + t)
          (HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t)) 1024
          (fun t _ d => ?_) _) ?_
      · split
        · rename_i v hv
          have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
            some v := hv
          refine ⟨_, rfl, ?_⟩
          rw [hv']
          simp only [bitPairs, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
          rfl
        · rename_i hv
          have hv' : HmRow.constBit c.ports[p]! (HmRow.tailOf c.ports[p]!) ((HmRow.rowSlots c.ports).2[p]!).size t =
            none := Option.eq_none_iff_forall_ne_some.mpr hv
          refine ⟨_, rfl, ?_⟩
          rw [hv']
          simp only [bitPairs, List.append_nil]
      refine (List.append_left_inj _).mpr ?_
      refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
      simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
      rfl
    · have hms : ∀ j, startMid c.ports[p]! j =
          if j < segs.size then HmRow.segMidstate c.ports[p]! j else HmRow.midstateOf c.ports[p]! := fun j => by
        simp only [startMid, segsOf, hsg, Option.getD_some]
      have hout : ∀ kk col, c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[kk]!.1 +
          (portCols (c.nets.getD c.shaNet default).2.outGroups)[((HmRow.rowSlots c.ports).2[p]!)[kk]!.2]!.1 + col =
          outAt c g p kk + col := fun kk col => by
        simp only [outAt, shaBase, shaOut, shaSub, Hidden.getElem!_eq_getD']
      simp only [Option.isSome_some, ↓reduceIte]
      refine ⟨_, rfl, ?_⟩
      rw [rowCopiesSeg, List.flatMap_assoc]
      refine forIn_range_append _ _ (fun k _ d => ?_) d
      rw [List.flatMap_append, cvmsg_eq]
      simp only
      split
      · rename_i j hj
        refine ⟨_, rfl, ?_⟩
        rw [hj, ← List.append_assoc]
        refine Eq.trans (forIn_range_append (fun t => (msgCopy c pin (msgAt c g p k + t) (outAt c g p)
            (HmRow.msgBit c.ports[p]! k t)).flatMap pair2) 1024 (fun t _ d => ?_) _) ?_
        · split
          · rename_i v hv
            refine ⟨_, rfl, ?_⟩
            rw [hv]
            simp only [msgCopy, pair2, Array.toList_push, List.append_assoc, List.flatMap_cons, List.flatMap_nil,
              List.append_nil, List.cons_append, List.nil_append]
            rfl
          · rename_i kk col hv
            refine ⟨_, rfl, ?_⟩
            rw [hv]
            simp only [msgCopy, pair2, Array.toList_push, List.append_assoc, List.flatMap_cons, List.flatMap_nil,
              List.append_nil, List.cons_append, List.nil_append, hout]
          · rename_i i hv
            refine ⟨_, rfl, ?_⟩
            rw [hv]
            simp only [msgCopy, List.flatMap_nil, List.append_nil]
        simp only [List.flatMap_assoc]
        refine (List.append_left_inj _).mpr ?_
        simp only [Option.toList_some, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.flatMap_map]
        refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
        simp only [pair2, hms, constSrc, Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
        rfl
      · rename_i hj
        have hj' : HmRow.chainStart c.ports[p]! k = none := Option.eq_none_iff_forall_ne_some.mpr hj
        refine ⟨_, rfl, ?_⟩
        rw [hj', Option.toList_none, List.flatMap_nil, List.flatMap_nil, List.nil_append, List.flatMap_assoc]
        refine forIn_range_append _ _ (fun t _ d => ?_) d
        split
        · rename_i v hv
          refine ⟨_, rfl, ?_⟩
          rw [hv]
          simp only [msgCopy, pair2, Array.toList_push, List.append_assoc, List.flatMap_cons, List.flatMap_nil,
            List.append_nil, List.cons_append, List.nil_append]
          rfl
        · rename_i kk col hv
          refine ⟨_, rfl, ?_⟩
          rw [hv]
          simp only [msgCopy, pair2, Array.toList_push, List.append_assoc, List.flatMap_cons, List.flatMap_nil,
            List.append_nil, List.cons_append, List.nil_append, hout]
        · rename_i i hv
          refine ⟨_, rfl, ?_⟩
          rw [hv]
          simp only [msgCopy, List.flatMap_nil, List.append_nil]
  intro d0 hd0
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsW c pin)
    (P := fun d' => d'.toList = d0.toList ++ Flat.leafPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun k _ d => ⟨_, rfl, ?_⟩) d0
    refine forIn_range_append _ _ (fun j _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d1 hd1
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsW c pin)
    (P := fun d' => d'.toList = d1.toList ++ Flat.cutPairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d1
    refine forIn_array_append _ _ (fun e _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d2 hd2
  refine FlockSoundness.run_bind_fst (Q := fun r => r.toList = flatPairsW c pin)
    (P := fun d' => d'.toList = d2.toList ++ Flat.wirePairs c) ?_ ?_
  · refine forIn_range_append _ _ (fun g _ d => ⟨_, rfl, ?_⟩) d2
    refine forIn_array_append _ _ (fun w _ d => ⟨_, rfl, ?_⟩) d
    refine forIn_range_append _ _ (fun t _ d => ⟨_, rfl, ?_⟩) d
    simp only [Array.toList_push, List.append_assoc, List.cons_append, List.nil_append]
    rfl
  intro d3 hd3
  simp only [ht]
  show d3.toList = _
  rw [hd3, hd2, hd1, hd0, hd]
  rfl

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
  unfold Typed.msgCol msgAt shaBase shaIn shaSub hmIn
  simp only [Hidden.getElem!_eq_getD']
  split <;> omega

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
  obtain ⟨hs, hb, -, hw⟩ := h
  have hj : (HmRow.blockBit t).1 = 8 * (t / 64) + 7 - t % 64 / 8 := rfl
  have hb8 : (HmRow.blockBit t).2 = t % 8 := rfl
  simp only [HmRow.msgBit, HmRow.constBit, hs,
    show 8 * (128 * k + (HmRow.blockBit t).1) + (HmRow.blockBit t).2 < p.bits by rw [hj, hb8]; omega, ↓reduceIte]

/-- **A whole row's `hm96` slot holds its padding block** (`HmRow.padWords`). -/
theorem msgBit_whole_pad {p : Port} (h : RowWhole p) {t : ℕ} (ht : t < 1024) :
    HmRow.msgBit p p.chunks t = .const (wordsBit (HmRow.padWords p.words) t) := by
  obtain ⟨hs, hb, -, hw⟩ := h
  simp only [HmRow.msgBit, hs, Hidden.constBit_row hs (Hidden.bytes_of_bits hb) hb hw ht]

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
  simp only [HmRow.blockBit]
  omega

/-- Byte `o`'s bit `b`'s message column is the block column of `(o, b)`. -/
theorem blockBit_of_col {o b : ℕ} (ho : o < 128) (hb : b < 8) :
    HmRow.blockBit (64 * (o / 8) + 56 - 8 * (o % 8) + b) = (o, b) := by
  simp only [HmRow.blockBit, Prod.mk.injEq]
  omega

theorem size_foldl_push (l : List ℕ) (b : ByteArray) (f : ℕ → UInt8) :
    (l.foldl (fun b i => b.push (f i)) b).size = b.size + l.length := by
  induction l generalizing b with
  | nil => rfl
  | cons a l ih => rw [List.foldl_cons, ih, ByteArray.size_push, List.length_cons]; omega

/-- SHA-512's padding after `total` bytes: `0x80`, zeros to 112 mod 128, 16 length bytes. -/
theorem padTail_size (total : ℕ) : (HmRow.padTail total).size = (239 - total % 128) % 128 + 17 := by
  simp only [HmRow.padTail, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind,
    Id.run_pure, Std.Legacy.Range.size, Nat.reduceSub, Nat.reduceAdd, Nat.sub_zero, Nat.div_one]
  rw [size_foldl_push, ByteArray.size_append, List.length_range']
  show 1 + (Array.replicate _ 0).size + 16 = _
  rw [Array.size_replicate]
  omega

/-- Block column `t`'s bit of a block's words is bit `(blockBit t).2` of its byte `(blockBit t).1`. -/
theorem wordsBit_wordsBE (B : ByteArray) {t : ℕ} (ht : t < 1024) :
    wordsBit (HmRow.wordsBE B 16) t = ((B.get! (HmRow.blockBit t).1).toNat >>> (HmRow.blockBit t).2 % 2 == 1) := by
  have hj : (HmRow.blockBit t).1 = 8 * (t / 64) + 7 - t % 64 / 8 := rfl
  have hb8 : (HmRow.blockBit t).2 = t % 8 := rfl
  rw [wordsBit_eq _ (n := 16) (by omega), wbit,
    List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range (by omega), Option.map_some, Option.getD_some,
    wordsBE_getElem! (by omega), getLsbD_wordL _ _ _ (by omega), get!_eq_getD,
    show 0 + 8 * (t / 64) + (7 - t % 64 / 8) = 8 * (t / 64) + 7 - t % 64 / 8 by omega,
    show t % 64 % 8 = t % 8 by omega, hj, hb8]
  generalize (B.data.toList.getD (8 * (t / 64) + 7 - t % 64 / 8) 0).toNat = x
  rw [Nat.testBit, Nat.one_and_eq_mod_two]
  rcases Nat.mod_two_eq_zero_or_one (x >>> (t % 8)) with h | h <;> simp [h]

/-- **A block of `pre`'s bytes then `tl`'s**: at a column of `tl`'s, the block's word bit is `tl`'s bit. -/
theorem tailBit_words {pre tl : ByteArray} {t : ℕ} (hsz : pre.size + tl.size = 128) (ht : t < 1024)
    (hr : pre.size ≤ (HmRow.blockBit t).1) :
    tailBit tl pre.size t = wordsBit (HmRow.wordsBE (pre ++ tl) 16) t := by
  clear hsz
  rw [wordsBit_wordsBE _ ht, tailBit, get!_eq_getD, get!_eq_getD, ByteArray.data_append, Array.toList_append,
    List.getD_append_right _ _ _ _ (by simpa using hr)]
  simp

theorem tailOf_short {p : Port} (h : RowV2Short p) : HmRow.tailOf p = HmRow.padTail (128 + 2 * p.words) := by
  obtain ⟨hv, hs, hb, -⟩ := h
  simp only [HmRow.tailOf, hs, Port.bytes, hv, ↓reduceIte, hb]
  congr 2
  omega

/-- **A short row's bytes and padding fill its `hm96` slot's block.** -/
theorem fill_short {p : Port} (h : RowV2Short p) : 2 * p.words + (HmRow.tailOf p).size = 128 := by
  rw [tailOf_short h, padTail_size]
  have := h.2.2.2.2.2
  omega

theorem tailOf_seg {p : Port} (h : RowSegWhole p) :
    HmRow.tailOf p = HmRow.padTail (128 + 64 * (segsOf p).size) := by
  simp only [HmRow.tailOf, segsOf, h.2.1, Option.getD_some]

/-- **A segmented row's outer message's last digest bytes and padding fill its `hm96` slot's block**: `64 s mod 128`
digest bytes (`64` for `s` odd, none for `s` even), then the padding. -/
theorem fill_seg {p : Port} (h : RowSegWhole p) : 64 * (segsOf p).size % 128 + (HmRow.tailOf p).size = 128 := by
  rw [tailOf_seg h, padTail_size]
  omega

/-! ## Short rows -/

theorem blockBit_lt {t : ℕ} (ht : t < 1024) : (HmRow.blockBit t).1 < 128 ∧ (HmRow.blockBit t).2 < 8 := by
  simp only [HmRow.blockBit]
  omega

/-- **A short row's `hm96` slot holds its bits**, at the block's first `2 · words` bytes. -/
theorem msgBit_short_row {p : Port} (h : RowV2Short p) {t : ℕ} (ht : t < 1024)
    (hr : (HmRow.blockBit t).1 < 2 * p.words) :
    HmRow.msgBit p p.chunks t = .row (8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2) := by
  obtain ⟨-, hs, hb, -, hc, -⟩ := h
  have hb8 := (blockBit_lt ht).2
  simp only [HmRow.msgBit, HmRow.constBit, hs, hc, Nat.mul_zero, Nat.zero_add,
    show 8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2 < p.bits by omega, ↓reduceIte]

/-- **A short row's `hm96` slot holds its padding** past its bytes. -/
theorem msgBit_short_tail {p : Port} (h : RowV2Short p) {t : ℕ} (ht : t < 1024)
    (hr : 2 * p.words ≤ (HmRow.blockBit t).1) :
    HmRow.msgBit p p.chunks t = .const (tailBit (HmRow.tailOf p) (2 * p.words) t) := by
  obtain ⟨hv, hs, hb, -, hc, -⟩ := h
  have hby : p.bytes = 2 * p.words := by simp only [Port.bytes, hv, ↓reduceIte, hb]; omega
  have hb8 := (blockBit_lt ht).2
  simp only [HmRow.msgBit, HmRow.constBit, hs, hc, Nat.mul_zero, Nat.zero_add, hby,
    show ¬ 8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2 < p.bits by omega, ↓reduceIte, decide_eq_true hr,
    Bool.true_and, tailBit]

/-! ## Segmented rows -/

theorem segFirst_zero (p : Port) : segFirst p 0 = 0 := by
  simp [segFirst]

/-- A segmented row's byte `byte`: block `byte % 1024 / 128` of segment `byte / 1024`. -/
theorem byteAt_seg {p : Port} {segs : Array (ℕ × ℕ)} (hs : p.segs = some segs) (byte : ℕ) :
    HmRow.byteAt p byte = (segFirst p (byte / 1024) + byte % 1024 / 128, byte % 128) := by
  simp [HmRow.byteAt, segFirst, segsOf, hs]

theorem segFirst_succ (p : Port) {j : ℕ} (hj : j < (segsOf p).size) :
    segFirst p (j + 1) = segFirst p j + ((segsOf p)[j]!).2 := by
  unfold segFirst
  rw [← Array.foldl_toList, ← Array.foldl_toList, Array.toList_extract, Array.toList_extract]
  simp only [List.extract, Nat.sub_zero, List.drop_zero]
  rw [List.take_add_one, List.foldl_append, List.getElem?_eq_getElem (by simpa using hj)]
  simp [getElem!_pos (segsOf p) j hj]

theorem segFirst_mono (p : Port) {i j : ℕ} (hij : i ≤ j) (hj : j ≤ (segsOf p).size) : segFirst p i ≤ segFirst p j := by
  induction j, hij using Nat.le_induction with
  | base => exact le_refl _
  | succ j _ ih =>
    rw [segFirst_succ p (by omega)]
    exact le_trans (ih (by omega)) (Nat.le_add_right _ _)

theorem segFirst_lt (p : Port) {i j : ℕ} (hij : i < j) (hj : j ≤ (segsOf p).size) (hpos : 0 < ((segsOf p)[i]!).2) :
    segFirst p i < segFirst p j := by
  have := segFirst_mono p (Nat.succ_le_of_lt hij) hj
  rw [segFirst_succ p (by omega)] at this
  omega

/-- A loop over `[0:n]` with early exits, then `K`: an invariant `I` at each index, kept by every step that yields, and
`K`'s result `r` at every exit and at the end. -/
theorem forIn_range_bind_eq {σ β : Type} {n : ℕ} {init : σ} {f : ℕ → σ → Id (ForInStep σ)} {K : σ → Id β} {r : β}
    (I : ℕ → σ → Prop) (h0 : I 0 init)
    (hstep : ∀ j s, j < n → I j s →
      (∃ s', f j s = .yield s' ∧ I (j + 1) s') ∨ (∃ s', f j s = .done s' ∧ (K s').run = r))
    (hend : ∀ s, I n s → (K s).run = r) :
    (forIn [:n] init f >>= K).run = r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices H : ∀ a m (init : σ), a + m = n → I a init → (forIn (List.range' a m) init f >>= K).run = r from
    H 0 n init (Nat.zero_add n) h0
  intro a m
  induction m generalizing a with
  | zero => intro init ha hI; simp only [List.range'_zero, List.forIn_nil]; exact hend init (by simpa using ha ▸ hI)
  | succ m ih =>
    intro init ha hI
    rw [List.range'_succ, List.forIn_cons]
    rcases hstep a init (by omega) hI with ⟨s', hs, hI'⟩ | ⟨s', hs, hQ⟩
    · rw [hs]; exact ih (a + 1) s' (by omega) hI'
    · rw [hs]; exact hQ

theorem segs_eq {p : Port} (h : RowSegWhole p) : p.segs = some (segsOf p) := by
  simp only [segsOf, h.2.1, Option.getD_some]

theorem segsOf_size {p : Port} (h : RowSegWhole p) : (segsOf p).size = (p.bits + 8191) / 8192 := by
  simp [segsOf, h.2.1, HmRow.segShape, HmRow.ROW_SEG_BITS]

/-- **A whole-block segmented row's segments**: segment `j`'s bits a positive multiple of 1024, its compressions its
data blocks and a padding block. -/
theorem segs_whole {p : Port} (h : RowSegWhole p) {j : ℕ} (hj : j < (segsOf p).size) :
    0 < ((segsOf p)[j]!).1 ∧ ((segsOf p)[j]!).1 % 1024 = 0 ∧ ((segsOf p)[j]!).2 = ((segsOf p)[j]!).1 / 1024 + 1 := by
  have hj' := hj
  rw [segsOf_size h] at hj'
  have hg : (segsOf p)[j]! =
      (min (p.bits - 8192 * j) 8192, ((min (p.bits - 8192 * j) 8192 + 7) / 8 + 17 + 127) / 128) := by
    rw [getElem!_pos (segsOf p) j hj]
    simp [segsOf, h.2.1, HmRow.segShape, HmRow.ROW_SEG_BITS]
  rw [hg]
  obtain ⟨-, -, hb, -, h64, -⟩ := h
  simp only
  omega

/-- **The outer message ends in the `hm96` slot**: its compressions run from `segFirst p s` to the row's `chunks`. -/
theorem seg_chunks {p : Port} (h : RowSegWhole p) :
    p.chunks = segFirst p (segsOf p).size + 64 * (segsOf p).size / 128 := by
  have hc := h.2.2.2.2.2
  have hsz := segsOf_size h
  have hss : segsOf p = (HmRow.segShape p.bits).1 := by simp [segsOf, h.2.1]
  have h2 : (HmRow.segShape p.bits).2 = (64 * ((p.bits + 8191) / 8192) + 17 + 127) / 128 := by
    simp [HmRow.segShape, HmRow.ROW_SEG_BITS]
  have hf : segFirst p (segsOf p).size = (HmRow.segShape p.bits).1.foldl (· + ·.2) 0 := by
    unfold segFirst
    rw [Array.extract_size, hss]
  rw [hf, hsz]
  omega

/-- A segmented row's chain starts at each segment's first compression and the outer message's, when every segment
has a compression. -/
theorem chainStart_seg' {p : Port} {j : ℕ} (hs : p.segs = some (segsOf p)) (hj : j ≤ (segsOf p).size)
    (hpos : ∀ i < (segsOf p).size, 0 < ((segsOf p)[i]!).2) :
    HmRow.chainStart p (segFirst p j) = some j := by
  unfold HmRow.chainStart
  rw [hs]
  simp only
  refine forIn_range_bind_eq (fun i s => i ≤ j ∧ s = (none, segFirst p i)) ⟨Nat.zero_le _, by rw [segFirst_zero]⟩ ?_ ?_
  · rintro i s hi ⟨hij, rfl⟩
    rcases Nat.lt_or_eq_of_le hij with hij | rfl
    · have hlt := segFirst_lt p hij hj (hpos i hi)
      have hle := segFirst_mono p (Nat.succ_le_of_lt hij) hj
      refine Or.inl ⟨_, ?_, Nat.succ_le_of_lt hij, rfl⟩
      simp only [beq_iff_eq, Nat.ne_of_gt hlt, ↓reduceIte, ← segFirst_succ p hi, Nat.not_lt.2 hle]
      rfl
    · exact Or.inr ⟨_, by simp only [beq_self_eq_true, ↓reduceIte]; rfl, rfl⟩
  · rintro s ⟨hij, rfl⟩
    have : j = (segsOf p).size := Nat.le_antisymm hj hij
    subst this
    simp

/-- **Each segment's chain, and the outer message's, starts at its first compression.** -/
theorem chainStart_seg {p : Port} (h : RowSegWhole p) {j : ℕ} (hj : j ≤ (segsOf p).size) :
    HmRow.chainStart p (segFirst p j) = some j :=
  chainStart_seg' (segs_eq h) hj fun i hi => by rw [(segs_whole h hi).2.2]; omega

/-- **No other compression starts a chain**, for a row of any shape. -/
theorem chainStart_none {p : Port} {k : ℕ} (hk : ∀ j ≤ (segsOf p).size, segFirst p j ≠ k) :
    HmRow.chainStart p k = none := by
  unfold HmRow.chainStart
  rcases hs : p.segs with _ | segs
  · have := hk 0 (by simp [segsOf, hs])
    rw [segFirst_zero] at this
    simp only [beq_iff_eq, Ne.symm this, ↓reduceIte]
  · have hss : segsOf p = segs := by simp [segsOf, hs]
    simp only
    refine forIn_range_bind_eq (fun j s => s = (none, segFirst p j)) (by rw [segFirst_zero]) ?_ ?_
    · rintro j s hj rfl
      have hne := hk j (by rw [hss]; omega)
      simp only [beq_iff_eq, Ne.symm hne, ↓reduceIte]
      have h1 : segFirst p (j + 1) = segFirst p j + segs[j]!.2 := by
        rw [segFirst_succ p (by rw [hss]; exact hj), hss]
      by_cases hlt : k < segFirst p j + segs[j]!.2
      · exact Or.inr ⟨_, by simp only [hlt, ↓reduceIte]; rfl, rfl⟩
      · exact Or.inl ⟨_, by simp only [hlt, ↓reduceIte]; rfl, by rw [h1]⟩
    · rintro s rfl
      have hne := hk segs.size (by rw [hss])
      simp [Ne.symm hne]

/-- Compression `k` of segment `j` of a segmented row: `HmRow.msgBit`'s segment branch at `j`. -/
theorem msgBit_seg_in {p : Port} (hs : p.segs = some (segsOf p)) {j k t : ℕ} (hj : j < (segsOf p).size)
    (h1 : segFirst p j ≤ k) (h2 : k < segFirst p (j + 1)) :
    HmRow.msgBit p k t =
      if 8 * (128 * (k - segFirst p j) + (HmRow.blockBit t).1) + (HmRow.blockBit t).2 < ((segsOf p)[j]!).1 then
        .row (HmRow.ROW_SEG_BITS * j + 8 * (128 * (k - segFirst p j) + (HmRow.blockBit t).1) + (HmRow.blockBit t).2)
      else .const (decide ((((segsOf p)[j]!).1 + 7) / 8 ≤ 128 * (k - segFirst p j) + (HmRow.blockBit t).1) &&
        (((HmRow.segTail p j).get! (128 * (k - segFirst p j) + (HmRow.blockBit t).1 -
          (((segsOf p)[j]!).1 + 7) / 8)).toNat >>> (HmRow.blockBit t).2) % 2 == 1) := by
  unfold HmRow.msgBit
  rw [hs]
  simp only
  refine forIn_range_bind_eq (fun i s => i ≤ j ∧ s = (none, segFirst p i)) ⟨Nat.zero_le _, by rw [segFirst_zero]⟩ ?_ ?_
  · rintro i s hi ⟨hij, rfl⟩
    rcases Nat.lt_or_eq_of_le hij with hij | rfl
    · have hle := segFirst_mono p (Nat.succ_le_of_lt hij) hj.le
      refine Or.inl ⟨_, ?_, Nat.succ_le_of_lt hij, rfl⟩
      rw [segFirst_succ p hi] at hle
      simp only [Nat.not_lt.2 (le_trans hle h1), ↓reduceIte]
      rw [← segFirst_succ p hi]
      rfl
    · rw [segFirst_succ p hi] at h2
      by_cases hc : 8 * (128 * (k - segFirst p i) + (HmRow.blockBit t).1) + (HmRow.blockBit t).2 < ((segsOf p)[i]!).1
      · simp only [h2, hc, ↓reduceIte]
        exact Or.inr ⟨_, rfl, rfl⟩
      · simp only [h2, hc, ↓reduceIte]
        exact Or.inr ⟨_, rfl, rfl⟩
  · rintro s ⟨hij, rfl⟩
    omega

/-- Compression `k` of a segmented row's outer message: `HmRow.msgBit` past its segments. -/
theorem msgBit_seg_out {p : Port} (hs : p.segs = some (segsOf p)) {k t : ℕ}
    (hk : segFirst p (segsOf p).size ≤ k) :
    HmRow.msgBit p k t =
      if 128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1 < 64 * (segsOf p).size then
        .digest (segFirst p ((128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1) / 64 + 1) - 1)
          (64 * ((128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1) % 64 / 8) + 56 -
            8 * ((128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1) % 64 % 8) + (HmRow.blockBit t).2)
      else .const ((((HmRow.tailOf p).get! (128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1 -
        64 * (segsOf p).size)).toNat >>> (HmRow.blockBit t).2) % 2 == 1) := by
  unfold HmRow.msgBit
  rw [hs]
  simp only
  refine forIn_range_bind_eq (fun i s => s = (none, segFirst p i)) (by rw [segFirst_zero]) ?_ ?_
  · rintro i s hi rfl
    have hle := segFirst_mono p (Nat.succ_le_of_lt hi) le_rfl
    refine Or.inl ⟨_, ?_, rfl⟩
    rw [segFirst_succ p hi] at hle
    simp only [Nat.not_lt.2 (le_trans hle hk), ↓reduceIte]
    rw [← segFirst_succ p hi]
    rfl
  · rintro s rfl
    by_cases hc : 128 * (k - segFirst p (segsOf p).size) + (HmRow.blockBit t).1 < 64 * (segsOf p).size
    · simp only [hc, ↓reduceIte]; rfl
    · simp only [hc, ↓reduceIte]; rfl

/-- **A segment's data blocks hold its bits**: block `i` of segment `j`, bit `t`, is the row's bit
`8192 j + 1024 i + 8 · byte + bit`. -/
theorem msgBit_seg_row {p : Port} (h : RowSegWhole p) {j i t : ℕ} (hj : j < (segsOf p).size)
    (hi : i < ((segsOf p)[j]!).1 / 1024) (ht : t < 1024) :
    HmRow.msgBit p (segFirst p j + i) t =
      .row (HmRow.ROW_SEG_BITS * j + 1024 * i + 8 * (HmRow.blockBit t).1 + (HmRow.blockBit t).2) := by
  obtain ⟨-, hm, hn⟩ := segs_whole h hj
  have hb := blockBit_lt ht
  rw [msgBit_seg_in (segs_eq h) hj (Nat.le_add_right _ _) (by rw [segFirst_succ p hj, hn]; omega),
    Nat.add_sub_cancel_left, ite_eq_left_of_eq_true _ _ (eq_true (by omega))]
  congr 1
  omega

/-- **A segment's last block is a padding block of its own.** -/
theorem msgBit_seg_pad {p : Port} (h : RowSegWhole p) {j t : ℕ} (hj : j < (segsOf p).size) (ht : t < 1024) :
    HmRow.msgBit p (segFirst p j + ((segsOf p)[j]!).1 / 1024) t =
      .const (wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf p)[j]!).1 / 8)) 16) t) := by
  obtain ⟨-, hm, hn⟩ := segs_whole h hj
  have hb := blockBit_lt ht
  have hst : HmRow.segTail p j = HmRow.padBlock (128 + ((segsOf p)[j]!).1 / 8) := by
    unfold HmRow.segTail
    rw [← Hidden.getElem!_eq_getD', show (((segsOf p)[j]!).1 + 7) / 8 = ((segsOf p)[j]!).1 / 8 by omega,
      padTail_eq (by omega)]
  rw [msgBit_seg_in (segs_eq h) hj (Nat.le_add_right _ _) (by rw [segFirst_succ p hj, hn]; omega),
    Nat.add_sub_cancel_left, ite_eq_right_of_eq_false _ _ (eq_false (by omega)), decide_eq_true (by omega),
    Bool.true_and, wordsBit_wordsBE _ ht, hst,
    show 128 * (((segsOf p)[j]!).1 / 1024) + (HmRow.blockBit t).1 - (((segsOf p)[j]!).1 + 7) / 8 =
      (HmRow.blockBit t).1 by omega]

/-- **The outer message's byte `e < 64 s` is a digest byte**: bit `b` of byte `e % 64` of segment `e / 64`'s digest,
column `64 (q / 8) + 56 − 8 (q % 8) + b` (`q = e % 64`) of that segment's last compression's output. -/
theorem msgBit_seg_digest {p : Port} (h : RowSegWhole p) {e b : ℕ} (he : e < 64 * (segsOf p).size) (hb : b < 8) :
    HmRow.msgBit p (segFirst p (segsOf p).size + e / 128) (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b) =
      .digest (segFirst p (e / 64 + 1) - 1) (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b) := by
  rw [msgBit_seg_out (segs_eq h) (Nat.le_add_right _ _), blockBit_of_col (Nat.mod_lt _ (by omega)) hb,
    Nat.add_sub_cancel_left, show 128 * (e / 128) + e % 128 = e by omega, ite_eq_left_of_eq_true _ _ (eq_true he)]

/-- **The `hm96` slot's block past the digests is the outer message's padding.** -/
theorem msgBit_seg_tail {p : Port} (h : RowSegWhole p) {t : ℕ} (ht : t < 1024)
    (hr : 64 * (segsOf p).size % 128 ≤ (HmRow.blockBit t).1) :
    HmRow.msgBit p p.chunks t = .const (tailBit (HmRow.tailOf p) (64 * (segsOf p).size % 128) t) := by
  have hc := seg_chunks h
  rw [msgBit_seg_out (segs_eq h) (by omega), hc, Nat.add_sub_cancel_left,
    ite_eq_right_of_eq_false _ _ (eq_false (by omega)), tailBit,
    show 128 * (64 * (segsOf p).size / 128) + (HmRow.blockBit t).1 - 64 * (segsOf p).size =
      (HmRow.blockBit t).1 - 64 * (segsOf p).size % 128 by omega]

end FlockSoundness.Discharge.ScopeW
