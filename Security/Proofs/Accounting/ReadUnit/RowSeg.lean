/-
Part A, `sha512/row-seg/v1`'s generators against `Hash.RowSeg`: the segment generator is SHA-512 continued from its
chaining value over the segment's bytes (`sound_segment`), so SHA-512 of its prefix and the segment from the prefix's
chaining value (`sound_segment_mid`); the outer generator is SHA-512 of `Q(l)` and the digests (`sound_outer`); the row
generator is `Hash.RowSeg.digest` of the row's bits wherever its callees compute their constants (`ConstSpec`) and SHA-512
of their messages (`SegSpec`, `OuterSpec`) (`rowSeg_spec`), so over `constBit`, `segment` and `outer` it meets
`Gates.RowSegSpec` (`rowSegNet_spec`).
-/
module

public import Flock.ReadNet.RowSeg
public import Proofs.Accounting.ReadUnit.Sha
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import all Init.Data.String.Defs

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

/-! ## A segment's bytes -/

theorem length_packBits (bs : List Bool) : (Hash.RowSeg.packBits bs).length = (bs.length + 7) / 8 := by
  simp [Hash.RowSeg.packBits]

theorem testBit_bitsVal : ∀ (bs : List Bool) (i : ℕ), (Hash.RowSeg.bitsVal bs).testBit i = bs.getD i false
  | [], i => by simp [Hash.RowSeg.bitsVal]
  | b :: bs, 0 => by
      simp only [Hash.RowSeg.bitsVal, Nat.testBit_zero, List.getD_cons_zero]
      cases b <;> simp [Nat.add_mod]
  | b :: bs, i + 1 => by
      rw [Nat.testBit_succ, List.getD_cons_succ, ← testBit_bitsVal bs i]
      congr 1
      simp only [Hash.RowSeg.bitsVal]
      cases b <;> simp
      omega

/-- Bit `q` of a row's bytes is the row's bit `q`, zero past its end. -/
theorem bitL_packBits {bs : List Bool} {q : ℕ} (hq : q < 8 * ((bs.length + 7) / 8)) :
    bitL (Hash.RowSeg.packBits bs) q = bs.getD q false := by
  have hj : q / 8 < (bs.length + 7) / 8 := by omega
  have hlt : Hash.RowSeg.bitsVal ((bs.drop (8 * (q / 8))).take 8) < 2 ^ 8 :=
    lt_of_lt_of_le (Hash.BindingProofs.bitsVal_lt _) (Nat.pow_le_pow_right (by norm_num) (by simp))
  rw [bitL, List.getD_eq_getElem _ _ (by rw [length_packBits]; exact hj)]
  simp only [Hash.RowSeg.packBits, List.getElem_map, List.getElem_range]
  rw [Nat.toUInt8_eq, UInt8.toNat_ofNat', Nat.mod_eq_of_lt hlt, testBit_bitsVal]
  simp only [List.getD_eq_getElem?_getD, List.getElem?_take, List.getElem?_drop, Nat.mod_lt q (by norm_num : 8 > 0),
    ite_true, Nat.div_add_mod]

theorem length_constBits (l : List UInt8) : (Flock.ReadNet.constBits l).length = 8 * l.length := by
  simp [Flock.ReadNet.constBits]

theorem getD_constBits {l : List UInt8} {q : ℕ} (hq : q < 8 * l.length) :
    (Flock.ReadNet.constBits l).getD q Lin.zero = Lin.const (bitL l q) := by
  rw [List.getD_eq_getElem _ _ (by rw [length_constBits]; exact hq)]
  simp only [Flock.ReadNet.constBits, List.getElem_map, List.getElem_range]
  rfl

/-- The segment generator's message bits are the segment's bytes and then `pad`, bit by bit. -/
theorem ev_segBits (z : Var → Bool) {m : ℕ} {bits : List Lin} (hbl : bits.length = m) (pad : List UInt8) {q : ℕ}
    (hq : q < 8 * ((m + 7) / 8) + 8 * pad.length) :
    ev z ((bits ++ List.replicate (8 * ((m + 7) / 8) - m) Lin.zero ++ Flock.ReadNet.constBits pad).getD q Lin.zero) =
      bitL (Hash.RowSeg.packBits (evs z bits) ++ pad) q := by
  have hn : (Hash.RowSeg.packBits (evs z bits)).length = (m + 7) / 8 := by rw [length_packBits, length_evs, hbl]
  have hpre : (bits ++ List.replicate (8 * ((m + 7) / 8) - m) Lin.zero).length = 8 * ((m + 7) / 8) := by
    rw [List.length_append, List.length_replicate, hbl]
    omega
  by_cases h1 : q < 8 * ((m + 7) / 8)
  · have hbL : bitL (Hash.RowSeg.packBits (evs z bits) ++ pad) q = bitL (Hash.RowSeg.packBits (evs z bits)) q := by
      rw [bitL, bitL, List.getD_append _ _ _ _ (by rw [hn]; omega)]
    rw [hbL, bitL_packBits (by rw [length_evs, hbl]; exact h1), List.getD_append _ _ _ _ (by rw [hpre]; exact h1)]
    by_cases h2 : q < m
    · rw [List.getD_append _ _ _ _ (by omega), List.getD_eq_getElem _ _ (by omega),
        List.getD_eq_getElem _ _ (by rw [length_evs]; omega)]
      simp [evs]
    · rw [List.getD_append_right _ _ _ _ (by omega), List.getD_eq_getElem _ _ (by simp; omega),
        List.getElem_replicate, List.getD_eq_default _ _ (by rw [length_evs]; omega)]
      simp
  · rw [List.getD_append_right _ _ _ _ (by rw [hpre]; omega), hpre, getD_constBits (by omega), ev_const, bitL, bitL,
      List.getD_append_right _ _ _ _ (by rw [hn]; omega), hn, show (q - 8 * ((m + 7) / 8)) / 8 = q / 8 - (m + 7) / 8 by
      omega, show (q - 8 * ((m + 7) / 8)) % 8 = q % 8 by omega]

/-! ## Message words -/

theorem length_msgWords (fs : List Lin) : (Flock.ReadNet.msgWords fs).length = fs.length / 64 := by
  simp [Flock.ReadNet.msgWords]

theorem words64_msgWords (fs : List Lin) : Words64 (Flock.ReadNet.msgWords fs) := by
  intro w hw
  simp only [Flock.ReadNet.msgWords, List.mem_map] at hw
  obtain ⟨i, _, rfl⟩ := hw
  simp

/-- Message words over forms that hold a byte string's bits hold its big-endian words. -/
theorem map_bv_msgWords (z : Var → Bool) {fs : List Lin} {L : List UInt8} (hl : fs.length = 8 * L.length)
    (h : ∀ q < fs.length, ev z (fs.getD q Lin.zero) = bitL L q) :
    (Flock.ReadNet.msgWords fs).map (bv z) = wordsBE L := by
  apply List.ext_getElem
  · rw [List.length_map, length_msgWords, length_wordsBE]
    omega
  · intro i h1 h2
    have hi : i < fs.length / 64 := by rw [List.length_map, length_msgWords] at h1; exact h1
    simp only [Flock.ReadNet.msgWords, wordsBE, List.getElem_map, List.getElem_range]
    refine bv_ext (by simp) fun p hp => ?_
    have hw : wordBit (64 * i + p) < fs.length := by unfold wordBit; omega
    simp only [FlockSoundness.Discharge.Hm.bit, List.getElem?_map, List.getElem?_range hp, Option.map_some,
      Option.getD_some]
    rw [h _ hw, bitL, getLsbD_wordL _ _ _ hp, show wordBit (64 * i + p) / 8 = 0 + 8 * i + (7 - p / 8) by
      unfold wordBit; omega, show wordBit (64 * i + p) % 8 = p % 8 by unfold wordBit; omega]

/-! ## The segment and the outer hash -/

/-- **The segment generator** (`Sha512RowSegSegment_v1{M}`): whenever its rows hold, its eight words are the chain from
its chaining value over the segment's bytes (its `m` bits, zero-filled to whole bytes) and the padding of a
`128 + ⌈m / 8⌉`-byte message. -/
theorem sound_segment (m : ℕ) (cv : List Word) (bits : List Lin) :
    Sound (Flock.ReadNet.segment m cv bits) fun z out => cv.length = 8 → Words64 cv → bits.length = m →
      out.length = 8 ∧ Words64 out ∧
        out.map (bv z) = chainW (blocks (128 + (m + 7) / 8) - 1) (cv.map (bv z))
          (wordsBE (Hash.RowSeg.packBits (evs z bits) ++ Flock.ReadNet.padTail (128 + (m + 7) / 8))) := by
  simp only [Flock.ReadNet.segment]
  refine Sound.mono (sound_shaW _ cv _) fun z out h hcl hcw hbl => ?_
  have hp := length_padTail (128 + (m + 7) / 8)
  have hb := blocks_mul (128 + (m + 7) / 8)
  have hfs : (bits ++ List.replicate (8 * ((m + 7) / 8) - m) Lin.zero ++
      Flock.ReadNet.constBits (Flock.ReadNet.padTail (128 + (m + 7) / 8))).length =
      1024 * (blocks (128 + (m + 7) / 8) - 1) := by
    simp only [List.length_append, List.length_replicate, length_constBits, hbl]
    omega
  have hmw : (Flock.ReadNet.msgWords (bits ++ List.replicate (8 * ((m + 7) / 8) - m) Lin.zero ++
      Flock.ReadNet.constBits (Flock.ReadNet.padTail (128 + (m + 7) / 8)))).map (bv z) =
      wordsBE (Hash.RowSeg.packBits (evs z bits) ++ Flock.ReadNet.padTail (128 + (m + 7) / 8)) :=
    map_bv_msgWords z (by rw [hfs, List.length_append, length_packBits, length_evs, hbl]; omega)
      fun q hq => ev_segBits z hbl _ (by rw [hfs] at hq; omega)
  obtain ⟨o1, o2, o3⟩ := h hcl hcw (by rw [length_msgWords, hfs]; omega) (words64_msgWords _)
  refine ⟨o1, o2, ?_⟩
  rw [o3, length_msgWords, hfs, show 1024 * (blocks (128 + (m + 7) / 8) - 1) / 64 / 16 =
    blocks (128 + (m + 7) / 8) - 1 by omega, hmw]

/-- **From a block's chaining value, the segment generator hashes the block and the segment**: whenever its rows hold
and `cv` holds the 128-byte `pre`'s chaining value, its words are SHA-512 of `pre` and the segment's bytes
(`sha512/row-seg/v1`'s `d_j` at `pre = S(j, s, m)`). -/
theorem sound_segment_mid {G : Hash.Msg → Verity.Digest} (hG : IsSha512 G) (m : ℕ) {pre : ByteArray}
    (hpre : pre.size = 128) (cv : List Word) (bits : List Lin) :
    Sound (Flock.ReadNet.segment m cv bits) fun z out => cv.length = 8 → Words64 cv → cv.map (bv z) = chainV pre 1 →
      bits.length = m → out.length = 8 ∧ Words64 out ∧
        (G (pre.data.toList ++ Hash.RowSeg.packBits (evs z bits))).bytes = bytesBE (out.map (bv z)) :=
  Sound.mono (sound_segment m cv bits) fun z out h hcl hcw hcv hbl => by
    obtain ⟨o1, o2, o3⟩ := h hcl hcw hbl
    refine ⟨o1, o2, ?_⟩
    rw [hash_mid hG hpre, o3, hcv, length_packBits, length_evs, hbl]

theorem be32_eq (x : UInt32) :
    Flock.be32 x = ⟨#[(x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, x.toUInt8]⟩ := by
  simp only [Flock.be32, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.size]
  rfl

theorem be32_1024 : (Flock.be32 1024).data.toList = Hash.be 4 1024 := by
  rw [be32_eq]
  refine List.ext_getElem (by simp [Hash.be]) fun i h1 h2 => UInt8.toNat_inj.1 ?_
  have hi : i < 4 := by simpa [Hash.be] using h2
  interval_cases i <;> simp [Hash.be]

/-- The code's `Q(l)` is `Hash.RowSeg.outerPrefix l`. -/
theorem outerPrefix_data (l : ℕ) : (Flock.HmRow.rowPrefixSeg l).data.toList = Hash.RowSeg.outerPrefix l := by
  rw [Flock.HmRow.rowPrefixSeg, pad_data, Hash.RowSeg.outerPrefix, ByteArray.toList_data_append,
    ByteArray.toList_data_append, ByteArray.toList_data_append, push_zero_data, be64_data, be64_data, be32_1024]
  rfl

/-- The code's `S(j, s, m)` is `Hash.RowSeg.segmentPrefix j s m`. -/
theorem segmentPrefix_data (j s m : ℕ) :
    (Flock.HmRow.segmentPrefix j s m).data.toList = Hash.RowSeg.segmentPrefix j s m := by
  rw [Flock.HmRow.segmentPrefix, pad_data, Hash.RowSeg.segmentPrefix, ByteArray.toList_data_append,
    ByteArray.toList_data_append, ByteArray.toList_data_append, push_zero_data, be64_data, be64_data, be64_data]

theorem size_rowPrefixSeg (l : ℕ) : (Flock.HmRow.rowPrefixSeg l).size = 128 := by
  rw [size_of_data (outerPrefix_data l), Hash.RowSeg.outerPrefix, length_padTo]
  have : (Hash.cstr "verity/sha512-row-seg/v1").length = 25 := by
    rw [← push_zero_data, Array.length_toList, ByteArray.size_data]
    decide +kernel
  simp only [List.length_append, Hash.BindingProofs.be_length, this]
  norm_num

theorem size_segmentPrefix (j s m : ℕ) : (Flock.HmRow.segmentPrefix j s m).size = 128 := by
  rw [size_of_data (segmentPrefix_data j s m), Hash.RowSeg.segmentPrefix, length_padTo]
  have : (Hash.cstr "verity/sha512-row-seg/v1/segment").length = 33 := by
    rw [← push_zero_data, Array.length_toList, ByteArray.size_data]
    decide +kernel
  simp only [List.length_append, Hash.BindingProofs.be_length, this]
  norm_num

/-- **The outer generator** (`Sha512RowSegOuter_v1{N_BITS}`): whenever its rows hold, its words are SHA-512 of `Q(l)` and
the digests' bytes. -/
theorem sound_outer {G : Hash.Msg → Verity.Digest} (hG : IsSha512 G) (l : ℕ) (d : List Word) :
    Sound (Flock.ReadNet.outer l d) fun z out => Words64 d →
      out.length = 8 ∧ Words64 out ∧
        (G (Hash.RowSeg.outerPrefix l ++ bytesBE (d.map (bv z)))).bytes = bytesBE (out.map (bv z)) := by
  simp only [Flock.ReadNet.outer]
  refine Sound.mono (sound_shaW _ _ _) fun z out h hd => ?_
  have hp := length_padTail (128 + 8 * d.length)
  have hb := blocks_mul (128 + 8 * d.length)
  have hn : (d ++ Flock.ReadNet.constWords (Flock.ReadNet.padTail (128 + 8 * d.length))).length / 16 =
      blocks (128 + 8 * d.length) - 1 := by
    rw [List.length_append, length_constWords]
    omega
  obtain ⟨o1, o2, o3⟩ := h (length_midstate _) (words64_midstate _)
    (by rw [List.length_append, length_constWords]; omega) (hd.append (words64_constWords _))
  refine ⟨o1, o2, ?_⟩
  rw [← outerPrefix_data, hash_mid hG (size_rowPrefixSeg l), o3, hn, map_bv_midstate, List.map_append,
    map_bv_constWords, wordsBE_append, length_bytesBE, List.length_map]

/-! ## The row -/

/-- What `rowSeg`'s constant callee must compute: the bit it is asked for. -/
def ConstSpec (const : Bool → M Lin) : Prop := ∀ b, Sound (const b) fun z r => ev z r = b

/-- What `rowSeg`'s segment callee must compute: from words holding a block's chaining value, SHA-512 of the block and
the bits' bytes. -/
def SegSpec (seg : ℕ → List Word → List Lin → M (List Word)) : Prop :=
  ∀ (m : ℕ) (pre : ByteArray) (cv : List Word) (bits : List Lin), pre.size = 128 →
    Sound (seg m cv bits) fun z out => cv.length = 8 → Words64 cv → cv.map (bv z) = chainV pre 1 → bits.length = m →
      out.length = 8 ∧ Words64 out ∧
        (Verity.sha512 (pre.data.toList ++ Hash.RowSeg.packBits (evs z bits))).bytes = bytesBE (out.map (bv z))

/-- What `rowSeg`'s outer callee must compute: SHA-512 of `Q(l)` and the digests' bytes. -/
def OuterSpec (out : ℕ → List Word → M (List Word)) : Prop :=
  ∀ (l : ℕ) (d : List Word), Sound (out l d) fun z o => Words64 d →
    o.length = 8 ∧ Words64 o ∧
      (Verity.sha512 (Hash.RowSeg.outerPrefix l ++ bytesBE (d.map (bv z)))).bytes = bytesBE (o.map (bv z))

theorem constBit_spec : ConstSpec Flock.ReadNet.constBit := fun b => Sound.pure' fun z => ev_const z b

theorem segment_spec : SegSpec Flock.ReadNet.segment := fun m _ cv bits hpre =>
  sound_segment_mid isSha512_sha512 m hpre cv bits

theorem outer_spec : OuterSpec Flock.ReadNet.outer := fun l d => sound_outer isSha512_sha512 l d

theorem length_cvWords (zero one : Lin) (pre : ByteArray) : (Flock.ReadNet.cvWords zero one pre).length = 8 := by
  simp [Flock.ReadNet.cvWords]

theorem words64_cvWords (zero one : Lin) (pre : ByteArray) : Words64 (Flock.ReadNet.cvWords zero one pre) := by
  intro w hw
  simp only [Flock.ReadNet.cvWords, List.mem_map] at hw
  obtain ⟨i, _, rfl⟩ := hw
  simp

/-- Words over a 0 and a 1 hold the chaining value they spell. -/
theorem map_bv_cvWords (z : Var → Bool) {zero one : Lin} (h0 : ev z zero = false) (h1 : ev z one = true)
    (pre : ByteArray) : (Flock.ReadNet.cvWords zero one pre).map (bv z) = chainV pre 1 := by
  rw [chainV, chain_one, Flock.ReadNet.cvWords, List.map_map]
  refine List.map_congr_left fun i _ => ?_
  dsimp only [Function.comp_apply]
  refine bv_ext (by simp) fun k hk => ?_
  simp only [FlockSoundness.Discharge.Hm.bit, List.getElem?_map, List.getElem?_range hk, Option.map_some,
    Option.getD_some]
  split
  · rename_i hc
    rw [h1, hc]
  · rename_i hc
    rw [h0]
    exact (Bool.eq_false_iff.2 hc).symm

/-- A generator mapped over a list: what it computes of each element, in order. -/
theorem sound_mapM_eq {β γ δ : Type} (f : β → M γ) (Q : (Var → Bool) → β → Prop) (R : (Var → Bool) → γ → Prop)
    (F : (Var → Bool) → γ → δ) (g : (Var → Bool) → β → δ)
    (hf : ∀ x, Sound (f x) fun z r => Q z x → R z r ∧ F z r = g z x) :
    ∀ l : List β, Sound (l.mapM f) fun z rs => (∀ x ∈ l, Q z x) → (∀ r ∈ rs, R z r) ∧ rs.map (F z) = l.map (g z)
  | [] => Sound.pure' fun z _ => ⟨fun _ h => absurd h (by simp), rfl⟩
  | x :: l => by
    rw [List.mapM_cons]
    refine Sound.bind' (hf x) fun r => Sound.bind' (sound_mapM_eq f Q R F g hf l) fun rs =>
      Sound.pure' fun z hl hr hq => ?_
    obtain ⟨hr1, hr2⟩ := hr (hq x List.mem_cons_self)
    obtain ⟨hl1, hl2⟩ := hl fun y hy => hq y (List.mem_cons_of_mem _ hy)
    refine ⟨fun v hv => ?_, by simp [hr2, hl2]⟩
    rcases List.mem_cons.1 hv with rfl | hv
    · exact hr1
    · exact hl1 v hv

theorem bytesBE_flatten (z : Var → Bool) : ∀ d : List (List Word),
    bytesBE (d.flatten.map (bv z)) = d.flatMap fun ds => bytesBE (ds.map (bv z))
  | [] => rfl
  | ds :: d => by
    rw [List.flatten_cons, List.map_append, List.flatMap_cons, ← bytesBE_flatten z d]
    simp only [bytesBE, List.flatMap_append]

theorem take_min_eq {α : Type} (l : List α) (a k : ℕ) :
    (l.drop a).take (min (l.length - a) k) = (l.drop a).take k := by
  apply List.ext_getElem
  · simp only [List.length_take, List.length_drop]
    omega
  · intro i h1 h2
    simp only [List.getElem_take]

/-- Segment `j`'s message is the code's prefix for it and the bytes of the bits the row generator hands it. -/
theorem segmentMsg_eq (z : Var → Bool) {row : List Lin} {n : ℕ} (hl : row.length = n) (j : ℕ) :
    Hash.RowSeg.segmentMsg (evs z row) j =
      (Flock.HmRow.segmentPrefix j ((n + Flock.HmRow.ROW_SEG_BITS - 1) / Flock.HmRow.ROW_SEG_BITS)
        (min (n - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS)).data.toList ++
      Hash.RowSeg.packBits (evs z ((row.drop (Flock.HmRow.ROW_SEG_BITS * j)).take
        (min (n - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS))) := by
  have hR : Flock.HmRow.ROW_SEG_BITS = Hash.RowSeg.SEG_BITS := rfl
  have hseg : Hash.RowSeg.segment (evs z row) j = evs z ((row.drop (Flock.HmRow.ROW_SEG_BITS * j)).take
      (min (n - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS)) := by
    have hlm : (row.map (ev z)).length = n := by rw [List.length_map, hl]
    simp only [Hash.RowSeg.segment, evs, List.map_take, List.map_drop, hR]
    rw [← hlm, take_min_eq]
  rw [Hash.RowSeg.segmentMsg, segmentPrefix_data, hseg, length_evs, length_evs, hl, List.length_take, List.length_drop,
    hl, hR, Hash.RowSeg.segCount, Nat.min_eq_left (by omega)]

/-- **The row generator computes `sha512/row-seg/v1`** wherever its callees compute their constants and SHA-512: on a row
of `n` forms it returns 8 words that hold the row's digest whenever its rows hold. -/
theorem rowSeg_spec {const : Bool → M Lin} {seg : ℕ → List Word → List Lin → M (List Word)}
    {out : ℕ → List Word → M (List Word)} (hconst : ConstSpec const) (hseg : SegSpec seg) (hout : OuterSpec out)
    (n : ℕ) : RowSegSpec n (Flock.ReadNet.rowSeg const seg out n) := by
  intro row
  simp only [Flock.ReadNet.rowSeg]
  refine Sound.bind' (hconst false) fun zero => Sound.bind' (hconst true) fun one => ?_
  refine Sound.bind' (sound_mapM_eq _ (fun z _ => row.length = n ∧ ev z zero = false ∧ ev z one = true)
    (fun _ ds => ds.length = 8 ∧ Words64 ds) (fun z ds => bytesBE (ds.map (bv z)))
    (fun z j => (Verity.sha512 (Hash.RowSeg.segmentMsg (evs z row) j)).bytes) (fun j => ?_) _) fun d =>
    Sound.mono (hout n d.flatten) fun z o ho hd h1 h0 hl => ?_
  · refine Sound.mono (hseg _ (Flock.HmRow.segmentPrefix j ((n + Flock.HmRow.ROW_SEG_BITS - 1) / Flock.HmRow.ROW_SEG_BITS)
      (min (n - Flock.HmRow.ROW_SEG_BITS * j) Flock.HmRow.ROW_SEG_BITS)) _ _ (size_segmentPrefix _ _ _))
      fun z ds h hq => ?_
    obtain ⟨hl, h0, h1⟩ := hq
    obtain ⟨o1, o2, o3⟩ := h (length_cvWords _ _ _) (words64_cvWords _ _ _) (map_bv_cvWords z h0 h1 _)
      (by simp only [List.length_take, List.length_drop, hl]; omega)
    refine ⟨⟨o1, o2⟩, ?_⟩
    show bytesBE (ds.map (bv z)) = (Verity.sha512 (Hash.RowSeg.segmentMsg (evs z row) j)).bytes
    rw [← o3, segmentMsg_eq z hl j]
  · obtain ⟨hd1, hd2⟩ := hd fun _ _ => ⟨hl, h0, h1⟩
    have hw : Words64 d.flatten := fun w hw => by
      obtain ⟨ds, hds, hw'⟩ := List.mem_flatten.1 hw
      exact (hd1 ds hds).2 w hw'
    obtain ⟨o1, o2, o3⟩ := ho hw
    refine ⟨⟨o1, o2⟩, ?_⟩
    have hmsg : bytesBE (d.flatten.map (bv z)) =
        (Hash.RowSeg.segmentMsgs (evs z row)).flatMap fun m => (Verity.sha512 m).bytes := by
      rw [bytesBE_flatten]
      show (d.map fun ds => bytesBE (ds.map (bv z))).flatten =
        ((Hash.RowSeg.segmentMsgs (evs z row)).map fun m => (Verity.sha512 m).bytes).flatten
      rw [hd2, Hash.RowSeg.segmentMsgs, List.map_map, length_evs, hl]
      rfl
    rw [← Hash.BindingProofs.bytes_inj, wordsDigest_bytes z ⟨o1, o2⟩, ← o3, hmsg, Hash.RowSeg.digest,
      Hash.RowSeg.outerMsg, length_evs, hl]

/-- **Part A**: the row generator over the constant, segment and outer generators meets `Gates.RowSegSpec`, for every
row length. -/
theorem rowSegNet_spec (n : ℕ) : RowSegSpec n (Flock.ReadNet.rowSegNet n) :=
  rowSeg_spec constBit_spec segment_spec outer_spec n

end Accounting.ReadUnit.Gates
