import FlockSoundness.Discharge.ScopeW.Scope
import FlockSoundness.Discharge.ScopeW.RowBytes
import FlockSoundness.Discharge.Layout.RowEnc
import FlockSoundness.Binding.Rows

set_option Elab.async false

/-!
# The bytes each row schema's digest hashes, and their extractors

`Layout.rowEnc` is the byte string `sha512/row/v1` hashes for a row; it is injective, so `ExplicitLeaf.hm96` binds a
row's value through it. Under the widened scope a row's schema is its port's (`HmRow.prefixOf`'s dispatch), so the
encoding is per row (`rowEncOf`):

* `rowEncV2`: `sha512/row/v2`'s message, `P(ℓ)` and the row's bits packed (`FlockLevel3.RowV2.enc`), for a value of
  fewer than `2^64` bits with nothing set past them (`BitShaped`); injective (`rowEncV2_inj`);
* `rowEncSeg`: the message `sha512/row-seg/v1`'s outer hash reads, `Q(ℓ)` and the digest of each segment's message
  `S(j, s, m) ‖ bits` (`segMsg`, `HmRow.segDigest`'s). It is not injective: two values with one outer message differ in
  a segment whose two messages share a digest, and `segPair` returns them, a SHA-512 collision (`segPair_spec`).

`rowPairOf` is each row's extractor at equal encodings (`rowPairOf_spec`): `segPair` at a segmented row; at the other
rows equal encodings are equal values.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Binding
open FlockSoundness.Discharge.Layout (BitVal rowEnc rowEnc_inj bitCode)

variable {W : ℕ}

/-- A value's bits below its width. -/
def valBits (v : BitVal W) : List Bool :=
  List.ofFn fun i : Fin v.1.val => v.2 ⟨i.val, lt_of_lt_of_le i.isLt (Nat.lt_succ_iff.1 v.1.isLt)⟩

/-- **A bit row's value**: fewer than `2^64` bits, and no bit set past them. -/
def BitShaped (v : BitVal W) : Prop := v.1.val < 2 ^ 64 ∧ ∀ i : Fin W, v.1.val ≤ i.val → v.2 i = false

instance : DecidablePred (BitShaped (W := W)) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- A code of every value that is no bit row, longer than every bit row's encoding under both schemas. -/
def junk (v : BitVal W) : List UInt8 := List.replicate (128 + 64 * (W + 1)) 0 ++ bitCode v

/-- **The bytes `sha512/row/v2` hashes for a value**: `P(ℓ)` and the bits packed, for a bit row of `ℓ` bits. -/
def rowEncV2 (v : BitVal W) : List UInt8 :=
  if BitShaped v then RowV2.enc (valBits v) else junk v

/-- The number of segments of a value's row: `⌈ℓ / 8192⌉` (`HmRow.ROW_SEG_BITS`). -/
def segCount (v : BitVal W) : ℕ := (v.1.val + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS

/-- Segment `j`'s bits: bits `8192 j` up to `8192 (j + 1)` of the row. -/
def segBits (v : BitVal W) (j : ℕ) : List Bool :=
  ((valBits v).drop (HmRow.ROW_SEG_BITS * j)).take HmRow.ROW_SEG_BITS

/-- **Segment `j`'s message**: `S(j, s, m)` and the segment's `m` bits packed, the inner hash's input in
`HmRow.segDigest`. -/
def segMsg (v : BitVal W) (j : ℕ) : List UInt8 :=
  (HmRow.segmentPrefix j (segCount v) (segBits v j).length).data.toList ++ HmRow.packBits (segBits v j)

/-- **The bytes `sha512/row-seg/v1`'s outer hash reads for a value**: `Q(ℓ)` and each segment's digest, for a bit row
of `ℓ` bits. -/
def rowEncSeg (v : BitVal W) : List UInt8 :=
  if BitShaped v then
    (HmRow.rowPrefixSeg v.1.val).data.toList ++
      (List.range (segCount v)).flatMap fun j => digBytes (Refine.H512 (segMsg v j))
  else junk v

/-- **The bytes a row's digest hashes**, by its port's schema as `HmRow.prefixOf` picks it: `sha512/row-seg/v1`'s outer
message, `sha512/row/v2`'s, or `sha512/row/v1`'s (`Layout.rowEnc`). -/
def rowEncOf (p : Port) : BitVal W → List UInt8 :=
  if p.segs.isSome then rowEncSeg else if p.v2 then rowEncV2 else rowEnc

/-- **The extractor of a segmented row at one outer message**: the first segment whose two messages differ. -/
def segPair (v v' : BitVal W) : List UInt8 × List UInt8 :=
  match (List.range (segCount v)).find? (fun j => segMsg v j != segMsg v' j) with
  | some j => (segMsg v j, segMsg v' j)
  | none => (rowEncSeg v, rowEncSeg v')

/-- **A row's extractor at equal encodings**: `segPair` at a segmented row, the encodings elsewhere. -/
def rowPairOf (p : Port) (v v' : BitVal W) : List UInt8 × List UInt8 :=
  if p.segs.isSome then segPair v v' else (rowEncOf p v, rowEncOf p v')

/-! ## Injectivity and the segment extractor -/

theorem length_valBits (v : BitVal W) : (valBits v).length = v.1.val := List.length_ofFn

/-- **A bit row is its bits**: two bit rows with the same bits below their widths are one value. -/
theorem valBits_inj {v v' : BitVal W} (hv : BitShaped v) (hv' : BitShaped v') (h : valBits v = valBits v') :
    v = v' := by
  have hl : v.1 = v'.1 := Fin.ext (by rw [← length_valBits v, ← length_valBits v', h])
  refine Prod.ext hl (funext fun i => ?_)
  by_cases hi : i.val < v.1.val
  · have e := List.getElem_of_eq h (by rw [length_valBits]; exact hi)
    simpa [valBits] using e
  · rw [hv.2 i (by omega), hv'.2 i (by rw [← hl]; omega)]

theorem length_junk (v : BitVal W) : (junk v).length = 128 + 64 * (W + 1) + (2 * W + 1) := by
  simp only [junk, List.length_append, List.length_replicate, Layout.bitCode_length]

theorem junk_inj {v v' : BitVal W} (h : junk v = junk v') : v = v' :=
  Layout.bitCode_inj (List.append_inj h rfl).2

theorem length_rowEncV2 {v : BitVal W} (hv : BitShaped v) : (rowEncV2 v).length = 128 + (v.1.val + 7) / 8 := by
  simp only [rowEncV2, hv, ↓reduceIte, RowV2.enc, List.length_append, RowV2.prefix_length, RowV2.packBits_length,
    length_valBits]

/-- **`sha512/row/v2`'s encoding is injective**: `RowV2.enc_inj` on bit rows, `bitCode` elsewhere, and a bit row's
encoding shorter than every other value's. -/
theorem rowEncV2_inj : Function.Injective (rowEncV2 (W := W)) := by
  intro v v' h
  have hW := v.1.isLt
  have hW' := v'.1.isLt
  by_cases hv : BitShaped v <;> by_cases hv' : BitShaped v'
  · simp only [rowEncV2, hv, hv', ↓reduceIte] at h
    have e := RowV2.enc_inj (a₁ := ⟨valBits v, by rw [length_valBits]; exact hv.1⟩)
      (a₂ := ⟨valBits v', by rw [length_valBits]; exact hv'.1⟩) h
    exact valBits_inj hv hv' (congrArg Subtype.val e)
  · have hl := congrArg List.length h
    rw [length_rowEncV2 hv] at hl
    simp only [rowEncV2, hv', ↓reduceIte, length_junk] at hl
    omega
  · have hl := congrArg List.length h
    rw [length_rowEncV2 hv'] at hl
    simp only [rowEncV2, hv, ↓reduceIte, length_junk] at hl
    omega
  · simp only [rowEncV2, hv, hv', ↓reduceIte] at h
    exact junk_inj h

theorem segCount_le (v : BitVal W) : segCount v ≤ W := by
  have := v.1.isLt
  simp only [segCount, HmRow.ROW_SEG_BITS]
  omega

theorem length_rowEncSeg {v : BitVal W} (hv : BitShaped v) : (rowEncSeg v).length = 128 + 64 * segCount v := by
  simp only [rowEncSeg, hv, ↓reduceIte, List.length_append, RowBytes.rowPrefixSeg_length]
  rw [RowBytes.length_flatMap_range fun j => Refine.hL512 (Refine.H512 (segMsg v j))]
  rfl

/-- **The segments rebuild a row**: two values of one width with the same segments have the same bits. -/
theorem valBits_eq_of_segBits {v v' : BitVal W} (hl : v.1 = v'.1)
    (h : ∀ j < segCount v, segBits v j = segBits v' j) : valBits v = valBits v' := by
  refine List.ext_getElem? fun i => ?_
  by_cases hi : i < v.1.val
  · have hj : i / HmRow.ROW_SEG_BITS < segCount v := by
      simp only [segCount, HmRow.ROW_SEG_BITS]
      omega
    have e := congrArg (·[i % HmRow.ROW_SEG_BITS]?) (h _ hj)
    simpa only [segBits, List.getElem?_take, List.getElem?_drop,
      Nat.mod_lt i (by decide : 0 < HmRow.ROW_SEG_BITS), ↓reduceIte, Nat.div_add_mod] using e
  · rw [List.getElem?_eq_none (by rw [length_valBits]; omega),
      List.getElem?_eq_none (by rw [length_valBits, ← hl]; omega)]

/-- **Two segmented rows with one outer message collide in a segment**: `Q(ℓ)` fixes `ℓ`, so `s`; the `64 s` digest
bytes fix every segment's digest; two different values of one width differ in a segment, whose messages then differ and
share a digest. -/
theorem segPair_spec {v v' : BitVal W} (hv : v ≠ v') (h : rowEncSeg v = rowEncSeg v') : Collides (segPair v v') := by
  by_cases hs : BitShaped v <;> by_cases hs' : BitShaped v'
  · simp only [rowEncSeg, hs, hs', ↓reduceIte] at h
    obtain ⟨hq, hd⟩ :=
      List.append_inj h (by rw [RowBytes.rowPrefixSeg_length, RowBytes.rowPrefixSeg_length])
    have hl : v.1 = v'.1 := Fin.ext (RowBytes.rowPrefixSeg_inj hs.1 hs'.1 hq)
    have hc : segCount v = segCount v' := by simp only [segCount, hl]
    rw [← hc] at hd
    have hdig : ∀ j < segCount v, Refine.H512 (segMsg v j) = Refine.H512 (segMsg v' j) := fun j hj =>
      Refine.enc512.dig_inj (RowBytes.flatMap_range_inj
        (f := fun j => digBytes (Refine.H512 (segMsg v j))) (g := fun j => digBytes (Refine.H512 (segMsg v' j)))
        (fun _ => Refine.hL512 _) (fun _ => Refine.hL512 _) hd j hj)
    unfold segPair
    split
    · rename_i j hj
      exact ⟨by simpa using List.find?_some hj, hdig j (List.mem_range.1 (List.mem_of_find?_eq_some hj))⟩
    · rename_i hn
      refine absurd (valBits_inj hs hs' (valBits_eq_of_segBits hl fun j hj => ?_)) hv
      have e : segMsg v j = segMsg v' j := by simpa using List.find?_eq_none.1 hn j (List.mem_range.2 hj)
      unfold segMsg at e
      exact RowV2.packBits_inj (by simp only [segBits, List.length_take, List.length_drop, length_valBits, hl])
        (List.append_inj e (by rw [RowBytes.segmentPrefix_length, RowBytes.segmentPrefix_length])).2
  · have hl := congrArg List.length h
    rw [length_rowEncSeg hs] at hl
    simp only [rowEncSeg, hs', ↓reduceIte, length_junk] at hl
    have := segCount_le v
    omega
  · have hl := congrArg List.length h
    rw [length_rowEncSeg hs'] at hl
    simp only [rowEncSeg, hs, ↓reduceIte, length_junk] at hl
    have := segCount_le v'
    omega
  · simp only [rowEncSeg, hs, hs', ↓reduceIte] at h
    exact absurd (junk_inj h) hv

/-- **`rowPairOf` outputs a SHA-512 collision** whenever two different values have one encoding. -/
theorem rowPairOf_spec (p : Port) {v v' : BitVal W} (hv : v ≠ v') (h : rowEncOf p v = rowEncOf p v') :
    Collides (rowPairOf p v v') := by
  unfold rowPairOf
  unfold rowEncOf at h
  by_cases hs : p.segs.isSome
  · simp only [hs, ↓reduceIte] at h ⊢
    exact segPair_spec hv h
  · simp only [hs, Bool.false_eq_true, ↓reduceIte] at h
    by_cases h2 : p.v2
    · simp only [h2, ↓reduceIte] at h
      exact absurd (rowEncV2_inj h) hv
    · simp only [h2, Bool.false_eq_true, ↓reduceIte] at h
      exact absurd (rowEnc_inj h) hv

/-! ## At a program's commit strings -/

variable (c : Flock.Circuit) (n : ℕ) (G : Type)

/-- **Each commit string's encoding** at readers of rows `(unit, port)` and single gates `G` (`keyProg`'s,
`keyProgP`'s): its row's schema's (`rowEncOf` of its port) at a row; `Layout.rowEnc` at a single gate, which is no row. -/
def encW : (Fin n × Fin c.ports.size) ⊕ G → BitVal W → List UInt8
  | .inl (_, p) => rowEncOf c.ports[p]
  | .inr _ => rowEnc

/-- Each commit string's extractor at equal encodings (`rowPairOf`). -/
def pairW : (Fin n × Fin c.ports.size) ⊕ G → BitVal W → BitVal W → List UInt8 × List UInt8
  | .inl (_, p) => rowPairOf c.ports[p]
  | .inr _ => fun v v' => (rowEnc v, rowEnc v')

variable {c n G} in
theorem pairW_spec : ∀ (q : (Fin n × Fin c.ports.size) ⊕ G) (v v' : BitVal W), v ≠ v' →
    encW c n G q v = encW c n G q v' → Collides (pairW c n G q v v')
  | .inl (_, p), _, _, hv, h => rowPairOf_spec c.ports[p] hv h
  | .inr _, _, _, hv, h => absurd (rowEnc_inj h) hv

end FlockSoundness.Discharge.ScopeW
