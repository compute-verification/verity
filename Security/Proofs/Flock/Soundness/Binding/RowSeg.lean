import Proofs.Flock.Soundness.Binding.Leaf
import Flock.HmRow

/-!
# `hm96-sha512/row-seg/v1` leaves, with their extractor

A `sha512/row-seg/v1` row of `l` bits hashes in two levels (`HmRow.segDigest`): each segment `j` of at most 8,192 bits
under its own first block `S(j, s, m)` (`segMsg`), then the outer message `Q(l) ‖ d₀ ‖ … ‖ d_{s-1}` of the segments'
digests (`enc`, `segDigest_eq`). The outer message is not injective in the row, so hm96's extractor can't stop at it:

* `ExplicitLeaf.hm96X` is `ExplicitLeaf.hm96` for any row bytes `enc`, given an extractor for two different values with
  the same bytes (`hm96XPair`: that one when the bytes agree, else hm96's).
* `extPair` is row-seg's: the messages of the first segment where two rows differ (`diffSeg`). Two rows of `l` bits that
  differ differ in a segment (`exists_segBytes_ne`), and equal outer messages hold equal digests of every segment
  (`digest_eq_of_enc`), so that segment's messages collide (`extPair_spec`).
* `ExplicitLeaf.rowSeg l` is the leaf of the rows of `l` bits, and `rowSeg_treeLeaf` ties it to the executable's: the
  `Hm96.Default512.treeLeaf` of `Hm96.Default512.commitString (HmRow.segDigest l row) salt`.
-/

namespace FlockSoundness.Binding

open Flock Refine Merkle

variable {Val : Type}

/-! ## Leaves whose bytes are not injective -/

/-- **hm96's extractor over row bytes `enc`**, with `ext` for two values whose bytes agree: `ext` then, else hm96's
(`hm96Pair`) on the two rows' bytes. -/
def hm96XPair (enc : Val → List UInt8) (ext : Val → Val → List UInt8 × List UInt8) (v : Val) (y : List UInt8) (v' : Val)
    (y' : List UInt8) : List UInt8 × List UInt8 :=
  if enc v = enc v' then ext v v' else hm96Pair (enc v) y (enc v') y'

/-- **An hm96 leaf over row bytes that need not be injective**: the leaf of `v` under salt `y` is
`SHA-512(leaf_prefix ‖ b ‖ c)` of the commit string of `enc v`, and `ext` finds a collision under two different values
whose bytes agree. -/
def ExplicitLeaf.hm96X (enc : Val → List UInt8) (ext : Val → Val → List UInt8 × List UInt8)
    (hext : ∀ v v', v ≠ v' → enc v = enc v' → Collides (ext v v')) : ExplicitLeaf Val where
  leaf v y := H512 (hm96Input (enc v) y)
  pair := hm96XPair enc ext
  pair_spec v y v' y' hv h := by
    unfold hm96XPair
    split_ifs with he
    · exact hext v v' hv he
    · exact hm96Pair_spec _ _ _ _ he h

/-- **`hm96X`'s leaf is the executable's**: the `Hm96.Default512.treeLeaf` of the commit string of the row digest
`Sha512.hash (enc v)` under the salt. -/
theorem hm96X_treeLeaf (enc : Val → List UInt8) (ext : Val → Val → List UInt8 × List UInt8)
    (hext : ∀ v v', v ≠ v' → enc v = enc v' → Collides (ext v v')) (v : Val) (salt : ByteArray) :
    ((ExplicitLeaf.hm96X enc ext hext).leaf v salt.data.toList).1 =
      Hm96.Default512.treeLeaf (Hm96.Default512.commitString (Sha512.hash ⟨(enc v).toArray⟩) salt) := by
  show (H512 (K512.leafPrefix ++ (xorBytes (enc512.dig (H512 (enc v))) (K512.mask salt.data.toList) ++
      enc512.dig (H512 (K512.saltPrefix ++ salt.data.toList))))).1 = _
  rw [H512_val, dig_eq, dig_eq, H512_val, H512_val, K512_mask, bytes_of_list,
    ← xor_range _ _ 64 (sha512_size _) (mask_size salt)]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ (_ ++
      (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList)).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++
      Hm96.Default512.commitString (Sha512.hash ⟨(enc v).toArray⟩) salt)
  rw [← data_append Hm96.Default512.saltPrefix salt, bytes_of_list, ← data_append, ← data_append, bytes_of_list]
  rfl

/-! ## `sha512/row-seg/v1` -/

namespace RowSeg

/-- The segments of a row of `l` bits: `ceil(l / 8192)`. -/
def count (l : ℕ) : ℕ := (l + HmRow.ROW_SEG_BITS - 1) / HmRow.ROW_SEG_BITS

/-- Segment `j`'s bits: the rest of the row, at most 8,192. -/
def segBits (l j : ℕ) : ℕ := min (l - HmRow.ROW_SEG_BITS * j) HmRow.ROW_SEG_BITS

theorem size_segShape (l : ℕ) : (HmRow.segShape l).1.size = count l := by
  simp [HmRow.segShape, count]

theorem segShape_get {l j : ℕ} (hj : j < count l) :
    (HmRow.segShape l).1[j]! = (segBits l j, ((segBits l j + 7) / 8 + 17 + 127) / 128) := by
  have hj' : j < (HmRow.segShape l).1.size := by rw [size_segShape]; exact hj
  rw [getElem!_pos _ j hj']
  simp [HmRow.segShape, segBits]

/-- Segment `j`'s bytes of a row whose bytes are `row`: from byte `1024 j`, `(m + 7) / 8` of them. -/
def segBytes (l : ℕ) (row : List UInt8) (j : ℕ) : List UInt8 := (row.drop (1024 * j)).take ((segBits l j + 7) / 8)

/-- **Segment `j`'s message**: its first block `S(j, s, m)`, then its bytes. -/
def segMsg (l : ℕ) (row : List UInt8) (j : ℕ) : List UInt8 :=
  (HmRow.segmentPrefix j (count l) (segBits l j)).data.toList ++ segBytes l row j

/-- The outer message's data: the segments' digests, in order. -/
def digests (l : ℕ) (row : List UInt8) : List UInt8 :=
  (List.range (count l)).flatMap fun j => digBytes (H512 (segMsg l row j))

/-- **The bytes `sha512/row-seg/v1`'s outer hash reads**: `Q(l)`, then the digests. -/
def enc (l : ℕ) (row : List UInt8) : List UInt8 := (HmRow.rowPrefixSeg l).data.toList ++ digests l row

theorem foldl_append_data (f : ℕ → ByteArray) (a : ByteArray) (n : ℕ) :
    ((List.range n).foldl (fun acc j => acc ++ f j) a).data.toList =
      a.data.toList ++ (List.range n).flatMap fun j => (f j).data.toList := by
  induction n generalizing a with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, data_append, ih, List.flatMap_append,
      List.flatMap_singleton, List.append_assoc]

/-- **`HmRow.segDigest` is SHA-512 of `enc`**: the outer hash over `Q(l)` and each segment's digest. -/
theorem segDigest_eq (l : ℕ) (row : List UInt8) :
    HmRow.segDigest l ⟨row.toArray⟩ = Sha512.hash ⟨(enc l row).toArray⟩ := by
  let f : ℕ → ByteArray := fun j => Sha512.hash (HmRow.segmentPrefix j (count l) (HmRow.segShape l).1[j]!.1 ++
    (⟨row.toArray⟩ : ByteArray).extract (1024 * j) (1024 * j + ((HmRow.segShape l).1[j]!.1 + 7) / 8))
  have hf : ∀ j ∈ List.range (count l), (f j).data.toList = digBytes (H512 (segMsg l row j)) := by
    intro j hj
    have hj' := List.mem_range.1 hj
    have hb : HmRow.segmentPrefix j (count l) (HmRow.segShape l).1[j]!.1 ++
        (⟨row.toArray⟩ : ByteArray).extract (1024 * j) (1024 * j + ((HmRow.segShape l).1[j]!.1 + 7) / 8) =
        ⟨(segMsg l row j).toArray⟩ := by
      refine ByteArray.ext (Array.ext' ?_)
      rw [data_append, ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop, segShape_get hj',
        List.toList_toArray, List.toList_toArray, segMsg, segBytes, Nat.add_sub_cancel_left]
    show (Sha512.hash _).data.toList = _
    rw [hb, digBytes, dig_eq, H512_val]
  have e := foldl_append_data f (HmRow.rowPrefixSeg l) (count l)
  rw [List.flatMap_congr hf] at e
  have hs := size_segShape l
  unfold HmRow.segDigest
  simp only [hs]
  refine congrArg Sha512.hash (ByteArray.ext (Array.ext' ?_))
  refine e.trans ?_
  rw [List.toList_toArray]
  rfl

theorem getElem?_segBytes {l : ℕ} {row : List UInt8} {i : ℕ} (hi : i < (l + 7) / 8) :
    (segBytes l row (i / 1024))[i % 1024]? = row[i]? := by
  have hm : i % 1024 < (segBits l (i / 1024) + 7) / 8 := by
    unfold segBits HmRow.ROW_SEG_BITS
    omega
  rw [segBytes, List.getElem?_take_of_lt hm, List.getElem?_drop]
  congr 1
  omega

theorem div_lt_count {l i : ℕ} (hi : i < (l + 7) / 8) : i / 1024 < count l := by
  unfold count HmRow.ROW_SEG_BITS
  omega

/-- **Two different rows of `l` bits differ in a segment.** -/
theorem exists_segBytes_ne {l : ℕ} {row row' : List UInt8} (hl : row.length = (l + 7) / 8)
    (hl' : row'.length = (l + 7) / 8) (hne : row ≠ row') : ∃ j < count l, segBytes l row j ≠ segBytes l row' j := by
  by_contra hall
  push Not at hall
  refine hne (List.ext_getElem? fun i => ?_)
  by_cases hi : i < (l + 7) / 8
  · rw [← getElem?_segBytes hi, ← getElem?_segBytes hi, hall _ (div_lt_count hi)]
  · rw [List.getElem?_eq_none (by omega), List.getElem?_eq_none (by omega)]

theorem take_drop_flatMap_range {α : Type} {k : ℕ} {f : ℕ → List α} (hf : ∀ j, (f j).length = k) {n j : ℕ}
    (hj : j < n) : (((List.range n).flatMap f).drop (k * j)).take k = f j := by
  have hlen : ∀ n, ((List.range n).flatMap f).length = k * n := fun n => by
    rw [List.length_flatMap, List.map_congr_left (g := fun _ => k) fun j _ => hf j, List.map_const', List.sum_replicate,
      List.length_range, smul_eq_mul, Nat.mul_comm]
  induction n with
  | zero => omega
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append, List.flatMap_singleton]
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | hj
    · have h1 : k * j ≤ ((List.range n).flatMap f).length := by rw [hlen]; exact Nat.mul_le_mul_left _ hj.le
      have h2 : k ≤ (((List.range n).flatMap f).drop (k * j)).length := by
        rw [List.length_drop, hlen, ← Nat.mul_sub]
        exact Nat.le_mul_of_pos_right _ (by omega)
      rw [List.drop_append_of_le_length h1, List.take_append_of_le_length h2, ih hj]
    · subst hj
      rw [List.drop_left' (hlen _), List.take_of_length_le (by rw [hf])]

/-- **Equal outer messages hold equal digests** of every segment. -/
theorem digest_eq_of_enc {l : ℕ} {row row' : List UInt8} (h : enc l row = enc l row') {j : ℕ} (hj : j < count l) :
    H512 (segMsg l row j) = H512 (segMsg l row' j) := by
  have hd : digests l row = digests l row' := List.append_cancel_left h
  have hf : ∀ (r : List UInt8) j, (digBytes (H512 (segMsg l r j))).length = 64 := fun r j => hL512 _
  have e := congrArg (fun d => (d.drop (64 * j)).take 64) hd
  simp only [digests] at e
  rw [take_drop_flatMap_range (hf row) hj, take_drop_flatMap_range (hf row') hj] at e
  exact enc512.dig_inj e

/-- The first segment where two rows' messages differ (`0` when none does). -/
def diffSeg (l : ℕ) (row row' : List UInt8) : ℕ :=
  ((List.range (count l)).find? fun j => segMsg l row j != segMsg l row' j).getD 0

/-- **row-seg's extractor**: the two rows' messages of the first segment where they differ. -/
def extPair (l : ℕ) (row row' : List UInt8) : List UInt8 × List UInt8 :=
  (segMsg l row (diffSeg l row row'), segMsg l row' (diffSeg l row row'))

/-- **row-seg's extractor outputs a SHA-512 collision** whenever two different rows of `l` bits have one outer
message: they differ in some segment, the first such segment's messages differ, and their digests agree. -/
theorem extPair_spec {l : ℕ} {row row' : List UInt8} (hl : row.length = (l + 7) / 8)
    (hl' : row'.length = (l + 7) / 8) (hne : row ≠ row') (h : enc l row = enc l row') :
    Collides (extPair l row row') := by
  obtain ⟨j, hj, hb⟩ := exists_segBytes_ne hl hl' hne
  have hm : (segMsg l row j != segMsg l row' j) = true := by
    simp only [bne_iff_ne, ne_eq, segMsg]
    exact fun e => hb (List.append_cancel_left e)
  obtain ⟨j₀, hfind⟩ : ∃ j₀, (List.range (count l)).find? (fun j => segMsg l row j != segMsg l row' j) = some j₀ := by
    rcases hf : (List.range (count l)).find? (fun j => segMsg l row j != segMsg l row' j) with _ | j₀
    · exact absurd hm (List.find?_eq_none.1 hf j (List.mem_range.2 hj))
    · exact ⟨j₀, rfl⟩
  have hd : diffSeg l row row' = j₀ := by rw [diffSeg, hfind, Option.getD_some]
  have h0 := List.find?_some hfind
  have hj₀ := List.mem_range.1 (List.mem_of_find?_eq_some hfind)
  unfold Collides extPair
  rw [hd]
  exact ⟨by simpa using h0, digest_eq_of_enc h hj₀⟩

end RowSeg

/-- **The rows of `l` bits**: their bytes, `(l + 7) / 8` of them. -/
abbrev SegRow (l : ℕ) := {row : List UInt8 // row.length = (l + 7) / 8}

/-- **`hm96-sha512/row-seg/v1`'s leaf** of the rows of `l` bits: hm96 over the outer message `Q(l) ‖ d₀ ‖ … ‖ d_{s-1}`,
with row-seg's extractor (`RowSeg.extPair`) when two rows have one outer message. -/
def ExplicitLeaf.rowSeg (l : ℕ) : ExplicitLeaf (SegRow l) :=
  ExplicitLeaf.hm96X (fun r => RowSeg.enc l r.1) (fun r r' => RowSeg.extPair l r.1 r'.1)
    fun r r' hne he => RowSeg.extPair_spec r.2 r'.2 (fun e => hne (Subtype.ext e)) he

/-- **row-seg's leaf is the executable's**: the `Hm96.Default512.treeLeaf` of the commit string of the row's
`HmRow.segDigest` under the salt, the leaf `HmRow.checkPublic` computes from a public `b ‖ c`. -/
theorem rowSeg_treeLeaf (l : ℕ) (r : SegRow l) (salt : ByteArray) :
    ((ExplicitLeaf.rowSeg l).leaf r salt.data.toList).1 =
      Hm96.Default512.treeLeaf (Hm96.Default512.commitString (HmRow.segDigest l ⟨r.1.toArray⟩) salt) := by
  rw [RowSeg.segDigest_eq]
  exact hm96X_treeLeaf _ _ _ r salt

end FlockSoundness.Binding
