import FlockSoundness.Discharge.ScopeW.Scope
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

/-- **`sha512/row/v2`'s encoding is injective**: `RowV2.enc_inj` on bit rows, `bitCode` elsewhere, and a bit row's
encoding shorter than every other value's. -/
theorem rowEncV2_inj : Function.Injective (rowEncV2 (W := W)) := sorry

/-- **Two segmented rows with one outer message collide in a segment**: `Q(ℓ)` fixes `ℓ`, so `s`; the `64 s` digest
bytes fix every segment's digest; two different values of one width differ in a segment, whose messages then differ and
share a digest. -/
theorem segPair_spec {v v' : BitVal W} (hv : v ≠ v') (h : rowEncSeg v = rowEncSeg v') : Collides (segPair v v') :=
  sorry

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

/-! ## At `keyProg`'s commit strings -/

open FlockSoundness.Discharge.Layout (keyN)

variable (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ)

/-- **Each commit string's encoding** at `keyProg`'s readers: its row's schema's (`rowEncOf` of its port) at a row;
`Layout.rowEnc` at a single gate, which is no row. -/
def encW : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n) → BitVal W → List UInt8
  | .inl (_, p) => rowEncOf c.ports[p]
  | .inr _ => rowEnc

/-- Each commit string's extractor at equal encodings (`rowPairOf`). -/
def pairW : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n) → BitVal W → BitVal W → List UInt8 × List UInt8
  | .inl (_, p) => rowPairOf c.ports[p]
  | .inr _ => fun v v' => (rowEnc v, rowEnc v')

variable {c hU n} in
theorem pairW_spec : ∀ (q : (Fin n × Fin c.ports.size) ⊕ Fin (keyN c hU n)) (v v' : BitVal W), v ≠ v' →
    encW c hU n q v = encW c hU n q v' → Collides (pairW c hU n q v v')
  | .inl (_, p), _, _, hv, h => rowPairOf_spec c.ports[p] hv h
  | .inr _, _, _, hv, h => absurd (rowEnc_inj h) hv

end FlockSoundness.Discharge.ScopeW
