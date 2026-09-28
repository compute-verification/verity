import Proofs.Flock.Soundness.Discharge.Layout.Pins
import Proofs.Flock.Soundness.Discharge.Layout.RowEnc
import Proofs.Flock.Soundness.Binding.RowsX
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegCommit

set_option Elab.async false

/-!
# A layout's rows each under its own encoding, and a `sha512/row-seg/v1` row's

`Readers.rows` hashes every row under one injective encoding (`Binding.HmRows`). A `sha512/row-seg/v1` row's outer
message (`RowSeg.enc`) is not injective in the row, so its rows are `Binding.HmRowsX`, each commit string under its own
encoding with an extractor:

* `encSeg l`: a value of `l` bits with nothing set past them (`RowShaped`) is `RowSeg.enc l` of its bytes (`rowBytes`),
  any other value a longer code; `extSeg l`, `RowSeg.extPair` of the two values' bytes, is a SHA-512 collision when two
  different values have the same `encSeg l` (`extSeg_spec`);
* `encSeg_row`: at a row of `l = 16 k` bits it is `RowSeg.enc l` of the row's `2 k` bytes;
* `Readers.rowsX`: `Readers.rows` with an encoding and an extractor per commit string.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Binding

variable {W : ℕ}

/-! ## A `sha512/row-seg/v1` row's encoding -/

/-- **The bytes `sha512/row-seg/v1`'s outer hash reads for a value**, at `l` bits: `RowSeg.enc l` of a row's bytes for a
value of `l` bits with nothing set past them, and for any other value zero bytes and `bitCode`, longer. -/
def encSeg (l : ℕ) (v : BitVal W) : List UInt8 :=
  if RowShaped v ∧ v.1.val = l then RowSeg.enc l (rowBytes v)
  else List.replicate (RowSeg.enc l []).length 0 ++ bitCode v

/-- **`encSeg`'s extractor**: `RowSeg.extPair` of the two values' bytes. -/
def extSeg (l : ℕ) (v v' : BitVal W) : List UInt8 × List UInt8 := RowSeg.extPair l (rowBytes v) (rowBytes v')

theorem length_rowSegEnc (l : ℕ) (row row' : List UInt8) : (RowSeg.enc l row).length = (RowSeg.enc l row').length := by
  simp only [RowSeg.enc, List.length_append, Hidden.length_digests]

theorem length_rowBytes (v : BitVal W) : (rowBytes v).length = 2 * (v.1.val / 16) := by
  simp only [rowBytes, List.length_ofFn]

theorem eq_of_rowBytes {v w : BitVal W} (hv : RowShaped v) (hw : RowShaped w) (hk : v.1 = w.1)
    (h : rowBytes v = rowBytes w) : v = w := by
  refine Prod.ext hk (funext fun i => ?_)
  by_cases hi : i.val < v.1.val
  · exact get_of_rowBytes hv hk h i hi
  · rw [hv.2 i (by omega), hw.2 i (by rw [← hk]; omega)]

/-- **Two different values with the same `encSeg l` give a SHA-512 collision** (`RowSeg.extPair_spec`): both are rows
of `l` bits, so their bytes differ; any other value's code is longer, and `bitCode` is injective. -/
theorem extSeg_spec (l : ℕ) (v v' : BitVal W) (hne : v ≠ v') (h : encSeg l v = encSeg l v') :
    Collides (extSeg l v v') := by
  have hl : ∀ {x : BitVal W}, RowShaped x ∧ x.1.val = l → (rowBytes x).length = (l + 7) / 8 := fun hx => by
    rw [length_rowBytes]
    have := hx.1.1
    have := hx.2
    omega
  unfold encSeg at h
  by_cases hv : RowShaped v ∧ v.1.val = l <;> by_cases hw : RowShaped v' ∧ v'.1.val = l
  · rw [if_pos hv, if_pos hw] at h
    exact RowSeg.extPair_spec (hl hv) (hl hw)
      (fun hb => hne (eq_of_rowBytes hv.1 hw.1 (Fin.ext (hv.2.trans hw.2.symm)) hb)) h
  · rw [if_pos hv, if_neg hw] at h
    have e := congrArg List.length h
    rw [List.length_append, List.length_replicate, bitCode_length, length_rowSegEnc l (rowBytes v) []] at e
    omega
  · rw [if_neg hv, if_pos hw] at h
    have e := congrArg List.length h
    rw [List.length_append, List.length_replicate, bitCode_length, length_rowSegEnc l (rowBytes v') []] at e
    omega
  · rw [if_neg hv, if_neg hw] at h
    exact absurd (bitCode_inj (List.append_inj h rfl).2) hne

/-- **A row's bytes at `16 k` bits**: byte `j`'s bit `t` its bit `8 j + t`. -/
theorem rowBytes_row {K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    rowBytes (widthOf W K, (padBits b : Fin W → Bool)) =
      List.ofFn fun j : Fin (2 * k) => byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩ := by
  have hw : (widthOf W K).val = K := by simp only [widthOf]; omega
  have hk : (widthOf W K).val / 16 = k := by rw [hw, hK]; omega
  refine List.ext_getElem (by simp [rowBytes, hk]) fun j h1 h2 => ?_
  simp only [rowBytes, List.getElem_ofFn]
  congr 1
  funext t
  have hj : 8 * j + t.val < K := by simp at h2; omega
  simp only [BitVal.get, show 8 * j + t.val < W by omega, ↓reduceDIte, padBits, hj]

/-- **`encSeg` of a row of `l = 16 k` bits**: `RowSeg.enc l` of the row's `2 k` bytes. -/
theorem encSeg_row {K k : ℕ} (hK : K = 16 * k) (hKW : K ≤ W) (b : Fin K → Bool) :
    encSeg K (widthOf W K, (padBits b : Fin W → Bool)) =
      RowSeg.enc K (List.ofFn fun j : Fin (2 * k) => byteOf fun t => b ⟨8 * j.val + t.val, by omega⟩) := by
  have hw : (widthOf W K).val = K := by simp only [widthOf]; omega
  have hs : RowShaped (widthOf W K, (padBits b : Fin W → Bool)) :=
    ⟨by rw [hw, hK]; omega, fun i hi => by
      change (widthOf W K).val ≤ i.val at hi
      rw [hw] at hi
      simp only [padBits, show ¬ i.val < K by omega, ↓reduceDIte]⟩
  unfold encSeg
  rw [if_pos ⟨hs, hw⟩, rowBytes_row hK hKW b]

/-! ## Rows each under its own encoding -/

namespace Readers

open Game Model Audit

variable {C : Audit.Circuit} {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  (rd : Readers C plan tab) {P : Audit.Partition C n}
  (place : ∀ S R u, u ∈ S → UnitPlace P (Partition.tabOf plan tab S R u).S u)

/-- **The layout's rows, each under its own encoding**: `Readers.rows` with `enc q` the bytes the row at commit string
`q` hashes and `ext q` its extractor. -/
noncomputable def rowsX (enc : rd.Pos → BitVal rd.W → List UInt8)
    (ext : rd.Pos → BitVal rd.W → BitVal rd.W → List UInt8 × List UInt8)
    (ext_spec : ∀ q v v', v ≠ v' → enc q v = enc q v' → Collides (ext q v v'))
    (encO : BitVal rd.W → List UInt8) (encO_inj : Function.Injective encO) (hcopy : rd.Copies place)
    (hbc : rd.BcPinned (P := P)) (hval : rd.ValPinned (P := P)) : Binding.HmRowsX (rd.layout place encO hcopy) where
  isRow := rd.isRow
  decRow := rd.decRow
  enc := enc
  ext := ext
  ext_spec := ext_spec
  other _ := ExplicitLeaf.plain encO encO_inj
  bcAt := rd.bcAt
  bc_registered := (rd.rows place encO encO_inj encO encO_inj hcopy hbc hval).bc_registered
  other_registered := (rd.rows place encO encO_inj encO encO_inj hcopy hbc hval).other_registered

end Readers

end FlockSoundness.Discharge.Layout
