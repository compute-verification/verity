import Proofs.Flock.VBridge.OfVStar
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.Soundness.Discharge.Hm.Salt
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import VStar.Link
import Flock.RecLinkRef

/-!
# The gadgets and the row check of V*'s `RecLink{UB, PB}` unit

`Proofs.Flock.VBridge.RecLink` proves the unit computes its reference; this file holds the parts it builds on.

* The gadgets. `rec_link._all` (`sound_allGo`, `sound_allForms`): 1 exactly when every form is 1; `_none`
  (`sound_noneForms`): 1 exactly when every form is 0 (`AllZero`); `orForm`.
* The check. `sound_okForms`: on forms that carry a 128-byte table row and two 64-byte digests (`Carries`), `okForms` is
  `okRef`: a field's 32 forms are its little-endian u32's bits (`bit_linkField`), its high bits zero exactly when it fits
  (`allZero_dropField`), the row's bytes past the fields zero exactly when `tailZero` (`tailZero_iff`), and the digests'
  XOR zero exactly when they are equal (`allZero_xor`).
* Freshness: each gadget adds rows only as `fresh` does (`fresh_okForms`).
-/

namespace FlockVBridge.RecLink

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm

/-! ## Bits of lists of forms -/

theorem bit_append_left {z : Var → Bool} {a b : List Lin} {i : ℕ} (h : i < a.length) : bit z (a ++ b) i = bit z a i := by
  simp only [bit, List.getElem?_append_left h]

theorem bit_append_right {z : Var → Bool} {a b : List Lin} {i : ℕ} (h : a.length ≤ i) :
    bit z (a ++ b) i = bit z b (i - a.length) := by
  simp only [bit, List.getElem?_append_right h]

theorem bit_drop (z : Var → Bool) (l : List Lin) (k i : ℕ) : bit z (l.drop k) i = bit z l (k + i) := by
  simp only [bit, List.getElem?_drop]

theorem bit_take {z : Var → Bool} {l : List Lin} {k i : ℕ} (h : i < k) : bit z (l.take k) i = bit z l i := by
  simp only [bit, List.getElem?_take_of_lt h]

theorem bit_ge {z : Var → Bool} {l : List Lin} {i : ℕ} (h : l.length ≤ i) : bit z l i = false := by
  simp only [bit, List.getElem?_eq_none h, Option.getD_none, ev_zero]

theorem ev_getD (z : Var → Bool) (l : List Lin) (i : ℕ) : ev z (l.getD i Lin.zero) = bit z l i := by
  rw [bit, List.getD_eq_getElem?_getD]

theorem bit_map_not {z : Var → Bool} {l : List Lin} {i : ℕ} (h : i < l.length) :
    bit z (l.map (· ^^^ Lin.one)) i = !bit z l i := by
  rw [bit_of_lt (show i < (l.map (· ^^^ Lin.one)).length by simpa using h), bit_of_lt h, List.getElem_map, ev_xor,
    ev_one, Bool.xor_true]

theorem bit_zipWith_xor (z : Var → Bool) {a b : List Lin} (hl : a.length = b.length) (i : ℕ) :
    bit z (List.zipWith (· ^^^ ·) a b) i = (bit z a i ^^ bit z b i) := by
  by_cases hi : i < a.length
  · rw [bit_of_lt (show i < (List.zipWith (· ^^^ ·) a b).length by simp [hl]; omega), List.getElem_zipWith, ev_xor,
      bit_of_lt hi, bit_of_lt (show i < b.length by omega)]
  · rw [bit_ge (show (List.zipWith (· ^^^ ·) a b).length ≤ i by simp [hl]; omega), bit_ge (show a.length ≤ i by omega),
      bit_ge (show b.length ≤ i by omega), Bool.xor_false]

/-- Forms that carry bytes read as those bytes' bits, past the forms too. -/
theorem bit_eq_bitL {z : Var → Bool} {w : List Lin} {B : ByteArray} (h : Carries z w B) : bit z w = bitL B.data.toList := by
  funext t
  by_cases ht : t < w.length
  · exact h.2 t ht
  · have hl := h.1
    rw [bit_ge (show w.length ≤ t by omega), bitL,
      List.getD_eq_default _ _ (show B.data.toList.length ≤ t / 8 by rw [Array.length_toList]; change B.size ≤ t / 8; omega)]
    simp

/-- Every form of `l` is 0. -/
def AllZero (z : Var → Bool) (l : List Lin) : Prop := ∀ i < l.length, bit z l i = false

theorem allZero_append {z : Var → Bool} {a b : List Lin} : AllZero z (a ++ b) ↔ AllZero z a ∧ AllZero z b := by
  unfold AllZero
  constructor
  · intro h
    refine ⟨fun i hi => ?_, fun i hi => ?_⟩
    · have := h i (by simp; omega)
      rwa [bit_append_left hi] at this
    · have := h (a.length + i) (by simp; omega)
      rwa [bit_append_right (show a.length ≤ a.length + i by omega), Nat.add_sub_cancel_left] at this
  · rintro ⟨ha, hb⟩ i hi
    rw [List.length_append] at hi
    by_cases h : i < a.length
    · rw [bit_append_left h]
      exact ha i h
    · rw [bit_append_right (show a.length ≤ i by omega)]
      exact hb _ (by omega)

/-! ## `rec_link._all`, `_none` and `or` -/

theorem allGo_nil (fuel : ℕ) : VStar.allGo fuel [] = pure Lin.one := by cases fuel <;> rfl

theorem allGo_single (fuel : ℕ) (a : Lin) : VStar.allGo fuel [a] = pure a := by cases fuel <;> rfl

theorem allGo_zero (a b : Lin) (rest : List Lin) : VStar.allGo 0 (a :: b :: rest) = pure a := rfl

theorem allGo_succ (fuel : ℕ) (a b : Lin) (rest : List Lin) :
    VStar.allGo (fuel + 1) (a :: b :: rest) =
      ((((a :: b :: rest).take ((a :: b :: rest).length / 2)).zip
          (((a :: b :: rest).drop ((a :: b :: rest).length / 2)).take ((a :: b :: rest).length / 2))).mapM
          (fun (p, q) => Flock.HmNets.and p q) >>= fun ands =>
        VStar.allGo fuel (ands ++ (a :: b :: rest).drop (2 * ((a :: b :: rest).length / 2)))) := by
  first
    | rfl
    | (rw [VStar.allGo.eq_def])
    | (rw [VStar.allGo.eq_def]; rfl)

/-- One round of `_all` keeps "every form is 1": the pairwise ANDs of the halves, then the odd last form. -/
theorem all_step {n m : ℕ} (hm : 2 * m ≤ n) (f g : ℕ → Bool)
    (hg : ∀ i < n - m, g i = if i < m then (f i && f (m + i)) else f (m + i)) :
    (∀ i < n - m, g i = true) ↔ ∀ j < n, f j = true := by
  constructor
  · intro H j hj
    by_cases h1 : j < m
    · have := H j (by omega)
      rw [hg j (by omega), if_pos h1, Bool.and_eq_true] at this
      exact this.1
    · by_cases h2 : j < 2 * m
      · have := H (j - m) (by omega)
        rw [hg _ (by omega), if_pos (show j - m < m by omega), Bool.and_eq_true, show m + (j - m) = j by omega] at this
        exact this.2
      · have := H (j - m) (by omega)
        rw [hg _ (by omega), if_neg (show ¬ j - m < m by omega), show m + (j - m) = j by omega] at this
        exact this
  · intro H i hi
    rw [hg i hi]
    split_ifs with h1
    · rw [H i (by omega), H (m + i) (by omega)]
      rfl
    · exact H (m + i) (by omega)

theorem sound_allRound (fuel : ℕ) (bits : List Lin) (hb : 2 ≤ bits.length) (k : List Lin → M Lin)
    (hk : ∀ L, Sound (k L) fun z r => L.length ≤ fuel + 1 → (ev z r = true ↔ ∀ i < L.length, bit z L i = true)) :
    Sound (((bits.take (bits.length / 2)).zip ((bits.drop (bits.length / 2)).take (bits.length / 2))).mapM
          (fun (p, q) => Flock.HmNets.and p q) >>= fun ands => k (ands ++ bits.drop (2 * (bits.length / 2))))
      fun z r => bits.length ≤ fuel + 2 → (ev z r = true ↔ ∀ i < bits.length, bit z bits i = true) := by
  refine Sound.bind' (sound_mapM _ (fun z x => ev z x.1 && ev z x.2)
    (fun x => by obtain ⟨p, q⟩ := x; exact sound_and p q) _) fun ands => Sound.mono (hk _) fun z r hr hands hlen => ?_
  obtain ⟨hal, hab⟩ := hands
  have hzl : ((bits.take (bits.length / 2)).zip ((bits.drop (bits.length / 2)).take (bits.length / 2))).length =
      bits.length / 2 := by
    simp only [List.length_zip, List.length_take, List.length_drop]
    omega
  have hL : (ands ++ bits.drop (2 * (bits.length / 2))).length = bits.length - bits.length / 2 := by
    rw [List.length_append, List.length_drop, hal, hzl]
    omega
  rw [hr (by rw [hL]; omega), hL]
  refine all_step (n := bits.length) (m := bits.length / 2) (by omega) (bit z bits)
    (bit z (ands ++ bits.drop (2 * (bits.length / 2)))) fun i hi => ?_
  split_ifs with h1
  · rw [bit_append_left (show i < ands.length by omega), hab i (by omega)]
    simp only [List.getElem_zip, List.getElem_take, List.getElem_drop]
    rw [bit_of_lt (show i < bits.length by omega), bit_of_lt (show bits.length / 2 + i < bits.length by omega)]
  · rw [bit_append_right (show ands.length ≤ i by omega), bit_drop, hal, hzl,
      show 2 * (bits.length / 2) + (i - bits.length / 2) = bits.length / 2 + i by omega]

/-- **`rec_link._all`'s rounds**: with fuel enough (a list's length), the result is 1 exactly when every form is 1. -/
theorem sound_allGo : ∀ (fuel : ℕ) (bits : List Lin), Sound (VStar.allGo fuel bits) fun z r =>
    bits.length ≤ fuel + 1 → (ev z r = true ↔ ∀ i < bits.length, bit z bits i = true)
  | fuel, [] => by
      rw [allGo_nil]
      exact Sound.pure' fun z _ => by simp
  | fuel, [a] => by
      rw [allGo_single]
      refine Sound.pure' fun z _ => ⟨fun h i hi => ?_, fun h => ?_⟩
      · obtain rfl : i = 0 := by simp at hi; omega
        simpa [bit] using h
      · simpa [bit] using h 0 (by simp)
  | 0, a :: b :: rest => by
      rw [allGo_zero]
      exact Sound.pure' fun z h => absurd h (by simp)
  | fuel + 1, a :: b :: rest => by
      rw [allGo_succ]
      exact sound_allRound fuel (a :: b :: rest) (by simp) _ fun L => sound_allGo fuel L

theorem sound_allForms (bits : List Lin) : Sound (VStar.allForms bits) fun z r =>
    (ev z r = true ↔ ∀ i < bits.length, bit z bits i = true) :=
  Sound.mono (sound_allGo bits.length bits) fun _ _ h => h (Nat.le_succ _)

theorem sound_noneForms (bits : List Lin) : Sound (VStar.noneForms bits) fun z r => (ev z r = true ↔ AllZero z bits) := by
  unfold VStar.noneForms
  refine Sound.ite' (fun h => Sound.pure' fun z => ?_) fun _ => Sound.mono (sound_allForms _) fun z r h => ?_
  · obtain rfl := List.isEmpty_iff.1 h
    simp [AllZero]
  · rw [h, List.length_map]
    exact forall₂_congr fun i hi => by rw [bit_map_not hi]; simp

theorem sound_orForm (a b : Lin) : Sound (VStar.orForm a b) fun z r => ev z r = (ev z a || ev z b) := by
  unfold VStar.orForm
  exact Sound.bind' (sound_and _ _) fun r => Sound.pure' fun z hr => by
    rw [ev_xor, hr, ev_xor, ev_xor, ev_one]
    cases ev z a <;> cases ev z b <;> rfl

theorem sound_all5 (a b c d e : Lin) : Sound (VStar.allForms [a, b, c, d, e]) fun z r =>
    ev z r = (ev z a && ev z b && ev z c && ev z d && ev z e) :=
  Sound.mono (sound_allForms _) fun z r h => by
    rw [Bool.eq_iff_iff, h]
    simp only [Bool.and_eq_true]
    constructor
    · intro H
      exact ⟨⟨⟨⟨H 0 (by simp), H 1 (by simp)⟩, H 2 (by simp)⟩, H 3 (by simp)⟩, H 4 (by simp)⟩
    · rintro ⟨⟨⟨⟨h0, h1⟩, h2⟩, h3⟩, h4⟩ i hi
      simp only [List.length_cons, List.length_nil] at hi
      interval_cases i <;> simpa [bit]

/-! ## The table row's fields -/

theorem u32At_lt (T : ByteArray) (off : ℕ) : RecLinks.u32At T off < 2 ^ 32 := by
  unfold RecLinks.u32At
  have h0 : (T.get! off).toNat < 256 := UInt8.toNat_lt _
  have h1 : (T.get! (off + 1)).toNat < 256 := UInt8.toNat_lt _
  have h2 : (T.get! (off + 2)).toNat < 256 := UInt8.toNat_lt _
  have h3 : (T.get! (off + 3)).toNat < 256 := UInt8.toNat_lt _
  rw [show (2 : ℕ) ^ 32 = 4294967296 from rfl]
  omega

theorem field_lt (T : ByteArray) (k : ℕ) : RecLinkRef.field T k < 2 ^ 32 := u32At_lt T (4 * k)

/-- Bit `j < 32` of a little-endian u32 is bit `j % 8` of its byte `j / 8`. -/
theorem testBit_u32At (T : ByteArray) (off : ℕ) {j : ℕ} (hj : j < 32) :
    (RecLinks.u32At T off).testBit j = (T.get! (off + j / 8)).toNat.testBit (j % 8) := by
  have e : RecLinks.u32At T off = 2 ^ 8 * (2 ^ 8 * (2 ^ 8 * (T.get! (off + 3)).toNat + (T.get! (off + 2)).toNat) +
      (T.get! (off + 1)).toNat) + (T.get! off).toNat := by
    unfold RecLinks.u32At
    ring
  have hb : ∀ x : UInt8, x.toNat < 2 ^ 8 := fun x => UInt8.toNat_lt x
  rw [e, Nat.testBit_two_pow_mul_add _ (hb _)]
  split_ifs with h1
  · rw [show j / 8 = 0 by omega, show j % 8 = j by omega, Nat.add_zero]
  · rw [Nat.testBit_two_pow_mul_add _ (hb _)]
    split_ifs with h2
    · rw [show j / 8 = 1 by omega, show j % 8 = j - 8 by omega]
    · rw [Nat.testBit_two_pow_mul_add _ (hb _)]
      split_ifs with h3
      · rw [show j / 8 = 2 by omega, show j % 8 = j - 8 - 8 by omega]
      · rw [show j / 8 = 3 by omega, show j % 8 = j - 8 - 8 - 8 by omega]

theorem length_linkField {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    (k : ℕ) (hk : k < 32) : (VStar.linkField row k).length = 32 := by
  rw [VStar.linkField, List.length_take, List.length_drop, hc.1, hT, show VStar.LINK_FIELD_BITS = 32 from rfl]
  omega

/-- **A field's forms are its bits**: on forms that carry a 128-byte row, bit `j` of field `k`'s forms is bit `j` of the
row's `k`-th little-endian u32. -/
theorem bit_linkField {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    (k j : ℕ) (hk : k < 32) (hj : j < 32) : bit z (VStar.linkField row k) j = (RecLinkRef.field T k).testBit j := by
  have hl : 32 * k + j < row.length := by rw [hc.1, hT]; omega
  rw [VStar.linkField, show VStar.LINK_FIELD_BITS = 32 from rfl, bit_take hj, bit_drop, hc.2 _ hl, RecLinkRef.field,
    testBit_u32At _ _ hj, get!_eq_getD, bitL, show (32 * k + j) / 8 = 4 * k + j / 8 by omega,
    show (32 * k + j) % 8 = j % 8 by omega]

theorem lt_two_pow_iff {f n : ℕ} (hf : f < 2 ^ 32) (hn : n ≤ 32) :
    (∀ i < 32 - n, f.testBit (n + i) = false) ↔ f < 2 ^ n := by
  constructor
  · intro h
    apply Nat.lt_pow_two_of_testBit
    intro i hi
    by_cases h32 : i < 32
    · have := h (i - n) (by omega)
      rwa [show n + (i - n) = i by omega] at this
    · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le hf (Nat.pow_le_pow_right (by norm_num) (by omega)))
  · intro h i _
    exact Nat.testBit_lt_two_pow (lt_of_lt_of_le h (Nat.pow_le_pow_right (by norm_num) (by omega)))

theorem eq_zero_iff_low {f n : ℕ} (hf : f < 2 ^ n) : (∀ i < n, f.testBit i = false) ↔ f = 0 := by
  constructor
  · intro h
    apply Nat.eq_of_testBit_eq
    intro i
    rw [Nat.zero_testBit]
    by_cases hi : i < n
    · exact h i hi
    · exact Nat.testBit_lt_two_pow (lt_of_lt_of_le hf (Nat.pow_le_pow_right (by norm_num) (by omega)))
  · rintro rfl i _
    exact Nat.zero_testBit i

/-- A field's forms past its low `U` are 0 exactly when it fits in `U` bits. -/
theorem allZero_dropField {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    (k U : ℕ) (hk : k < 32) (hU : U ≤ 32) :
    AllZero z ((VStar.linkField row k).drop U) ↔ RecLinkRef.field T k < 2 ^ U := by
  rw [← lt_two_pow_iff (field_lt T k) hU]
  unfold AllZero
  rw [List.length_drop, length_linkField hc hT k hk]
  exact forall₂_congr fun i hi => by rw [bit_drop, bit_linkField hc hT k (U + i) hk (by omega)]

/-- A field that fits in `U` bits has its low `U` forms 0 exactly when it is 0. -/
theorem allZero_takeField {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    (k U : ℕ) (hk : k < 32) (hU : U ≤ 32) (hf : RecLinkRef.field T k < 2 ^ U) :
    AllZero z ((VStar.linkField row k).take U) ↔ RecLinkRef.field T k = 0 := by
  rw [← eq_zero_iff_low hf]
  unfold AllZero
  rw [List.length_take, length_linkField hc hT k hk, Nat.min_eq_left hU]
  exact forall₂_congr fun i hi => by rw [bit_take hi, bit_linkField hc hT k i hk (by omega)]

theorem byte_eq_zero_iff (x : UInt8) : x = 0 ↔ ∀ b < 8, x.toNat.testBit b = false := by
  constructor
  · rintro rfl b _
    simp
  · intro h
    apply UInt8.toNat_inj.1
    apply Nat.eq_of_testBit_eq
    intro b
    by_cases hb : b < 8
    · rw [h b hb]
      simp
    · rw [Nat.testBit_lt_two_pow (lt_of_lt_of_le (UInt8.toNat_lt x) (Nat.pow_le_pow_right (by norm_num) (by omega)))]
      simp

/-- The row's forms past its five fields are 0 exactly when `tailZero`. -/
theorem tailZero_iff {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128) :
    AllZero z (row.drop (8 * RecLinks.FIELD_BYTES)) ↔ RecLinkRef.tailZero T = true := by
  unfold AllZero RecLinkRef.tailZero
  rw [List.all_eq_true, List.length_drop, hc.1, hT, show RecLinks.FIELD_BYTES = 20 from rfl]
  simp only [List.mem_range, beq_iff_eq]
  constructor
  · intro h i hi
    rw [byte_eq_zero_iff]
    intro b hb
    have := h (8 * i + b) (by omega)
    rw [bit_drop, hc.2 _ (show 8 * 20 + (8 * i + b) < row.length by rw [hc.1, hT]; omega), bitL,
      show (8 * 20 + (8 * i + b)) / 8 = 20 + i by omega, show (8 * 20 + (8 * i + b)) % 8 = b by omega,
      ← get!_eq_getD] at this
    exact this
  · intro h t ht
    rw [bit_drop, hc.2 _ (show 8 * 20 + t < row.length by rw [hc.1, hT]; omega), bitL,
      show (8 * 20 + t) / 8 = 20 + t / 8 by omega, ← get!_eq_getD, h (t / 8) (by omega)]
    simp

/-- Two digests' forms XOR to 0 exactly when the digests are equal. -/
theorem allZero_xor {z : Var → Bool} {a b : List Lin} {A B : ByteArray} (ha : Carries z a A) (hb : Carries z b B)
    (hAB : A.size = B.size) : AllZero z (List.zipWith (· ^^^ ·) a b) ↔ A.data = B.data := by
  have hl : a.length = b.length := by rw [ha.1, hb.1, hAB]
  unfold AllZero
  rw [List.length_zipWith, ← hl, Nat.min_self, ← Array.toList_inj]
  constructor
  · intro h
    refine eq_of_bitL (by rw [Array.length_toList, Array.length_toList]; exact hAB) fun t ht => ?_
    have ht' : t < a.length := by rw [ha.1]; rw [Array.length_toList] at ht; exact ht
    have := h t ht'
    rw [bit_zipWith_xor z hl, ha.2 t ht', hb.2 t (by omega)] at this
    revert this
    cases bitL A.data.toList t <;> cases bitL B.data.toList t <;> decide
  · intro h t ht
    rw [bit_zipWith_xor z hl, ha.2 t ht, hb.2 t (by omega), h]
    simp

/-- `okRef` as the Boolean combination `okForms` computes: the kind's bits (`rz`: bits 1 to 31 zero, `b0`: bit 0), the
tail, the fields' fit and zero tests, and the digests' equality. -/
theorem okRef_eq {UB PB : ℕ} {T Xr Xw : ByteArray} (rz b0 tl ft fz eq : Bool)
    (hF : (rz && !b0) = decide (RecLinkRef.field T 0 = 0)) (hL : (rz && b0) = decide (RecLinkRef.field T 0 = 1))
    (htl : tl = RecLinkRef.tailZero T)
    (hft : ft = decide (RecLinkRef.field T 1 < 2 ^ UB ∧ RecLinkRef.field T 3 < 2 ^ UB ∧ RecLinkRef.field T 2 < 2 ^ PB ∧
      RecLinkRef.field T 4 < 2 ^ PB))
    (hfz : ft = true → (fz = true ↔ RecLinkRef.field T 1 = 0 ∧ RecLinkRef.field T 3 = 0 ∧ RecLinkRef.field T 2 = 0 ∧
      RecLinkRef.field T 4 = 0))
    (heq : eq = (Xr.data == Xw.data)) :
    ((rz && !b0 || rz && b0) && tl && ft && (!(rz && !b0) || fz) && (!(rz && b0) || eq)) =
      RecLinkRef.okRef UB PB T Xr Xw := by
  have hft' : ft = true ↔ (RecLinkRef.field T 1 < 2 ^ UB ∧ RecLinkRef.field T 3 < 2 ^ UB ∧
      RecLinkRef.field T 2 < 2 ^ PB ∧ RecLinkRef.field T 4 < 2 ^ PB) := by
    rw [hft, decide_eq_true_eq]
  rw [hF, hL, htl, heq, Bool.eq_iff_iff]
  unfold RecLinkRef.okRef RecLinkRef.okShape
  by_cases hk : RecLinkRef.field T 0 = 0
  · simp [hk, RecLinks.KIND_LINK]
    tauto
  · simp [hk, RecLinks.KIND_LINK]
    tauto

/-- **`rec_link._ok` is `okRef`**: on forms that carry a 128-byte table row and two 64-byte digests, whenever its rows
hold. -/
theorem sound_okForms (UB PB : ℕ) (hUB : UB ≤ 32) (hPB : PB ≤ 32) (row xr xw : List Lin) :
    Sound (VStar.okForms UB PB row xr xw) fun z ok => ∀ T Xr Xw : ByteArray, T.size = 128 → Xr.size = 64 →
      Xw.size = 64 → Carries z row T → Carries z xr Xr → Carries z xw Xw →
      ev z ok = RecLinkRef.okRef UB PB T Xr Xw := by
  unfold VStar.okForms
  dsimp only
  refine Sound.bind' (sound_noneForms _) fun rz => Sound.bind' (sound_and _ _) fun isF =>
    Sound.bind' (sound_and _ _) fun isL => Sound.bind' (sound_noneForms _) fun tl =>
    Sound.bind' (sound_noneForms _) fun ft => Sound.bind' (sound_noneForms _) fun fz =>
    Sound.bind' (sound_noneForms _) fun eq => Sound.bind' (sound_orForm _ _) fun kinds =>
    Sound.bind' (sound_orForm _ _) fun fo => Sound.bind' (sound_orForm _ _) fun lo =>
    Sound.mono (sound_all5 _ _ _ _ _) fun z r hr hlo hfo hkinds heq hfz hft htl hisL hisF hrz T Xr Xw hT hXr hXw hrow
      hxr hxw => ?_
  have hrz' : ev z rz = decide (RecLinkRef.field T 0 < 2) := by
    rw [Bool.eq_iff_iff, hrz, decide_eq_true_eq, allZero_dropField hrow hT 0 1 (by norm_num) (by norm_num), pow_one]
  have hb0 : ev z ((VStar.linkField row 0).getD 0 Lin.zero) = decide (RecLinkRef.field T 0 % 2 = 1) := by
    rw [ev_getD, bit_linkField hrow hT 0 0 (by norm_num) (by norm_num), Nat.testBit_zero]
  have hF : (ev z rz && !ev z ((VStar.linkField row 0).getD 0 Lin.zero)) = decide (RecLinkRef.field T 0 = 0) := by
    rw [hrz', hb0, Bool.eq_iff_iff]
    simp only [Bool.and_eq_true, Bool.not_eq_true', decide_eq_true_eq, decide_eq_false_iff_not]
    omega
  have hL : (ev z rz && ev z ((VStar.linkField row 0).getD 0 Lin.zero)) = decide (RecLinkRef.field T 0 = 1) := by
    rw [hrz', hb0, Bool.eq_iff_iff]
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    omega
  have htl' : ev z tl = RecLinkRef.tailZero T := by
    rw [Bool.eq_iff_iff, htl, tailZero_iff hrow hT]
  have hft' : ev z ft = decide (RecLinkRef.field T 1 < 2 ^ UB ∧ RecLinkRef.field T 3 < 2 ^ UB ∧
      RecLinkRef.field T 2 < 2 ^ PB ∧ RecLinkRef.field T 4 < 2 ^ PB) := by
    rw [Bool.eq_iff_iff, hft, decide_eq_true_eq, allZero_append, allZero_append, allZero_append,
      allZero_dropField hrow hT 1 UB (by norm_num) hUB, allZero_dropField hrow hT 3 UB (by norm_num) hUB,
      allZero_dropField hrow hT 2 PB (by norm_num) hPB, allZero_dropField hrow hT 4 PB (by norm_num) hPB]
    simp only [and_assoc]
  have hfz' : ev z ft = true → (ev z fz = true ↔ RecLinkRef.field T 1 = 0 ∧ RecLinkRef.field T 3 = 0 ∧
      RecLinkRef.field T 2 = 0 ∧ RecLinkRef.field T 4 = 0) := by
    intro h
    rw [hft', decide_eq_true_eq] at h
    obtain ⟨h1, h3, h2, h4⟩ := h
    rw [hfz, allZero_append, allZero_append, allZero_append, allZero_takeField hrow hT 1 UB (by norm_num) hUB h1,
      allZero_takeField hrow hT 3 UB (by norm_num) hUB h3, allZero_takeField hrow hT 2 PB (by norm_num) hPB h2,
      allZero_takeField hrow hT 4 PB (by norm_num) hPB h4]
    simp only [and_assoc]
  have heq' : ev z eq = (Xr.data == Xw.data) := by
    rw [Bool.eq_iff_iff, heq, allZero_xor hxr hxw (by rw [hXr, hXw]), beq_iff_eq]
  refine Eq.trans ?_ (okRef_eq (ev z rz) (ev z ((VStar.linkField row 0).getD 0 Lin.zero)) (ev z tl) (ev z ft)
    (ev z fz) (ev z eq) hF hL htl' hft' hfz' heq')
  rw [hr, hkinds, hfo, hlo, hisF, hisL]
  simp only [ev_xor, ev_one, Bool.xor_true, hisF, hisL]

/-! ## Freshness -/

theorem fresh_allGo : ∀ (fuel : ℕ) (bits : List Lin), Fresh (VStar.allGo fuel bits)
  | fuel, [] => by rw [allGo_nil]; exact Fresh.pure' _
  | fuel, [a] => by rw [allGo_single]; exact Fresh.pure' _
  | 0, a :: b :: rest => by rw [allGo_zero]; exact Fresh.pure' _
  | fuel + 1, a :: b :: rest => by
      rw [allGo_succ]
      exact Fresh.bind' (Fresh.mapM' _ (fun x => by obtain ⟨p, q⟩ := x; exact fresh_and p q) _) fun _ =>
        fresh_allGo fuel _

theorem fresh_allForms (bits : List Lin) : Fresh (VStar.allForms bits) := fresh_allGo _ _

theorem fresh_noneForms (bits : List Lin) : Fresh (VStar.noneForms bits) := by
  unfold VStar.noneForms
  exact Fresh.ite' (Fresh.pure' _) (fresh_allForms _)

theorem fresh_orForm (a b : Lin) : Fresh (VStar.orForm a b) := by
  unfold VStar.orForm
  exact Fresh.bind' (fresh_and _ _) fun _ => Fresh.pure' _

theorem fresh_okForms (UB PB : ℕ) (row xr xw : List Lin) : Fresh (VStar.okForms UB PB row xr xw) := by
  unfold VStar.okForms
  dsimp only
  exact Fresh.bind' (fresh_noneForms _) fun _ => Fresh.bind' (fresh_and _ _) fun _ =>
    Fresh.bind' (fresh_and _ _) fun _ => Fresh.bind' (fresh_noneForms _) fun _ =>
    Fresh.bind' (fresh_noneForms _) fun _ => Fresh.bind' (fresh_noneForms _) fun _ =>
    Fresh.bind' (fresh_noneForms _) fun _ => Fresh.bind' (fresh_orForm _ _) fun _ =>
    Fresh.bind' (fresh_orForm _ _) fun _ => Fresh.bind' (fresh_orForm _ _) fun _ => fresh_allForms _

end FlockVBridge.RecLink
