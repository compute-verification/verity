import Proofs.Flock.VBridge.OfVStar
import Proofs.Flock.VBridge.AccUnits
import Proofs.Flock.Soundness.Discharge.Hm.Salt
import Proofs.Flock.Soundness.Discharge.Hm.Commit
import VStar.Link
import Flock.RecLinkRef

/-!
# V*'s `RecLink{UB, PB}` unit computes its reference

`VStar.linkUnit UB PB` is the netlist V* writes for one row of a recursive session's link table. This file proves that,
wherever its rows hold, its outputs are the bytes `Flock.RecLinkRef` says: each opened row's hm96 tree leaf
(`RecLinkTree.leaf` of `Hm96.Default512.commitString`) climbed by its siblings (`Flock.climb` under `RecLinkRef.scheme`) to
the position the table row names (`RecLinkRef.posOf`), then 64 zero bytes, when the row passes `RecLinkRef.okRef`, and
128 zero bytes when it doesn't (`rootB`, `linkB`).

* The gadgets. `rec_link._all` (`sound_allGo`, `sound_allForms`): 1 exactly when every form is 1; `_none`
  (`sound_noneForms`): 1 exactly when every form is 0 (`AllZero`); `orForm`.
* The check. `sound_okForms`: on forms that carry a 128-byte table row and two 64-byte digests (`Carries`), `okForms` is
  `okRef`: a field's 32 forms are its little-endian u32's bits (`bit_linkField`), its high bits zero exactly when it fits
  (`allZero_dropField`), the row's bytes past the fields zero exactly when `tailZero` (`tailZero_iff`), and the digests'
  XOR zero exactly when they are equal (`allZero_xor`).
* The leaf. `sound_hm96Leaf`: the commit string `b ‖ c` (`b = x ⊕ M·y` word by word, `c` the salt's SHA-512 from its
  prefix's midstate) and its SHA-512 from the leaf prefix's midstate are `RecLinkTree.leaf (commitString X Y)`.
* The steps. `sound_linkSteps`: the climbs' direction bits are the bits of `dirIdx`, which is `posOf` wherever the row
  passes `okRef` (`dirIdx_eq`), and each root is ANDed with `ok` (`carries_gate`).
* The unit. `linkUnit_ok` (its builder is `linkRun`, the steps on the seven ports' input bits), `laid_linkUnit` (the
  laid-out unit's output columns hold `linkB` wherever its rows and input columns hold) and `unit_linkUnit` (a copy of it
  in a unit of any partition).
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
  simp only [ev_xor, ev_one, Bool.xor_true]

/-! ## The hm96 leaf -/

theorem wordBit_wordBit {t : ℕ} (ht : t < 512) : wordBit (wordBit t) = t := by
  unfold wordBit
  omega

theorem size_leafPrefix : Hm96.Default512.leafPrefix.size = 128 := by
  rw [Hm96.Default512.leafPrefix, Hm96.leafPrefix, Hm96.pad, ByteArray.size_append, ByteArray.size_append,
    show Hm96.sha512.hash = Sha512.hash from rfl, Sha512.hash_size]
  have ht : Hm96.sha512.leafTag.size = 27 := by decide +kernel
  rw [ht]
  rfl

theorem preCv_one (p : List UInt8) :
    preCv p 1 = (List.range 8).map fun i => (Sha512.compress Sha512.iv ⟨p.toArray⟩ 0)[i]!.toBitVec := by
  rw [preCv, chainV, chain_succ] <;> rfl

theorem saltMid_eq : saltMid = preCv Hm96.Default512.saltPrefix.data.toList 1 := by
  rw [preCv_one]
  rfl

theorem midstate_eq {p : ByteArray} (hp : p.size = 128) : VStar.midstate p = preCv p.data.toList 1 := by
  rw [preCv_one, map_range_eq_toBV _ (size_compress _ _ _)]
  unfold VStar.midstate toBV
  rw [hp]
  rfl

theorem catBytes_eq (a b : ByteArray) : catBytes a.data.toList b.data.toList = a ++ b :=
  ByteArray.ext (Array.toList_inj.1 (by rw [data_catBytes, ByteArray.toList_data_append]))

/-- `commitString`'s first 64 bytes, `x ⊕ M·y`. -/
def xorPart (X Y : ByteArray) : ByteArray :=
  ⟨(Array.range 64).map fun i => X.get! i ^^^ (Hm96.Default512.mask Y).get! i⟩

theorem commitString_eq (X Y : ByteArray) :
    Hm96.Default512.commitString X Y = xorPart X Y ++ Sha512.hash (Hm96.Default512.saltPrefix ++ Y) := rfl

theorem size_xorPart (X Y : ByteArray) : (xorPart X Y).size = 64 := by
  show ((Array.range 64).map fun i => X.get! i ^^^ (Hm96.Default512.mask Y).get! i).size = 64
  rw [Array.size_map, Array.size_range]

theorem size_commitString (X Y : ByteArray) : (Hm96.Default512.commitString X Y).size = 128 := by
  rw [commitString_eq, ByteArray.size_append, size_xorPart, Sha512.hash_size]

theorem bitL_xorPart (X Y : ByteArray) {q : ℕ} (hq : q < 512) :
    bitL (xorPart X Y).data.toList q = (bitL X.data.toList q ^^ maskV Hm96.Default512.key (bitL Y.data.toList) q) := by
  have hm : bitL (Hm96.Default512.mask Y).data.toList q = maskV Hm96.Default512.key (bitL Y.data.toList) q :=
    bitL_mask Y.data.toList hq
  have hb : (xorPart X Y).data.toList.getD (q / 8) 0 = X.get! (q / 8) ^^^ (Hm96.Default512.mask Y).get! (q / 8) := by
    show (((Array.range 64).map fun i => X.get! i ^^^ (Hm96.Default512.mask Y).get! i).toList).getD (q / 8) 0 = _
    rw [Array.toList_map, Array.toList_range, List.getD_eq_getElem _ _ (by simp; omega), List.getElem_map,
      List.getElem_range]
  rw [bitL, hb, UInt8.toNat_xor, Nat.testBit_xor, ← hm, bitL, bitL, get!_eq_getD, get!_eq_getD]

theorem carries_xorPart {z : Var → Bool} {bw x y : List Lin} {X Y : ByteArray} (hx : Carries z x X)
    (hy : Carries z y Y)
    (hbw : bw.length = 512 ∧ ∀ i (h : i < (List.range 512).length), bit z bw i =
      (ev z (x.getD (wordBit (List.range 512)[i]) Lin.zero) ^^
        maskV Hm96.Default512.key (bit z y) (wordBit (List.range 512)[i]))) :
    Carries z ((List.range 512).map fun t => bw.toArray.getD (wordBit t) Lin.zero) (xorPart X Y) := by
  refine ⟨by rw [List.length_map, List.length_range, size_xorPart], fun q hq => ?_⟩
  have hq' : q < 512 := by simpa using hq
  rw [bit_of_lt hq, List.getElem_map, List.getElem_range, OfVStar.getD_toArray, ev_getD,
    hbw.2 (wordBit q) (by simpa using wordBit_lt hq'), List.getElem_range, wordBit_wordBit hq', ev_getD,
    bit_eq_bitL hx, bit_eq_bitL hy, bitL_xorPart X Y hq']

/-- **V*'s hm96 leaf gadget is the tree's leaf**: whenever its rows hold, on forms that carry a 64-byte digest `X` and a
192-byte salt `Y`, its output carries `RecLinkTree.leaf (commitString X Y)`. -/
theorem sound_hm96Leaf (x y : List Lin) : Sound (VStar.hm96Leaf x y) fun z out => ∀ X Y : ByteArray,
    X.size = 64 → Y.size = 192 → Carries z x X → Carries z y Y →
    Carries z out (RecLinkTree.leaf (Hm96.Default512.commitString X Y)) := by
  unfold VStar.hm96Leaf
  simp only [OfVStar.shaFrom_eq]
  rw [show Hm96.Default512.saltPrefix.size = 128 * 1 from size_saltPrefix,
    show Hm96.Default512.leafPrefix.size = 128 * 1 from size_leafPrefix]
  refine Sound.bind' (sound_mapM _ (fun z p => ev z (x.getD (wordBit p) Lin.zero) ^^
      maskV Hm96.Default512.key (bit z y) (wordBit p))
    (fun p => Sound.mono (sound_commit _) fun z r hr => by rw [hr, eval_maskBit, OfVStar.getD_toArray]) _) fun bw => ?_
  refine Sound.bind' (sound_shaFrom saltMid y 1) fun c => ?_
  refine Sound.mono (sound_shaFrom (VStar.midstate Hm96.Default512.leafPrefix) _ 1)
    fun z out hout hc hbw X Y hX hY hx hy => ?_
  obtain ⟨hcl, hcb⟩ := hc Hm96.Default512.saltPrefix.data.toList Y.data.toList (by rw [length_saltPrefix])
    (by rw [Array.length_toList]; exact hy.1) hy.2 saltMid_eq
  rw [catBytes_eq] at hcb
  have hcc : Carries z c (Sha512.hash (Hm96.Default512.saltPrefix ++ Y)) :=
    ⟨by rw [hcl, Sha512.hash_size], fun t ht => hcb t (by omega)⟩
  have hbc := carries_append (carries_xorPart hx hy hbw) hcc
  rw [← commitString_eq] at hbc
  obtain ⟨hol, hob⟩ := hout Hm96.Default512.leafPrefix.data.toList (Hm96.Default512.commitString X Y).data.toList
    (by rw [Array.length_toList]; exact size_leafPrefix) (by rw [Array.length_toList]; exact hbc.1) hbc.2
    (midstate_eq size_leafPrefix)
  rw [catBytes_eq] at hob
  exact ⟨by rw [hol]; exact (congrArg (8 * ·) (Sha512.hash_size _)).symm, fun t ht => hob t (by omega)⟩

/-! ## The steps -/

def zeroB (n : ℕ) : ByteArray := ⟨Array.replicate n 0⟩

theorem size_zeroB (n : ℕ) : (zeroB n).size = n := by
  show (Array.replicate n (0 : UInt8)).size = n
  exact Array.size_replicate

theorem bitL_zeroB (n t : ℕ) : bitL (zeroB n).data.toList t = false := by
  have h : (zeroB n).data.toList.getD (t / 8) 0 = 0 := by
    show (Array.replicate n (0 : UInt8)).toList.getD (t / 8) 0 = 0
    rw [Array.toList_replicate, List.getD_eq_getElem?_getD, List.getElem?_replicate]
    split_ifs <;> rfl
  rw [bitL, h]
  simp

theorem carries_zeros (z : Var → Bool) (n : ℕ) : Carries z (List.replicate (8 * n) Lin.zero) (zeroB n) :=
  ⟨by rw [List.length_replicate, size_zeroB], fun t ht => by
    rw [bit_of_lt ht, List.getElem_replicate, ev_zero, bitL_zeroB]⟩

/-- A climb's root and 64 zero bytes when `ok`, else 128 zero bytes: `rec_link.expected` of the root, or `zeros`. -/
def rootB (ok : Bool) (r : ByteArray) : ByteArray := (if ok then r else zeroB 64) ++ zeroB 64

theorem size_rootB (ok : Bool) {r : ByteArray} (hr : r.size = 64) : (rootB ok r).size = 128 := by
  unfold rootB
  split <;> rw [ByteArray.size_append, size_zeroB] <;> simp [hr, size_zeroB]

theorem rootB_eq {ok : Bool} {r r' : ByteArray} (h : ok = true → r = r') :
    rootB ok r = (if ok = true then r' else zeroB 64) ++ zeroB 64 := by
  unfold rootB
  by_cases hk : ok = true
  · rw [if_pos hk, if_pos hk, h hk]
  · rw [if_neg hk, if_neg hk]

/-- The unit's outputs `(rin, rout)`: the reader's and the writer's climbs to the positions the row names, each through
`rootB` at `okRef`. -/
def linkB (UB PB : ℕ) (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray) : ByteArray × ByteArray :=
  (rootB (RecLinkRef.okRef UB PB T Xr Xw)
      (Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xr Yr))
        (RecLinkRef.posOf T 1 2 PB (UB + PB)) sr),
    rootB (RecLinkRef.okRef UB PB T Xr Xw)
      (Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xw Yw))
        (RecLinkRef.posOf T 3 4 PB (UB + PB)) sw))

/-- The climb's direction bits as a number: the row field `r`'s low `PB` bits, the unit field `u`'s low `UB` above. -/
def dirIdx (T : ByteArray) (u r UB PB : ℕ) : ℕ :=
  2 ^ PB * (RecLinkRef.field T u % 2 ^ UB) + RecLinkRef.field T r % 2 ^ PB

theorem dirIdx_eq {T : ByteArray} {u r UB PB : ℕ} (hu : RecLinkRef.field T u < 2 ^ UB)
    (hr : RecLinkRef.field T r < 2 ^ PB) : dirIdx T u r UB PB = RecLinkRef.posOf T u r PB (UB + PB) := by
  rw [dirIdx, Nat.mod_eq_of_lt hu, Nat.mod_eq_of_lt hr, RecLinkRef.posOf_fits hu hr, RecLinkTree.index, Nat.mul_comm]

/-- **The climb's direction forms are `dirIdx`'s bits.** -/
theorem dirs_ok {z : Var → Bool} {row : List Lin} {T : ByteArray} (hc : Carries z row T) (hT : T.size = 128)
    {UB PB : ℕ} (hUB : UB ≤ 32) (hPB : PB ≤ 32) (u r : ℕ) (hu : u < 32) (hr : r < 32) (j : ℕ) (hj : j < UB + PB) :
    ev z (((VStar.linkField row r).take PB ++ (VStar.linkField row u).take UB).getD (0 + j) Lin.zero) =
      (dirIdx T u r UB PB).testBit j := by
  have hl : ((VStar.linkField row r).take PB).length = PB := by
    rw [List.length_take, length_linkField hc hT r hr]
    omega
  rw [Nat.zero_add, ev_getD, dirIdx, Nat.testBit_two_pow_mul_add _ (Nat.mod_lt _ (Nat.two_pow_pos PB))]
  split_ifs with h
  · rw [bit_append_left (show j < ((VStar.linkField row r).take PB).length by omega), bit_take h,
      bit_linkField hc hT r j hr (by omega), Nat.testBit_mod_two_pow, decide_eq_true h, Bool.true_and]
  · rw [bit_append_right (show ((VStar.linkField row r).take PB).length ≤ j by omega), hl,
      bit_take (show j - PB < UB by omega), bit_linkField hc hT u (j - PB) hu (by omega), Nat.testBit_mod_two_pow,
      decide_eq_true (show j - PB < UB by omega), Bool.true_and]

/-- A digest's forms ANDed with `ok` carry the digest when `ok` holds and 64 zero bytes when it doesn't. -/
theorem carries_gate {z : Var → Bool} {ok : Lin} {cr rin : List Lin} {C : ByteArray} (hC : Carries z cr C)
    (hCs : C.size = 64)
    (hr : rin.length = cr.length ∧ ∀ i (h : i < cr.length), bit z rin i = (ev z ok && ev z cr[i])) :
    Carries z rin (if ev z ok then C else zeroB 64) := by
  refine ⟨?_, fun t ht => ?_⟩
  · rw [hr.1, hC.1]
    split_ifs
    · rfl
    · rw [size_zeroB, hCs]
  · have ht' : t < cr.length := by rw [← hr.1]; exact ht
    rw [hr.2 t ht', ← bit_of_lt ht', hC.2 t ht']
    split_ifs with h
    · simp [h]
    · simp [h, bitL_zeroB]

theorem carries_linkZeros (z : Var → Bool) :
    Carries z (List.replicate (16 * (VStar.LINK_OUT_WORDS - VStar.NODE_WORDS)) Lin.zero) (zeroB 64) :=
  carries_zeros z 64

/-- **V*'s link steps compute the unit's reference outputs**: whenever their rows hold, on forms that carry a 128-byte
table row `T`, the reader's digest and salt `Xr, Yr` and siblings `sr`, and the writer's `Xw, Yw` and `sw`, the outputs
carry `linkB`: each opened row's leaf climbed to the position the row names, gated by `okRef`, then 64 zero bytes. -/
theorem sound_linkSteps (UB PB : ℕ) (hUB : UB ≤ 32) (hPB : PB ≤ 32) (trow xr yr raux xw yw waux : List Lin) :
    Sound (VStar.linkSteps UB PB trow xr yr raux xw yw waux) fun z o =>
      ∀ (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray),
        T.size = 128 → Xr.size = 64 → Yr.size = 192 → Xw.size = 64 → Yw.size = 192 →
        Carries z trow T → Carries z xr Xr → Carries z yr Yr → Carries z xw Xw → Carries z yw Yw →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms raux (UB + PB)) sr →
        List.Forall₂ (fun w b => b.size = 64 ∧ Carries z w b) (VStar.sibForms waux (UB + PB)) sw →
        Carries z o.1 (linkB UB PB T Xr Yr Xw Yw sr sw).1 ∧ Carries z o.2 (linkB UB PB T Xr Yr Xw Yw sr sw).2 := by
  unfold VStar.linkSteps
  simp only [OfVStar.climbFrom_eq]
  refine Sound.bind' (sound_okForms UB PB hUB hPB trow xr xw) fun ok =>
    Sound.bind' (sound_hm96Leaf xr yr) fun lr =>
    Sound.bind' (sound_climbFrom RecLinkRef.scheme (fun _ _ => rfl) _ 0 lr _) fun cr =>
    Sound.bind' (sound_mapM _ (fun z x => ev z ok && ev z x) (fun x => sound_and ok x) cr) fun rin =>
    Sound.bind' (sound_hm96Leaf xw yw) fun lw =>
    Sound.bind' (sound_climbFrom RecLinkRef.scheme (fun _ _ => rfl) _ 0 lw _) fun cw =>
    Sound.bind' (sound_mapM _ (fun z x => ev z ok && ev z x) (fun x => sound_and ok x) cw) fun rout =>
    Sound.pure' fun z hrout hcw hlw hrin hcr hlr hok T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw htrow hxr hyr hxw hyw
      hsr hsw => ?_
  have hls : ∀ X Y : ByteArray, (RecLinkTree.leaf (Hm96.Default512.commitString X Y)).size = 64 :=
    fun _ _ => Sha512.hash_size _
  have hD : ∀ aux : List Lin, (VStar.sibForms aux (UB + PB)).length = UB + PB := fun _ => by simp [VStar.sibForms]
  have hcr' := hcr _ (dirIdx T 1 2 UB PB) sr (hls Xr Yr) (hlr Xr Yr hXr hYr hxr hyr) hsr fun j hj =>
    dirs_ok htrow hT hUB hPB 1 2 (by norm_num) (by norm_num) j (by rw [hD] at hj; exact hj)
  have hcw' := hcw _ (dirIdx T 3 4 UB PB) sw (hls Xw Yw) (hlw Xw Yw hXw hYw hxw hyw) hsw fun j hj =>
    dirs_ok htrow hT hUB hPB 3 4 (by norm_num) (by norm_num) j (by rw [hD] at hj; exact hj)
  have hok' := hok T Xr Xw hT hXr hXw htrow hxr hxw
  have hR := carries_append (carries_gate hcr' (AccChain.climb_size_of _ (fun _ _ => rfl) _ _ _ (hls Xr Yr)) hrin)
    (carries_linkZeros z)
  have hW := carries_append (carries_gate hcw' (AccChain.climb_size_of _ (fun _ _ => rfl) _ _ _ (hls Xw Yw)) hrout)
    (carries_linkZeros z)
  rw [hok'] at hR hW
  refine ⟨?_, ?_⟩
  · show Carries z _ (rootB _ _)
    rw [rootB_eq (r' := Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xr Yr))
      (dirIdx T 1 2 UB PB) sr) fun hk => by
        rw [dirIdx_eq (RecLinkRef.okRef_fits hk).2.1 (RecLinkRef.okRef_fits hk).2.2.2.1]]
    exact hR
  · show Carries z _ (rootB _ _)
    rw [rootB_eq (r' := Flock.climb RecLinkRef.scheme (RecLinkTree.leaf (Hm96.Default512.commitString Xw Yw))
      (dirIdx T 3 4 UB PB) sw) fun hk => by
        rw [dirIdx_eq (RecLinkRef.okRef_fits hk).2.2.1 (RecLinkRef.okRef_fits hk).2.2.2.2]]
    exact hW

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

theorem fresh_hm96Leaf (x y : List Lin) : Fresh (VStar.hm96Leaf x y) := by
  unfold VStar.hm96Leaf
  simp only [OfVStar.shaFrom_eq]
  exact Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_commit _) _) fun _ =>
    Fresh.bind' (fresh_shaFrom _ _ _) fun _ => fresh_shaFrom _ _ _

theorem fresh_linkSteps (UB PB : ℕ) (trow xr yr raux xw yw waux : List Lin) :
    Fresh (VStar.linkSteps UB PB trow xr yr raux xw yw waux) := by
  unfold VStar.linkSteps
  simp only [OfVStar.climbFrom_eq]
  exact Fresh.bind' (fresh_okForms _ _ _ _ _) fun _ => Fresh.bind' (fresh_hm96Leaf _ _) fun _ =>
    Fresh.bind' (fresh_climbFrom _ _ _ _) fun _ => Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ =>
    Fresh.bind' (fresh_hm96Leaf _ _) fun _ => Fresh.bind' (fresh_climbFrom _ _ _ _) fun _ =>
    Fresh.bind' (Fresh.mapM' _ (fun _ => fresh_and _ _) _) fun _ => Fresh.pure' _

/-! ## V*'s `RecLink` unit -/

/-- An `aux` port's bits: `16 A(D)`. -/
abbrev auxBits (UB PB : ℕ) : ℕ := 16 * VStar.auxWords (UB + PB)

/-- The unit's input bits: its seven ports' words, 16 bits each. -/
def linkIn (UB PB : ℕ) : ℕ := 16 * (VStar.linkPorts UB PB).sum

theorem linkIn_eq (UB PB : ℕ) : linkIn UB PB = 5120 + 2 * auxBits UB PB := by
  simp only [linkIn, VStar.linkPorts, List.sum_cons, List.sum_nil, VStar.LINK_ROW_WORDS, VStar.LINK_X_WORDS,
    VStar.LINK_Y_WORDS, auxBits]
  omega

/-- The unit's builder after its input: the steps on the ports `trow, xr, yr, raux, xw, yw, waux` in that order. -/
def linkRun (UB PB : ℕ) : M (List Lin × List Lin) :=
  VStar.linkSteps UB PB (vars 0 1024) (vars 1024 512) (vars 1536 1536) (vars 3072 (auxBits UB PB))
    (vars (3072 + auxBits UB PB) 512) (vars (3584 + auxBits UB PB) 1536) (vars (5120 + auxBits UB PB) (auxBits UB PB))

theorem cons_vars {k k' n n' : ℕ} {l l' : List (List Lin)} (hk : k = k') (hn : n = n') (hl : l = l') :
    vars k n :: l = vars k' n' :: l' := by
  subst hk hn hl
  rfl

theorem varSlices_link (a : ℕ) : OfVStar.varSlices 0 [64, 32, 96, a, 32, 96, a] =
    [vars 0 1024, vars 1024 512, vars 1536 1536, vars 3072 (16 * a), vars (3072 + 16 * a) 512,
      vars (3584 + 16 * a) 1536, vars (5120 + 16 * a) (16 * a)] := by
  refine cons_vars rfl rfl (cons_vars rfl rfl (cons_vars rfl rfl (cons_vars rfl rfl (cons_vars ?_ rfl
    (cons_vars ?_ rfl (cons_vars ?_ rfl rfl)))))) <;> omega

theorem varSlices_linkPorts (UB PB : ℕ) : OfVStar.varSlices 0 (VStar.linkPorts UB PB) =
    [vars 0 1024, vars 1024 512, vars 1536 1536, vars 3072 (auxBits UB PB), vars (3072 + auxBits UB PB) 512,
      vars (3584 + auxBits UB PB) 1536, vars (5120 + auxBits UB PB) (auxBits UB PB)] :=
  varSlices_link (VStar.auxWords (UB + PB))

/-- **V*'s unit is `linkRun`**: its builder's state and outputs are `linkRun`'s from its input bits, its input ports
`VStar.linkPorts` and its output ports `rin`, `rout`. -/
theorem linkUnit_ok {UB PB : ℕ} {U : VStar.UnitC} (h : VStar.linkUnit UB PB = .ok U) :
    U.st = ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).2 ∧
      U.outs = (((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.1 ++ ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.2).toArray ∧
      U.inPorts = VStar.linkPorts UB PB ∧ U.outPorts = [VStar.LINK_OUT_WORDS, VStar.LINK_OUT_WORDS] := by
  unfold VStar.linkUnit at h
  cases hc : RecLinkTree.checkShape UB PB with
  | error e => simp [hc, bind, Except.bind] at h
  | ok _ =>
    simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
    subst h
    have hn : 16 * (VStar.linkPorts UB PB).foldl (· + ·) 0 = 16 * (VStar.linkPorts UB PB).sum + 0 := by
      simp [VStar.linkPorts]
      omega
    simp only [OfVStar.run_input_bind]
    rw [hn, OfVStar.slices_vars, varSlices_linkPorts, Nat.add_zero]
    exact ⟨rfl, rfl, rfl, rfl⟩

theorem checkShape_ok {UB PB : ℕ} {x : Unit} (h : RecLinkTree.checkShape UB PB = .ok x) : UB ≤ 32 ∧ PB ≤ 32 := by
  unfold RecLinkTree.checkShape at h
  by_cases hc : (decide (UB ≤ RecLinkTree.UB_MAX) && decide (PB ≤ RecLinkTree.PB_MAX) &&
      decide (UB + PB ≤ RecLinkTree.D_MAX)) = true
  · simp only [Bool.and_eq_true, decide_eq_true_eq, RecLinkTree.UB_MAX, RecLinkTree.PB_MAX, RecLinkTree.D_MAX] at hc
    omega
  · simp [hc] at h

theorem linkUnit_shape {UB PB : ℕ} {U : VStar.UnitC} (h : VStar.linkUnit UB PB = .ok U) : UB ≤ 32 ∧ PB ≤ 32 := by
  unfold VStar.linkUnit at h
  cases hc : RecLinkTree.checkShape UB PB with
  | error e => simp [hc, bind, Except.bind] at h
  | ok x => exact checkShape_ok hc

theorem aux_ge (D : ℕ) : 512 * D ≤ 16 * VStar.auxWords D := by
  unfold VStar.auxWords VStar.NODE_WORDS VStar.ROW_BLOCK
  have := Nat.le_max_right 1 ((32 * D + 64 - 1) / 64)
  omega

/-- **V*'s laid-out `RecLink` unit computes its reference outputs**: whenever its rows hold at `v`, if its input columns
hold the table row `T` (from column 0), the reader's digest `Xr` (1024) and salt `Yr` (1536), its 64-byte siblings
`sr` (from 3072, 512 columns each), the writer's `Xw` and `Yw` (past the reader's `aux` port) and its siblings `sw`
(past the writer's salt), then its output columns hold `linkB`'s `rin ‖ rout`. -/
theorem laid_linkUnit {UB PB : ℕ} {U : VStar.UnitC} (hU : VStar.linkUnit UB PB = .ok U) {l : Laid}
    (hl : U.laid = .ok l) (v : ℕ → Bool) (h : LaidRows l (inRowsOf (Array.replicate U.inWords 16)) v)
    (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray)
    (hT : T.size = 128) (hXr : Xr.size = 64) (hYr : Yr.size = 192) (hXw : Xw.size = 64) (hYw : Yw.size = 192)
    (hsr : sr.length = UB + PB) (hsw : sw.length = UB + PB)
    (htrow : ColsHold v 0 T) (hxr : ColsHold v 1024 Xr) (hyr : ColsHold v 1536 Yr)
    (hraux : ∀ i (hi : i < sr.length), sr[i].size = 64 ∧ ColsHold v (3072 + 512 * i) sr[i])
    (hxw : ColsHold v (3072 + auxBits UB PB) Xw) (hyw : ColsHold v (3584 + auxBits UB PB) Yw)
    (hwaux : ∀ i (hi : i < sw.length), sw[i].size = 64 ∧ ColsHold v (5120 + auxBits UB PB + 512 * i) sw[i]) :
    ColsHold v (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((linkB UB PB T Xr Yr Xw Yw sr sw).1 ++ (linkB UB PB T Xr Yr Xw Yw sr sw).2) := by
  obtain ⟨hst, houts, hin, hop⟩ := linkUnit_ok hU
  obtain ⟨hUB, hPB⟩ := linkUnit_shape hU
  have hN : (Array.replicate U.inWords 16).foldl (· + ·) 0 = 5120 + 2 * auxBits UB PB := by
    rw [OfVStar.foldl_replicate, VStar.UnitC.inWords, hin]
    simp only [VStar.linkPorts, List.foldl_cons, List.foldl_nil, VStar.LINK_ROW_WORDS, VStar.LINK_X_WORDS,
      VStar.LINK_Y_WORDS, auxBits]
    omega
  have hA := aux_ge (UB + PB)
  have hg : Good ((Array.replicate U.inWords 16).foldl (· + ·) 0) U.st := by
    rw [hst, hN, ← linkIn_eq]
    exact fresh_linkSteps UB PB _ _ _ _ _ _ _ (linkIn UB PB) ⟨linkIn UB PB, []⟩ ⟨by simp, fun k r h => by simp at h⟩
  rw [OfVStar.laid_ok hl] at h
  have hz := laid_rows U.st _ _ U.outs v hg h
  rw [hst] at hz
  have hP := (sound_linkSteps UB PB hUB hPB (vars 0 1024) (vars 1024 512) (vars 1536 1536) (vars 3072 (auxBits UB PB))
    (vars (3072 + auxBits UB PB) 512) (vars (3584 + auxBits UB PB) 1536) (vars (5120 + auxBits UB PB) (auxBits UB PB))
    ⟨linkIn UB PB, []⟩).2 _ hz
  obtain ⟨h1, h2⟩ := hP T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw
    (carries_vars (by omega) (by omega) htrow) (carries_vars (by omega) (by omega) hxr)
    (carries_vars (by omega) (by omega) hyr) (carries_vars (by omega) (by omega) hxw)
    (carries_vars (by omega) (by omega) hyw) (forall₂_sibs sr hsr hA (by omega) hraux)
    (forall₂_sibs sw hsw hA (by omega) hwaux)
  have hls : ∀ X Y : ByteArray, (RecLinkTree.leaf (Hm96.Default512.commitString X Y)).size = 64 :=
    fun _ _ => Sha512.hash_size _
  have hlen : (((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.1 ++ ((linkRun UB PB).run ⟨linkIn UB PB, []⟩).1.2).length ≤
      (Array.replicate U.outWords 16).foldl (· + ·) 0 := by
    rw [OfVStar.foldl_replicate, VStar.UnitC.outWords, hop, List.length_append]
    show _ + _ ≤ 2048
    rw [h1.1, h2.1, linkB, size_rootB _ (AccChain.climb_size_of _ (fun _ _ => rfl) _ _ _ (hls Xr Yr)),
      size_rootB _ (AccChain.climb_size_of _ (fun _ _ => rfl) _ _ _ (hls Xw Yw))]
  rw [houts] at h
  exact colsHold_outs (st := U.st) hlen h (carries_append h1 h2)

section Unit

open FlockSoundness FlockSoundness.Audit

/-- **A copy of V*'s laid-out `RecLink` unit, in a unit of any partition, computes its reference outputs**
(`laid_linkUnit`): let the verifier build `net` from the layout of `VStar.linkUnit UB PB` (`netOf`), and let unit `u` be
an instance of `k` stacked copies of the net's rows, the transcript `X` carrying 1 at the stack's constant. Then at copy
`j`'s column values (`copyVal`), `laid_linkUnit` holds. -/
theorem unit_linkUnit {C : Audit.Circuit} {n : ℕ} {P : Partition C n} {u : Fin n} {net : Flock.Net} (hB : Built net)
    {k : ℕ} (I : P.IsRowsUnit u ((Rows.ofNet hB).stack k)) (X : Fin C.N → Bool) {j : ℕ} (hj : j < k)
    (hone : X (I.wire ((Rows.ofNet hB).stack k).one) = true) {UB PB : ℕ} {U : VStar.UnitC}
    (hU : VStar.linkUnit UB PB = .ok U) {l : Laid} (hl : U.laid = .ok l) {name : String}
    (hn : Flock.HmNets.netOf name l = .ok net) (hu : net.unitLog ≤ 32)
    (T Xr Yr Xw Yw : ByteArray) (sr sw : List ByteArray)
    (hT : T.size = 128) (hXr : Xr.size = 64) (hYr : Yr.size = 192) (hXw : Xw.size = 64) (hYw : Yw.size = 192)
    (hsr : sr.length = UB + PB) (hsw : sw.length = UB + PB)
    (htrow : ColsHold (copyVal I X hj) 0 T) (hxr : ColsHold (copyVal I X hj) 1024 Xr)
    (hyr : ColsHold (copyVal I X hj) 1536 Yr)
    (hraux : ∀ i (hi : i < sr.length), sr[i].size = 64 ∧ ColsHold (copyVal I X hj) (3072 + 512 * i) sr[i])
    (hxw : ColsHold (copyVal I X hj) (3072 + auxBits UB PB) Xw)
    (hyw : ColsHold (copyVal I X hj) (3584 + auxBits UB PB) Yw)
    (hwaux : ∀ i (hi : i < sw.length), sw[i].size = 64 ∧
      ColsHold (copyVal I X hj) (5120 + auxBits UB PB + 512 * i) sw[i]) :
    ColsHold (copyVal I X hj) (outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16))
      ((linkB UB PB T Xr Yr Xw Yw sr sw).1 ++ (linkB UB PB T Xr Yr Xw Yw sr sw).2) := by
  obtain ⟨hs, h1⟩ := sat_copy I X hj
  have hL := laidRows_ofNet hB hn hu hs (h1.trans hone)
  rw [OfVStar.inWords_laid hl hn] at hL
  exact laid_linkUnit hU hl _ hL T Xr Yr Xw Yw sr sw hT hXr hYr hXw hYw hsr hsw htrow hxr hyr hraux hxw hyw hwaux

end Unit

/-! ## At the reference's u16 words -/

theorem bytesOf_append (a b : List ℕ) :
    RecLinkRef.bytesOf (a ++ b) = RecLinkRef.bytesOf a ++ RecLinkRef.bytesOf b := by
  apply ByteArray.ext
  simp [RecLinkRef.bytesOf, List.flatMap_append]

theorem flatMap_zeros : ∀ n : ℕ,
    (List.replicate n 0).flatMap (fun w : ℕ => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]) =
      List.replicate (2 * n) (0 : UInt8)
  | 0 => rfl
  | n + 1 => by
    rw [List.replicate_succ, List.flatMap_cons, flatMap_zeros n, show 2 * (n + 1) = 2 * n + 1 + 1 by omega,
      List.replicate_succ, List.replicate_succ]
    rfl

theorem bytesOf_replicate (n : ℕ) : RecLinkRef.bytesOf (List.replicate n 0) = zeroB (2 * n) := by
  unfold RecLinkRef.bytesOf zeroB
  rw [flatMap_zeros]
  rfl

theorem zeroB_add (m n : ℕ) : zeroB (m + n) = zeroB m ++ zeroB n := by
  apply ByteArray.ext
  simp [zeroB]

theorem toUInt8_toNat (a : UInt8) : a.toNat.toUInt8 = a := by
  apply UInt8.toNat_inj.1
  simp

theorem pair_lo (a b : UInt8) : ((a.toNat + 256 * b.toNat) % 256).toUInt8 = a := by
  rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt (UInt8.toNat_lt a), toUInt8_toNat]

theorem pair_hi (a b : UInt8) : ((a.toNat + 256 * b.toNat) / 256 % 256).toUInt8 = b := by
  rw [show a.toNat + 256 * b.toNat = a.toNat + b.toNat * 256 by omega, Nat.add_mul_div_right _ _ (by omega),
    Nat.div_eq_of_lt (UInt8.toNat_lt a), Nat.zero_add, Nat.mod_eq_of_lt (UInt8.toNat_lt b), toUInt8_toNat]

theorem get!_eq (b : ByteArray) (i : ℕ) : b.get! i = b.data.toList.getD i 0 := by
  cases b with
  | mk bs =>
    show bs[i]! = _
    rw [Array.getElem!_eq_getD]
    unfold Array.getD
    split <;> simp_all <;> rfl

theorem pairs_eq : ∀ (l : List UInt8) (m : ℕ), l.length = 2 * m →
    ((List.range m).map fun i => (l.getD (2 * i) 0).toNat + 256 * (l.getD (2 * i + 1) 0).toNat).flatMap
      (fun w : ℕ => [(w % 256).toUInt8, (w / 256 % 256).toUInt8]) = l
  | [], 0, _ => rfl
  | a :: b :: r, m + 1, h => by
    rw [List.range_succ_eq_map, List.map_cons, List.flatMap_cons, List.map_map]
    have hr : r.length = 2 * m := by rw [List.length_cons, List.length_cons] at h; omega
    have := pairs_eq r m hr
    simp only [Nat.mul_zero, Nat.zero_add, List.getD_cons_zero, List.getD_cons_succ] at *
    rw [pair_lo, pair_hi]
    show a :: b :: _ = a :: b :: r
    congr 2
  | [], m + 1, h => by rw [List.length_nil] at h; omega
  | [_], _, h => by rw [List.length_singleton] at h; omega
  | _ :: _ :: _, 0, h => by rw [List.length_cons, List.length_cons] at h; omega

/-- `bytes_of(words_of(b))` is `b` for `b` of even size. -/
theorem bytesOf_wordsOf (b : ByteArray) (h : b.size % 2 = 0) : RecLinkRef.bytesOf (RecLinkRef.wordsOf b) = b := by
  apply ByteArray.ext
  apply Array.toList_inj.1
  unfold RecLinkRef.bytesOf RecLinkRef.wordsOf
  simp only [get!_eq]
  have hl : b.data.toList.length = 2 * (b.size / 2) := by
    rw [Array.length_toList, ByteArray.size_data]
    omega
  rw [List.toList_toArray]
  exact pairs_eq _ _ hl

theorem bytesOf_expected {r : ByteArray} (hr : r.size = 64) :
    RecLinkRef.bytesOf (RecLinkRef.expected r) = r ++ zeroB 64 :=
  (bytesOf_append _ _).trans (congr (congrArg (· ++ ·) (bytesOf_wordsOf r (by omega))) (bytesOf_replicate 32))

theorem bytesOf_zeros : RecLinkRef.bytesOf RecLinkRef.zeros = zeroB 64 ++ zeroB 64 :=
  (bytesOf_replicate 64).trans (zeroB_add 64 64)

/-- **The unit's outputs are the reference's `linkOut`**: on ports that are the bytes of u16 words (`bytesOf`), with
the siblings `sibsOf` reads off `raux` and `waux`, `linkB`'s `(rin, rout)` are the bytes of `linkOut`'s. -/
theorem linkB_words (UB PB : ℕ) (trow xr yr raux xw yw waux : List ℕ) :
    linkB UB PB (RecLinkRef.bytesOf trow) (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf yr) (RecLinkRef.bytesOf xw)
        (RecLinkRef.bytesOf yw) (RecLinkRef.sibsOf raux (UB + PB)) (RecLinkRef.sibsOf waux (UB + PB)) =
      (RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).1,
        RecLinkRef.bytesOf (RecLinkRef.linkOut UB PB trow xr yr raux xw yw waux).2) := by
  have hls : ∀ X Y : ByteArray, (RecLinkTree.leaf (Hm96.Default512.commitString X Y)).size = 64 :=
    fun _ _ => Sha512.hash_size _
  have h1 := AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) (RecLinkRef.sibsOf raux (UB + PB))
    (RecLinkRef.posOf (RecLinkRef.bytesOf trow) 1 2 PB (UB + PB)) _
    (hls (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf yr))
  have h2 := AccChain.climb_size_of RecLinkRef.scheme (fun _ _ => rfl) (RecLinkRef.sibsOf waux (UB + PB))
    (RecLinkRef.posOf (RecLinkRef.bytesOf trow) 3 4 PB (UB + PB)) _
    (hls (RecLinkRef.bytesOf xw) (RecLinkRef.bytesOf yw))
  unfold linkB RecLinkRef.linkOut
  cases RecLinkRef.okRef UB PB (RecLinkRef.bytesOf trow) (RecLinkRef.bytesOf xr) (RecLinkRef.bytesOf xw)
  · exact Prod.ext bytesOf_zeros.symm bytesOf_zeros.symm
  · exact Prod.ext (bytesOf_expected h1).symm (bytesOf_expected h2).symm

end FlockVBridge.RecLink
