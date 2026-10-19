import Flock.Bytes
import Flock.Field

/-!
# The record's hex round trips

The record writes bytes as lowercase hex (`hexOf`) and an F128 as `{hi:016x}{lo:016x}` (`F128.toHex`); the decoder reads
them back with `unhex` and `f128OfHex`. Both round trips are exact: `unhex_hexOf`, `f128OfHex_toHex`.
-/

namespace FlockComplete.JsonRT

open Flock

theorem foldl_loop_eq {β : Type} (f : β → UInt8 → β) (as : ByteArray) (stop : Nat) (h : stop ≤ as.size) :
    ∀ (i j : Nat) (b : β), i + j = stop →
      ByteArray.foldlM.loop (m := Id) (fun b a => pure (f b a)) as stop h i j b =
        ((as.data.toList.drop j).take i).foldl f b := by
  intro i
  induction i with
  | zero => intro j b _; unfold ByteArray.foldlM.loop; split <;> rfl
  | succ i ih =>
    intro j b hij
    unfold ByteArray.foldlM.loop
    have hj : j < stop := by omega
    rw [dite_eq_left_of_eq_true (eq_true hj)]
    have hjs : j < as.data.toList.length := by simp; omega
    rw [List.drop_eq_getElem_cons hjs]
    simp only [List.take_succ_cons, List.foldl_cons]
    rw [← ih (j + 1) _ (by omega)]
    simp [Array.getElem_toList]
    rfl

theorem byteArray_foldl_eq {β : Type} (f : β → UInt8 → β) (init : β) (as : ByteArray) :
    as.foldl f init = as.data.toList.foldl f init := by
  unfold ByteArray.foldl ByteArray.foldlM
  rw [dite_eq_left_of_eq_true (eq_true (Nat.le_refl _))]
  dsimp only
  rw [foldl_loop_eq f as as.size (Nat.le_refl _) (as.size - 0) 0 init (by omega)]
  simp only [List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (by simp)]
  rfl

/-- A nibble's lowercase hex digit. -/
def hd (d : Nat) : Char := if d < 10 then Char.ofNat (48 + d) else Char.ofNat (87 + d)

theorem foldl_push2 (d₁ d₂ : UInt8 → Char) : ∀ (l : List UInt8) (s : String),
    (l.foldl (fun s x => (s.push (d₁ x)).push (d₂ x)) s).toList =
      s.toList ++ l.flatMap fun x => [d₁ x, d₂ x]
  | [], s => by simp
  | x :: l, s => by
    rw [List.foldl_cons, foldl_push2 d₁ d₂ l]
    simp [String.toList_push]

theorem hexOf_toList (b : ByteArray) :
    (hexOf b).toList = b.data.toList.flatMap fun x => [hd (x.toNat / 16), hd (x.toNat % 16)] := by
  unfold hexOf
  rw [byteArray_foldl_eq, foldl_push2]
  simp only [String.toList_empty, List.nil_append]
  congr 1


/-- `unhex`'s digit value. -/
def hv (c : Char) : Option Nat :=
  if c.isDigit = true then some (c.toNat - 48)
  else if (decide ('a' ≤ c) && decide (c ≤ 'f')) = true then some (c.toNat - 87) else none

theorem hv_hd_nat : ∀ d < 16, hv (hd d) = some d := by decide +kernel

theorem hv_hd {d : Nat} (h : d < 16) : hv (hd d) = some d := hv_hd_nat d h

theorem length_flatMap2 (L : List UInt8) (f g : UInt8 → Char) :
    (L.flatMap fun x => [f x, g x]).length = 2 * L.length := by
  induction L with
  | nil => rfl
  | cons a L ih => simp only [List.flatMap_cons, List.length_append, ih, List.length_cons, List.length_nil]; omega

theorem flatMap_get? (L : List UInt8) (f g : UInt8 → Char) (i : Nat) (hi : i < L.length) :
    (L.flatMap fun x => [f x, g x])[2 * i]? = some (f L[i]) ∧
      (L.flatMap fun x => [f x, g x])[2 * i + 1]? = some (g L[i]) := by
  induction L generalizing i with
  | nil => simp at hi
  | cons a L ih =>
    cases i with
    | zero => simp
    | succ i =>
      have := ih i (by simpa using hi)
      simp only [List.flatMap_cons, List.getElem_cons_succ]
      rw [List.getElem?_append_right (by simp; omega), List.getElem?_append_right (by simp; omega)]
      simpa [show 2 * (i + 1) - 2 = 2 * i by omega, show 2 * (i + 1) + 1 - 2 = 2 * i + 1 by omega] using this

theorem flatMap_getD (L : List UInt8) (f g : UInt8 → Char) (i : Nat) (hi : i < L.length) :
    (L.flatMap fun x => [f x, g x]).toArray.getD (2 * i) ' ' = f L[i] ∧
      (L.flatMap fun x => [f x, g x]).toArray.getD (2 * i + 1) ' ' = g L[i] := by
  obtain ⟨h1, h2⟩ := flatMap_get? L f g i hi
  simp [Array.getD_eq_getD_getElem?, h1, h2]

theorem forIn_bytes (L : List UInt8) (body : Nat → ByteArray → V (ForInStep ByteArray))
    (hb : ∀ i (hi : i < L.length) acc, body i acc = .ok (.yield (acc.push L[i]))) :
    ∀ n s acc, s + n = L.length →
      forIn (List.range' s n) acc body = .ok ⟨acc.data ++ (L.drop s).toArray⟩ := by
  intro n
  induction n with
  | zero =>
    intro s acc h
    simp [show L.drop s = [] from List.drop_eq_nil_of_le (by omega)]
    rfl
  | succ n ih =>
    intro s acc h
    rw [List.range'_succ, List.forIn_cons, hb s (by omega) acc]
    show forIn (List.range' (s + 1) n) (acc.push L[s]) body = _
    rw [ih (s + 1) _ (by omega), List.drop_eq_getElem_cons (i := s) (l := L) (by omega)]
    simp [ByteArray.data_push]

/-- **Hex reads back**: `unhex` of `hexOf b` is `b`. -/
theorem unhex_hexOf (b : ByteArray) : unhex (hexOf b) = .ok b := by
  have hl := hexOf_toList b
  have hlen : (hexOf b).toList.toArray.size = 2 * b.data.toList.length := by
    rw [hl, List.size_toArray, length_flatMap2]
  unfold unhex
  simp only [hlen, Nat.mul_mod_right, bne_self_eq_false, Bool.false_eq_true, ite_false,
    Nat.mul_div_cancel_left _ (by decide : 0 < 2), Std.Legacy.Range.forIn_eq_forIn_range']
  have hd2 : ∀ i (hi : i < b.data.toList.length),
      (hexOf b).toList.toArray.getD (2 * i) ' ' = hd (b.data.toList[i].toNat / 16) ∧
        (hexOf b).toList.toArray.getD (2 * i + 1) ' ' = hd (b.data.toList[i].toNat % 16) := by
    intro i hi
    rw [hl]
    exact flatMap_getD _ _ _ i hi
  rw [forIn_bytes b.data.toList _ ?_ _ 0 _ (by simp)]
  · simp only [ByteArray.emptyWithCapacity, List.drop_zero, Array.toArray_toList]
    show Except.ok (⟨#[] ++ b.data⟩ : ByteArray) = _
    rw [Array.empty_append]
  · intro i hi acc
    have hx : b.data.toList[i].toNat < 256 := UInt8.toNat_lt _
    obtain ⟨e1, e2⟩ := hd2 i hi
    have v1 := hv_hd (d := b.data.toList[i].toNat / 16) (by omega)
    have v2 := hv_hd (d := b.data.toList[i].toNat % 16) (by omega)
    unfold hv at v1 v2
    simp only [e1, e2, v1, v2, Nat.div_add_mod', Nat.toUInt8_eq, UInt8.ofNat_toNat]
    rfl

/-- A word's eight bytes, most significant first. -/
def be8 (x : UInt64) : List UInt8 :=
  [(x >>> 56).toUInt8, (x >>> 48).toUInt8, (x >>> 40).toUInt8, (x >>> 32).toUInt8,
   (x >>> 24).toUInt8, (x >>> 16).toUInt8, (x >>> 8).toUInt8, x.toUInt8]

theorem foldl_push_list (g : Nat → Char) : ∀ (l : List Nat) (s : String),
    (l.foldl (fun s i => s.push (g i)) s).toList = s.toList ++ l.map g
  | [], s => by simp
  | i :: l, s => by rw [List.foldl_cons, foldl_push_list g l]; simp [String.toList_push]

theorem and_15 (x : Nat) : x &&& 15 = x % 16 := Nat.and_two_pow_sub_one_eq_mod x 4

theorem toHex_toList (a : F128) : (F128.toHex a).toList =
    (be8 a.hi ++ be8 a.lo).flatMap fun x => [hd (x.toNat / 16), hd (x.toNat % 16)] := by
  unfold F128.toHex
  simp only [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, pure_bind, Id.run_pure]
  rw [foldl_push_list, foldl_push_list]
  have hhi := a.hi.toNat_lt
  have hlo := a.lo.toNat_lt
  rw [show List.range' 0 [:16].size 1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] from rfl]
  simp only [be8, String.toList_empty, List.nil_append, List.map_cons, List.map_nil,
    List.cons_append, List.nil_append, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.cons.injEq,
    and_true]
  and_intros
  all_goals
    show hd _ = hd _
    congr 1
    simp only [UInt64.toNat_and, UInt64.toNat_shiftRight, UInt64.toNat_toUInt8, Nat.shiftRight_eq_div_pow]
    simp [and_15] <;> omega

theorem shift_or (a : UInt64) (c : UInt8) : (a <<< 8 ||| c.toUInt64).toNat = (a.toNat * 256 + c.toNat) % 2 ^ 64 := by
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft, UInt8.toNat_toUInt64]
  have hc := c.toNat_lt
  have e : a.toNat <<< ((8 : UInt64).toNat % 64) % 2 ^ 64 = (a.toNat % 2 ^ 56) <<< 8 := by
    simp only [Nat.shiftLeft_eq]
    show a.toNat * 2 ^ 8 % 2 ^ 64 = a.toNat % 2 ^ 56 * 2 ^ 8
    omega
  rw [e, ← Nat.shiftLeft_add_eq_or_of_lt hc, Nat.shiftLeft_eq]
  omega

theorem be8_fold (x : UInt64) : (be8 x).foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0 = x := by
  apply UInt64.toNat_inj.1
  have hx := x.toNat_lt
  simp only [be8, List.foldl_cons, List.foldl_nil, shift_or, UInt64.toNat_toUInt8, UInt64.toNat_shiftRight,
    Nat.shiftRight_eq_div_pow]
  simp
  omega

theorem fold_get (B : ByteArray) (o : Nat) (l : List UInt8)
    (hl : (List.range' 0 8).map (fun i => B.get! (o + i)) = l) :
    (List.range' 0 [:8].size 1).foldl (fun (s : UInt64) i => s <<< 8 ||| (B.get! (o + i)).toUInt64) 0 =
      l.foldl (fun (s : UInt64) c => s <<< 8 ||| c.toUInt64) 0 := by
  rw [← hl, List.foldl_map]
  rfl

/-- **An F128's hex reads back**: `f128OfHex` of `toHex a` is `a`. -/
theorem f128OfHex_toHex (a : F128) : f128OfHex a.toHex = .ok a := by
  have e : a.toHex = hexOf ⟨(be8 a.hi ++ be8 a.lo).toArray⟩ := by
    apply String.toList_inj.1
    rw [toHex_toList, hexOf_toList]
  unfold f128OfHex
  rw [e, unhex_hexOf]
  simp only [bind, Except.bind]
  have h16 : ((ByteArray.mk (be8 a.hi ++ be8 a.lo).toArray).size != 16) = false := rfl
  rw [h16]
  simp only [Bool.false_eq_true, ite_false, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  rw [fold_get _ 0 (be8 a.hi) rfl, fold_get _ 8 (be8 a.lo) rfl, be8_fold, be8_fold]
  rfl

end FlockComplete.JsonRT
