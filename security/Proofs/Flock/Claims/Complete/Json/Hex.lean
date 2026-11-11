module

public import Flock.Bytes

/-!
# The record's hex round trips

The record writes bytes as lowercase hex (`hexOf`) and an F128 as `{hi:016x}{lo:016x}` (`F128.toHex`); the decoder reads
them back with `unhex` and `f128OfHex`. Both round trips are exact: `unhex_hexOf` (`Flock.Hex`, beside the codec),
`f128OfHex_toHex`.
-/

@[expose] public section

namespace FlockComplete.JsonRT

open Flock

export Flock.Hex (foldl_loop_eq byteArray_foldl_eq hd foldl_push2 hexOf_toList hv hv_hd_nat hv_hd length_flatMap2
  flatMap_get? flatMap_getD forIn_bytes unhex_hexOf)

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
