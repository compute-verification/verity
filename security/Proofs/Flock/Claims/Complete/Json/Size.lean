module

public import Proofs.Flock.Claims.Complete.Json.Record
public import Proofs.Flock.Claims.Complete.Prover.Print
import Std.Data.TreeMap.Raw.Lemmas
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Algebra.Order.BigOperators.Group.List
import Mathlib.Data.List.Perm.Subperm
import all Lean.Data.Json.Basic

/-!
# A printed JSON value's length

`printJ`'s bytes are at most `jsz`, a weight read off the value's tree (`printJ_size_le`): four for `null`, five for a
boolean, a number's own text, and for a string eighteen per character and two quotes (`strJ_size_le`: `escC` writes a
character as `\u` and four characters, each at most four bytes, or fewer); an array or an object adds two brackets and
a comma per element, an object's field its key, printed, and a colon. An object of distinct keys weighs at most what
its list does (`jsz_mkObj_le`), whatever order its tree holds them in.
-/

@[expose] public section

namespace FlockComplete

open Lean (Json)
open Std.DTreeMap.Internal (Impl)

/-! ## Strings -/

theorem ofList_size_le : ∀ l : List Char, (String.ofList l).utf8ByteSize ≤ 4 * l.length
  | [] => by decide +kernel
  | c :: l => by
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, List.length_cons]
    have h1 := Char.utf8Size_le_four c
    have h2 := ofList_size_le l
    omega

theorem escC_size_le (c : Char) : (escC c).utf8ByteSize ≤ 18 := by
  unfold escC
  split
  · decide +kernel
  split
  · decide +kernel
  split
  · decide +kernel
  split
  · decide +kernel
  split
  · rw [String.utf8ByteSize_singleton]
    have := Char.utf8Size_le_four c
    omega
  · try dsimp only
    rw [String.utf8ByteSize_append]
    have h1 : ("\\u" : String).utf8ByteSize = 2 := by decide +kernel
    refine Nat.le_trans (Nat.add_le_add_left (ofList_size_le _) _) ?_
    rw [h1]
    show 2 + 4 * 4 ≤ 18
    omega

theorem join_escC_size_le : ∀ l : List Char, (String.join (l.map escC)).utf8ByteSize ≤ 18 * l.length
  | [] => by
    rw [List.map_nil, String.join_nil, String.utf8ByteSize_empty]
    exact Nat.zero_le _
  | c :: l => by
    rw [List.map_cons, String.join_cons, String.utf8ByteSize_append, List.length_cons]
    have h1 := escC_size_le c
    have h2 := join_escC_size_le l
    omega

/-- **A printed string's bytes**: at most eighteen per character and its two quotes. -/
theorem strJ_size_le (s : String) : (strJ s).utf8ByteSize ≤ 18 * s.length + 2 := by
  unfold strJ
  rw [String.utf8ByteSize_append, String.utf8ByteSize_append]
  have h1 := join_escC_size_le s.toList
  rw [String.length_toList] at h1
  have h2 : ("\"" : String).utf8ByteSize = 1 := by decide +kernel
  omega

/-- Strings joined by commas: their bytes and a comma each. -/
theorem intercalate_size_le : ∀ l : List String,
    (",".intercalate l).utf8ByteSize ≤ (l.map String.utf8ByteSize).sum + l.length
  | [] => by
    rw [String.intercalate_nil, String.utf8ByteSize_empty]
    all_goals exact Nat.zero_le _
  | [t] => by
    rw [String.intercalate_singleton, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, List.length_cons,
      List.length_nil]
    all_goals omega
  | t :: u :: l => by
    have ih := intercalate_size_le (u :: l)
    have h : (",":String).utf8ByteSize = 1 := by decide +kernel
    rw [String.intercalate_cons_cons, String.utf8ByteSize_append, String.utf8ByteSize_append, List.map_cons,
      List.sum_cons, List.length_cons]
    all_goals omega

/-! ## JSON values -/

mutual
/-- **A JSON value's weight**, which bounds its printed bytes (`printJ_size_le`). -/
def jsz : Json → ℕ
  | .null => 4
  | .bool _ => 5
  | .num n => (toString n).utf8ByteSize
  | .str s => 18 * s.length + 2
  | .arr ⟨l⟩ => jszL l + 2
  | .obj ⟨⟨t⟩⟩ => jszF t + 2

/-- An array's elements' weights, a comma each. -/
def jszL : List Json → ℕ
  | [] => 0
  | j :: js => jsz j + 1 + jszL js

/-- An object's fields' weights: each key's string weight, a colon and a comma, and its value's. -/
def jszF : Impl String (fun _ => Json) → ℕ
  | .leaf => 0
  | .inner _ k v l r => jszF l + (18 * k.length + 4 + jsz v) + jszF r
end

mutual
/-- **`printJ`'s bytes are at most `jsz`.** -/
theorem printJ_size_le : ∀ j, (printJ j).utf8ByteSize ≤ jsz j
  | .null => by decide +kernel
  | .bool b => by cases b <;> decide +kernel
  | .num _ => Nat.le_refl _
  | .str s => strJ_size_le s
  | .arr ⟨l⟩ => by
    have h1 := intercalate_size_le (itemsJ l)
    have h2 := itemsJ_size_le l
    have h3 : ("[" : String).utf8ByteSize = 1 := by decide +kernel
    have h4 : ("]" : String).utf8ByteSize = 1 := by decide +kernel
    rw [printJ, jsz, String.utf8ByteSize_append, String.utf8ByteSize_append]
    all_goals omega
  | .obj ⟨⟨t⟩⟩ => by
    have h1 := intercalate_size_le (fieldsJ t)
    have h2 := fieldsJ_size_le t
    have h3 : ("{" : String).utf8ByteSize = 1 := by decide +kernel
    have h4 : ("}" : String).utf8ByteSize = 1 := by decide +kernel
    rw [printJ, jsz, String.utf8ByteSize_append, String.utf8ByteSize_append]
    all_goals omega

theorem itemsJ_size_le : ∀ l : List Json, ((itemsJ l).map String.utf8ByteSize).sum + (itemsJ l).length ≤ jszL l
  | [] => by decide +kernel
  | j :: js => by
    have h1 := printJ_size_le j
    have h2 := itemsJ_size_le js
    rw [itemsJ, jszL, List.map_cons, List.sum_cons, List.length_cons]
    all_goals omega

theorem fieldsJ_size_le : ∀ t : Impl String (fun _ => Json),
    ((fieldsJ t).map String.utf8ByteSize).sum + (fieldsJ t).length ≤ jszF t
  | .leaf => by decide +kernel
  | .inner _ k v l r => by
    have h1 := fieldsJ_size_le l
    have h2 := printJ_size_le v
    have h3 := fieldsJ_size_le r
    have h4 := strJ_size_le k
    have h5 : (":" : String).utf8ByteSize = 1 := by decide +kernel
    rw [fieldsJ, jszF, List.map_append, List.map_cons, List.sum_append, List.sum_cons, List.length_append,
      List.length_cons, String.utf8ByteSize_append, String.utf8ByteSize_append]
    all_goals omega
end

theorem jszL_eq : ∀ l : List Json, jszL l = (l.map fun j => jsz j + 1).sum
  | [] => rfl
  | j :: js => by
    rw [jszL, jszL_eq js, List.map_cons, List.sum_cons]

theorem jszF_model : ∀ t : Impl String (fun _ => Json),
    jszF t = (t.toListModel.map fun e => 18 * e.1.length + 4 + jsz e.2).sum
  | .leaf => rfl
  | .inner _ k v l r => by
    rw [jszF, jszF_model l, jszF_model r, Impl.toListModel_inner, List.map_append, List.map_cons, List.sum_append,
      List.sum_cons]
    try dsimp only
    all_goals omega

/-- **An array's weight**: its elements' and a comma each, and two brackets. -/
theorem jsz_arr (a : Array Json) : jsz (.arr a) = (a.toList.map fun j => jsz j + 1).sum + 2 := by
  obtain ⟨l⟩ := a
  rw [jsz, jszL_eq]

/-- A sublist up to order of a list weighs at most what the list does. -/
theorem sum_le_of_subperm {α : Type} (f : α → ℕ) {l₁ l₂ : List α} (h : List.Subperm l₁ l₂) :
    (l₁.map f).sum ≤ (l₂.map f).sum := by
  obtain ⟨l, hp, hs⟩ := h
  rw [← (hp.map f).sum_eq]
  exact (hs.map f).sum_le_sum fun _ _ => Nat.zero_le _

open JsonRT in
/-- **An object of distinct keys weighs at most its list**: each key's string weight, a colon and a comma, and its
value's, and two braces. -/
theorem jsz_mkObj_le {l : List (String × Json)} (hd : Distinct l) :
    jsz (Json.mkObj l) ≤ (l.map fun p => 18 * p.1.length + 4 + jsz p.2).sum + 2 := by
  have hwf : (Std.TreeMap.Raw.ofList l compare).WF := Std.TreeMap.Raw.WF.ofList
  have e0 : ∀ m : Std.TreeMap.Raw String Json, jsz (.obj m) = jszF m.inner.inner + 2 := fun ⟨⟨t⟩⟩ => by
    rw [jsz]
    all_goals rfl
  have e1 : jsz (Json.mkObj l) = jszF (Std.TreeMap.Raw.ofList l compare).inner.inner + 2 := by
    unfold Json.mkObj
    exact e0 (Std.TreeMap.Raw.ofList l compare)
  have e2 : Std.TreeMap.Raw.toList (Std.TreeMap.Raw.ofList l compare) =
      (Std.TreeMap.Raw.ofList l compare).inner.inner.toListModel.map fun e => (e.1, e.2) := by
    show Impl.Const.toList (Std.TreeMap.Raw.ofList l compare).inner.inner = _
    exact Impl.Const.toList_eq_toListModel_map
  have hnd : (Std.TreeMap.Raw.toList (Std.TreeMap.Raw.ofList l compare)).Nodup := by
    show List.Pairwise (· ≠ ·) _
    refine (Std.TreeMap.Raw.distinct_keys_toList hwf).imp ?_
    intro a b hab heq
    subst heq
    exact hab Std.ReflCmp.compare_self
  have hsub : Std.TreeMap.Raw.toList (Std.TreeMap.Raw.ofList l compare) ⊆ l := by
    intro ⟨k, v⟩ hp
    have h := ((Std.TreeMap.Raw.mem_toList_iff_getKey?_eq_some_and_getElem?_eq_some hwf).1 hp).2
    rw [← Std.TreeMap.Raw.get?_eq_getElem?] at h
    exact mem_of_get?_mkObj hd h
  have hle := sum_le_of_subperm (fun p : String × Json => 18 * p.1.length + 4 + jsz p.2) (hnd.subperm hsub)
  rw [e2, List.map_map] at hle
  rw [e1, jszF_model]
  exact Nat.add_le_add_right hle 2

end FlockComplete
