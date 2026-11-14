module

public import Batteries.Data.List.Basic
public import Std.Data.TreeMap.Raw.WF
public import Proofs.Properties.Pouw.LiveSalt.Canon
public import Proofs.Properties.Accounting.Remote.Sort
public import Proofs.Flock.Claims.JsonEq
public import Proofs.Core.QSort
public import Proofs.Flock.Claims.CanonRead
import all Init.Data.Repr
import all Init.Data.Int.Repr
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-!
# Field reads of canonical JSON

`Proofs.Flock.Claims.CanonRead` has that `Flock.canonUtf8`'s text reads back (`canon_inj`) and that a field read of a
sorted object is a lookup of the field in its fields' list (`get?_lookup`). So, of `Spot.getD`: equivalent sorted values
have equivalent fields (`getD_equiv`), every field of a sorted value is sorted (`sorted_getD`), and an entry without its
`sig` (`Core.Record.unsigned`) is sorted and has the entry's other fields (`sorted_unsigned`, `getD_unsigned`).
-/

@[expose] public section

namespace Properties.Challenger.CanonRead

open Lean (Json JsonNumber)
open Spot (sortedKeys increasing getD field?)
open FlockDeploy (JsonEquiv jsonForm jsonForms jsonForms_eq jsonForm_obj FieldForms JsonForm JsonForms)
open Properties.Pouw.LiveSalt.Canon (canon_arr canon_obj canonStr_split canonStr_toList nat_split)
open Std.DTreeMap.Internal (Impl)

/-! ## The texts `canonUtf8` prints -/

theorem toList_lit (s : String) (h : s = "null" ∨ s = "true" ∨ s = "false") :
    s.toList = ['n', 'u', 'l', 'l'] ∨ s.toList = ['t', 'r', 'u', 'e'] ∨ s.toList = ['f', 'a', 'l', 's', 'e'] := by
  rcases h with rfl | rfl | rfl
  · exact .inl rfl
  · exact .inr (.inl rfl)
  · exact .inr (.inr rfl)

/-! ## Field reads -/

theorem sortedKeys_null : sortedKeys .null = true := by
  unfold sortedKeys
  rfl

theorem getD_obj (m : Std.TreeMap.Raw String Json) (k : String) : getD (.obj m) k = (m.get? k).getD .null := by
  unfold getD field? Json.getObjVal?
  dsimp only
  cases m.get? k <;> rfl

theorem getD_of_not_obj {a : Json} (h : ¬ ∃ m, a = .obj m) (k : String) : getD a k = .null := by
  cases a with
  | obj m => exact absurd ⟨m, rfl⟩ h
  | _ => rfl

theorem getD_lookup {m : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj m) = true) (k : String) :
    getD (.obj m) k = (m.toList.lookup k).getD .null := by
  rw [getD_obj, get?_lookup (sorted_toList h)]

/-- **Equivalent sorted values have equivalent fields.** -/
theorem getD_equiv {a b : Json} (ha : sortedKeys a = true) (hb : sortedKeys b = true) (h : JsonEquiv a b)
    (k : String) : JsonEquiv (getD a k) (getD b k) := by
  by_cases hA : ∃ m, a = .obj m
  · obtain ⟨m, rfl⟩ := hA
    obtain ⟨n, rfl⟩ := form_obj (Eq.trans (Eq.symm h) (jsonForm_obj m))
    have hf := h
    rw [JsonEquiv, jsonForm_obj, jsonForm_obj, JsonForm.obj.injEq, FieldForms.ofList_inj] at hf
    have hl := lookup_forms hf k
    rw [getD_lookup ha, getD_lookup hb]
    revert hl
    cases m.toList.lookup k <;> cases n.toList.lookup k <;> intro hl <;> simp only [Option.map_none,
      Option.map_some, reduceCtorEq, Option.some.injEq] at hl
    · exact JsonEquiv.refl _
    · exact hl
  · have hB : ¬ ∃ n, b = .obj n := fun ⟨n, hn⟩ => hA (form_obj (Eq.trans h (hn ▸ jsonForm_obj n)))
    rw [getD_of_not_obj hA, getD_of_not_obj hB]
    exact JsonEquiv.refl _

/-- **Every field of a sorted value is sorted.** -/
theorem sorted_getD {a : Json} (ha : sortedKeys a = true) (k : String) : sortedKeys (getD a k) = true := by
  by_cases hA : ∃ m, a = .obj m
  · obtain ⟨m, rfl⟩ := hA
    rw [getD_lookup ha]
    cases hl : m.toList.lookup k with
    | none => exact sortedKeys_null
    | some v =>
      have hm := lookup_toList hl
      rw [← Std.TreeMap.Raw.toList_toArray] at hm
      exact (sortedKeys_obj.1 ha).2 _ (Array.mem_toList_iff.1 hm)
  · rw [getD_of_not_obj hA]
    exact sortedKeys_null

theorem equiv_bool {x : Json} {b : Bool} (h : JsonEquiv x (.bool b)) : x = .bool b := by
  cases x with
  | bool c => cases h; rfl
  | arr a => rw [JsonEquiv, jsonForm_arr] at h; cases h
  | obj m => rw [JsonEquiv, jsonForm_obj] at h; cases h
  | _ => cases h

/-! ## An entry without its `sig` -/

theorem unsigned_obj (kvs : Std.TreeMap.Raw String Json) :
    Core.Record.unsigned (.obj kvs) = .obj (Std.TreeMap.Raw.ofList (kvs.toList.filter (·.1 != "sig")) compare) := by
  rw [← Std.TreeMap.Raw.toList_toArray]
  rfl

theorem filter_keys {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) :
    ((l.filter (·.1 != "sig")).map (·.1)).Pairwise (· < ·) := by
  rw [List.pairwise_map] at h ⊢
  exact h.sublist List.filter_sublist

theorem distinct_of_keys {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) :
    l.Pairwise (fun a b => ¬ compare a.1 b.1 = .eq) := by
  rw [List.pairwise_map] at h
  exact h.imp fun {a b} hab he => by
    rw [(Std.LawfulEqCmp.compare_eq_iff_eq (cmp := compare)).1 he] at hab
    exact String.lt_irrefl _ hab

theorem lookup_none {l : List (String × Json)} {k : String} (h : l.lookup k = none) :
    (l.map Prod.fst).contains k = false := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hk : k = k'
    · subst hk
      simp [List.lookup] at h
    · simp only [List.lookup, beq_eq_false_iff_ne.2 hk] at h
      simp only [List.map_cons, List.contains_cons, ih h, Bool.or_false]
      exact beq_eq_false_iff_ne.2 hk

theorem get?_ofList {l : List (String × Json)} (h : (l.map (·.1)).Pairwise (· < ·)) (k : String) :
    (Std.TreeMap.Raw.ofList l compare).get? k = l.lookup k := by
  cases hl : l.lookup k with
  | some v =>
    exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl) (distinct_of_keys h) (lookup_toList hl)
  | none => exact Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false (lookup_none hl)

theorem lookup_filter {l : List (String × Json)} {k : String} (hk : k ≠ "sig") :
    (l.filter (·.1 != "sig")).lookup k = l.lookup k := by
  induction l with
  | nil => rfl
  | cons p l ih =>
    obtain ⟨k', v'⟩ := p
    by_cases hs : k' = "sig"
    · subst hs
      simp only [List.filter, bne_self_eq_false, List.lookup, beq_eq_false_iff_ne.2 hk, ih]
    · have hs' : (k' != "sig") = true := bne_iff_ne.2 hs
      by_cases hk' : k = k'
      · subst hk'
        simp only [List.filter, hs', List.lookup, beq_self_eq_true]
      · simp only [List.filter, hs', List.lookup, beq_eq_false_iff_ne.2 hk', ih]

/-- **An entry without its `sig` has the entry's other fields.** -/
theorem getD_unsigned {kvs : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj kvs) = true) {k : String}
    (hk : k ≠ "sig") : getD (Core.Record.unsigned (.obj kvs)) k = getD (.obj kvs) k := by
  rw [unsigned_obj, getD_obj, get?_ofList (filter_keys (sorted_toList h)), getD_lookup h, lookup_filter hk]

/-- **An entry without its `sig` is sorted** when the entry is. -/
theorem sorted_unsigned {kvs : Std.TreeMap.Raw String Json} (h : sortedKeys (.obj kvs) = true) :
    sortedKeys (Core.Record.unsigned (.obj kvs)) = true := by
  have hL := filter_keys (sorted_toList h)
  have hwf : (Std.TreeMap.Raw.ofList (kvs.toList.filter (·.1 != "sig")) compare).WF := Std.TreeMap.Raw.WF.ofList
  rw [unsigned_obj, sortedKeys_obj, increasing_iff, Std.TreeMap.Raw.toList_toArray]
  refine ⟨?_, fun p hp => ?_⟩
  · rw [List.pairwise_map]
    exact (Std.TreeMap.Raw.ordered_keys_toList hwf).imp fun hab => compareOfLessAndEq_eq_lt.1 hab
  · have hp' := Array.mem_toList_iff.2 hp
    rw [Std.TreeMap.Raw.toList_toArray] at hp'
    have hg := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hwf).1 hp'
    rw [← Std.TreeMap.Raw.get?_eq_getElem?, get?_ofList hL] at hg
    have hm := List.mem_filter.1 (lookup_toList hg)
    rw [← Std.TreeMap.Raw.toList_toArray] at hm
    exact (sortedKeys_obj.1 h).2 _ (Array.mem_toList_iff.1 hm.1)

end Properties.Challenger.CanonRead
