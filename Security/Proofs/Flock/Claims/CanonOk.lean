module

public import Proofs.Flock.Claims.CanonEq
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# `canon` prints strings, Booleans and naturals, and arrays and objects of what it prints

`Flock.canon` refuses only a number that isn't an integer, so it prints a string, a Boolean and a natural (`canon_str`,
`canon_bool`, `canon_nat`), an array of values it prints (`canon_arr`) and an object `Json.mkObj` builds from values it
prints (`canon_mkObj`).
-/

@[expose] public section

namespace FlockDeploy

open Lean (Json toJson)

theorem canon_str (s : String) : ∃ t, Flock.canon (.str s) = .ok t := by
  simp only [Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_bool (b : Bool) : ∃ t, Flock.canon (.bool b) = .ok t := by
  simp only [Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_nat (n : Nat) : ∃ t, Flock.canon (toJson n) = .ok t := by
  simp only [toJson, Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_arr {a : Array Json} (h : ∀ x ∈ a, ∃ t, Flock.canon x = .ok t) : ∃ t, Flock.canon (.arr a) = .ok t := by
  obtain ⟨ps, hps⟩ := array_mapM_ok (f := fun x : {x // x ∈ a} => Flock.canon x.1) (a := a.attach)
    fun x _ => h x.1 x.2
  simp only [Flock.canon]
  rw [hps]
  exact ⟨_, rfl⟩

/-- Every pair of an object `Json.mkObj` builds holds a value of its list. -/
theorem mkObj_value {l : List (String × Json)} {k : String} {v : Json}
    (h : (k, v) ∈ (Std.TreeMap.Raw.ofList l compare).toArray) : ∃ k', (k', v) ∈ l := by
  rw [Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some Std.TreeMap.Raw.WF.ofList,
    Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.getElem?_insertMany_list Std.TreeMap.Raw.WF.emptyc,
    Std.TreeMap.Raw.getElem?_emptyc, Option.or_none, List.findSomeRev?_eq_findSome?_reverse] at h
  obtain ⟨⟨k', v'⟩, hm, he⟩ := List.exists_of_findSome?_eq_some h
  by_cases hc : compare k' k = .eq
  · simp only [hc, ↓reduceIte, Option.some.injEq] at he
    exact ⟨k', he ▸ List.mem_reverse.1 hm⟩
  · simp [hc] at he

theorem canon_mkObj {l : List (String × Json)} (h : ∀ p ∈ l, ∃ t, Flock.canon p.2 = .ok t) :
    ∃ t, Flock.canon (Json.mkObj l) = .ok t := by
  obtain ⟨ps, hps⟩ := array_mapM_ok
    (f := fun p : {x // x ∈ (Std.TreeMap.Raw.ofList l compare).toArray} =>
      (do pure (p.1.1, ← Flock.canon p.1.2) : Flock.V (String × String)))
    (a := (Std.TreeMap.Raw.ofList l compare).toArray.attach) fun ⟨⟨k, v⟩, hp⟩ _ => by
      obtain ⟨k', hk'⟩ := mkObj_value hp
      obtain ⟨t, ht⟩ := h _ hk'
      exact ⟨(k, t), by simp only [ht]; rfl⟩
  simp only [Json.mkObj, Flock.canon]
  rw [hps]
  exact ⟨_, rfl⟩

end FlockDeploy
