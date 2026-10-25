import Proofs.Flock.Claims.CanonEq
import Definitions.Claims.Json

/-!
# Lookups agree on equivalent values

`JsonEquiv` fixes an object's fields in the order of its tree. A lookup (`Std.TreeMap.Raw.get?`) searches the tree by
key, so on an ordered tree it finds what the field list holds (`Impl.Const.get?_eq_getValue?`), whatever the tree's
shape; and ordered is a property of that list, so an object equivalent to a well-formed one is ordered too
(`ordered_of_keys`). On a well-formed value (`Claims.Json.Wf`: every object a well-formed tree), an equivalent value's
lookups find equivalent values (`get?_jsonEquiv`, `getObjVal?_jsonEquiv`). Without well-formedness they need not: two
unordered trees with one field list can search to different fields.
-/

namespace FlockDeploy

open Lean (Json)
open Std.DTreeMap.Internal (Impl)
open Std.Internal.List (getValue?)

/-- The fields' lookups of two field lists that agree key by key, with equivalent values, are equivalent. -/
theorem getValue?_forall₂ (k : String) : ∀ {l₁ l₂ : List ((_ : String) × Json)},
    List.Forall₂ (fun p q => p.1 = q.1 ∧ JsonEquiv p.2 q.2) l₁ l₂ →
      Option.Rel JsonEquiv (getValue? k l₁) (getValue? k l₂)
  | [], [], .nil => .none
  | ⟨k₁, v₁⟩ :: _, ⟨k₂, v₂⟩ :: _, .cons ⟨hk, hv⟩ h => by
    simp only at hk hv
    subst hk
    simp only [Std.Internal.List.getValue?_cons]
    cases k₁ == k
    · exact getValue?_forall₂ k h
    · exact .some hv

/-- A tree whose keys, in order, are an ordered tree's is ordered. -/
theorem ordered_of_keys {t u : Impl String fun _ => Json} (ht : t.Ordered)
    (h : t.toListModel.map (·.1) = u.toListModel.map (·.1)) : u.Ordered := by
  unfold Impl.Ordered at *
  have := List.pairwise_map.2 ht
  rw [h] at this
  exact List.pairwise_map.1 this

theorem get?_eq_getValue? {t : Impl String fun _ => Json} (h : t.Ordered) (k : String) :
    (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json compare).get? k = getValue? k t.toListModel :=
  Impl.Const.get?_eq_getValue? h

/-- **An equivalent object's lookups**: where one of two equivalent objects is a well-formed tree, both find, at every
key, equivalent values or nothing. -/
theorem get?_jsonEquiv {m n : Std.TreeMap.Raw String Json compare} (hm : m.WF) (h : JsonEquiv (.obj m) (.obj n))
    (k : String) : Option.Rel JsonEquiv (m.get? k) (n.get? k) := by
  obtain ⟨⟨t⟩⟩ := m
  obtain ⟨⟨u⟩⟩ := n
  have ht : t.Ordered := hm.out.out.ordered
  obtain ⟨hl, he⟩ := jsonEquiv_obj_toList h
  have hlt : (Impl.Const.toList t) = t.toListModel.map fun p => (p.1, p.2) := Impl.Const.toList_eq_toListModel_map
  have hlu : (Impl.Const.toList u) = u.toListModel.map fun p => (p.1, p.2) := Impl.Const.toList_eq_toListModel_map
  change (Impl.Const.toList t).length = (Impl.Const.toList u).length at hl
  change ∀ i (h₁ : i < (Impl.Const.toList t).length) (h₂ : i < (Impl.Const.toList u).length),
    (Impl.Const.toList t)[i].1 = (Impl.Const.toList u)[i].1 ∧
      JsonEquiv (Impl.Const.toList t)[i].2 (Impl.Const.toList u)[i].2 at he
  simp only [hlt, hlu, List.length_map, List.getElem_map] at hl he
  have hf : List.Forall₂ (fun p q => p.1 = q.1 ∧ JsonEquiv p.2 q.2) t.toListModel u.toListModel :=
    List.forall₂_iff_get.2 ⟨hl, fun i h₁ h₂ => he i h₁ h₂⟩
  have hk : t.toListModel.map (·.1) = u.toListModel.map (·.1) := by
    apply List.ext_getElem (by simpa using hl)
    intro i h₁ h₂
    simp only [List.getElem_map]
    exact (he i (by simpa using h₁) (by simpa using h₂)).1
  rw [get?_eq_getValue? ht, get?_eq_getValue? (ordered_of_keys ht hk)]
  exact getValue?_forall₂ k hf

/-- **`getObjVal?` on equivalent values**, the first well-formed: the same refusal, or equivalent values, the first's
well-formed. -/
theorem getObjVal?_jsonEquiv {a b : Json} (hw : Claims.Json.Wf a) (h : JsonEquiv a b) (k : String) :
    (∃ e, a.getObjVal? k = .error e ∧ b.getObjVal? k = .error e) ∨
      ∃ v w, a.getObjVal? k = .ok v ∧ b.getObjVal? k = .ok w ∧ Claims.Json.Wf v ∧ JsonEquiv v w := by
  cases hw with
  | null => rw [jsonEquiv_null h]; exact .inl ⟨_, rfl, rfl⟩
  | bool x => rw [jsonEquiv_bool h]; exact .inl ⟨_, rfl, rfl⟩
  | num x => rw [jsonEquiv_num h]; exact .inl ⟨_, rfl, rfl⟩
  | str x => rw [jsonEquiv_str h]; exact .inl ⟨_, rfl, rfl⟩
  | arr _ =>
    obtain ⟨b', rfl⟩ := jsonEquiv_arr h
    exact .inl ⟨_, rfl, rfl⟩
  | @obj m hm hv =>
    obtain ⟨n, rfl⟩ := jsonEquiv_obj h
    have hr := get?_jsonEquiv hm h k
    simp only [Json.getObjVal?]
    cases hmk : m.get? k <;> cases hnk : n.get? k <;> rw [hmk, hnk] at hr
    · exact .inl ⟨_, rfl, rfl⟩
    · cases hr
    · cases hr
    · rename_i v w
      cases hr with
      | some hvw => exact .inr ⟨v, w, rfl, rfl, hv k v hmk, hvw⟩

end FlockDeploy
