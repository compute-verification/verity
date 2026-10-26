import Flock.Canon
import Flock.Norm
import Std.Data.TreeMap.Raw.WF
import Definitions.Claims.Json
import Proofs.Flock.Claims.JsonEq

/-!
# The rebuilt draw: one value per equivalence class

`Json.parse` builds an object's tree in the order its keys arrived, and no proof sees into it (`parse` is a `partial
def`), so two draws that `drawEq` passes on (`JsonEquiv`: the same fields, whatever their order) can be different terms
(`FlockDeploy.KS.drawEq_ne`). Where the verifier keeps a draw, the record's (`Exec.recordDraw`) and the statement's
(`Raw.ins`), it keeps `Flock.norm` of what it parsed: every object rebuilt by inserting its fields into an empty map,
and that map's fields into another. So:

* `norm_congr`: `norm` is the same on equivalent values;
* `norm_norm`: `norm` is idempotent, so `eq_of_jsonEquiv_norm`, `eq_of_rel_norm`: two rebuilt values the draw check
  passes on are equal, with no premise on the parser;
* `wf_norm`: a rebuilt value is well formed (`Claims.Json.Wf`) by construction;
* `jsonEquiv_norm`: rebuilding a well-formed value keeps its form, which completeness reads (the honest record's draw,
  parsed, is the statement's up to `JsonEquiv`).
-/

namespace FlockDeploy

open Lean (Json JsonNumber)
open Std.DTreeMap.Internal (Impl)
open Flock (normList normFields normObj)

theorem normList_eq (a : List Json) : normList a = a.map Flock.norm := by
  induction a with
  | nil => rfl
  | cons a as ih => simp [normList, ih]

theorem normFields_eq (t : Impl String (fun _ => Json)) :
    normFields t = (Impl.Const.toList t).map fun p => (p.1, Flock.norm p.2) := by
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map]
  induction t with
  | leaf => rfl
  | inner _ k v l r ihl ihr => simp [normFields, ihl, ihr]

theorem norm_arr (a : Array Json) : Flock.norm (.arr a) = .arr (a.map Flock.norm) := by
  obtain ⟨a⟩ := a
  simp [Flock.norm, normList_eq]

theorem norm_obj (m : Std.TreeMap.Raw String Json compare) :
    Flock.norm (.obj m) = .obj (normObj (m.toList.map fun p => (p.1, Flock.norm p.2))) := by
  obtain ⟨⟨t⟩⟩ := m
  simp only [Flock.norm, normFields_eq]
  rfl

theorem norm_null : Flock.norm .null = .null := rfl
theorem norm_bool (b : Bool) : Flock.norm (.bool b) = .bool b := rfl
theorem norm_num (n : JsonNumber) : Flock.norm (.num n) = .num n := rfl
theorem norm_str (s : String) : Flock.norm (.str s) = .str s := rfl

theorem sizeOf_lt_of_mem_toList {m : Std.TreeMap.Raw String Json compare} {p : String × Json}
    (h : p ∈ m.toList) : sizeOf p.2 < sizeOf (Json.obj m) :=
  Flock.sizeOf_lt_of_mem_toArray (by rw [← Array.mem_toList_iff, Std.TreeMap.Raw.toList_toArray]; exact h)

theorem sizeOf_lt_of_mem_arr {a : Array Json} {x : Json} (h : x ∈ a) : sizeOf x < sizeOf (Json.arr a) := by
  have := Array.sizeOf_lt_of_mem h
  simp only [Json.arr.sizeOf_spec]
  omega

/-- Induction on `Json` through arrays' elements and objects' values. -/
theorem json_ind {P : Json → Prop} (null : P .null) (bool : ∀ b, P (.bool b)) (num : ∀ n, P (.num n))
    (str : ∀ s, P (.str s)) (arr : ∀ a : Array Json, (∀ x ∈ a, P x) → P (.arr a))
    (obj : ∀ m : Std.TreeMap.Raw String Json compare, (∀ p ∈ m.toList, P p.2) → P (.obj m)) : ∀ j, P j
  | .null => null
  | .bool b => bool b
  | .num n => num n
  | .str s => str s
  | .arr a => arr a fun x _ => json_ind null bool num str arr obj x
  | .obj m => obj m fun p _ => json_ind null bool num str arr obj p.2
termination_by j => sizeOf j
decreasing_by
  · exact sizeOf_lt_of_mem_arr ‹_›
  · exact sizeOf_lt_of_mem_toList ‹_›

/-! ## The rebuilt object -/

theorem wf_normObj (l : List (String × Json)) : (normObj l).WF := Std.TreeMap.Raw.WF.ofList

/-- A well-formed map's fields, inserted into an empty map, are its own fields. -/
theorem toList_ofList_toList {t : Std.TreeMap.Raw String Json compare} (ht : t.WF) :
    (Std.TreeMap.Raw.ofList t.toList compare).toList = t.toList := by
  refine (Std.TreeMap.Raw.equiv_iff_toList_eq Std.TreeMap.Raw.WF.ofList ht).1 ?_
  refine Std.TreeMap.Raw.Equiv.of_forall_getElem?_eq Std.TreeMap.Raw.WF.ofList ht fun k => ?_
  cases hk : t[k]? with
  | some v =>
    exact Std.TreeMap.Raw.getElem?_ofList_of_mem (Std.ReflCmp.compare_self) (Std.TreeMap.Raw.distinct_keys_toList ht)
      ((Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some ht).2 hk)
  | none =>
    refine Std.TreeMap.Raw.getElem?_ofList_of_contains_eq_false ?_
    rw [Std.TreeMap.Raw.map_fst_toList_eq_keys, Std.TreeMap.Raw.contains_keys ht]
    rw [← Std.TreeMap.Raw.isSome_getElem?_eq_contains ht, hk]
    rfl

theorem toList_normObj (l : List (String × Json)) :
    (normObj l).toList = (Std.TreeMap.Raw.ofList l compare).toList :=
  toList_ofList_toList Std.TreeMap.Raw.WF.ofList

/-- **Rebuilding a rebuilt object's fields gives it back.** -/
theorem normObj_toList_normObj (l : List (String × Json)) : normObj (normObj l).toList = normObj l := by
  unfold normObj
  rw [toList_ofList_toList (Std.TreeMap.Raw.WF.ofList), toList_ofList_toList (Std.TreeMap.Raw.WF.ofList)]

theorem wf_foldl_insert {l : List (String × Json)} {t : Std.TreeMap.Raw String Json compare} (ht : t.WF) :
    (l.foldl (init := t) fun acc p => acc.insert p.1 p.2).WF := by
  induction l generalizing t with
  | nil => exact ht
  | cons p l ih => exact ih (Std.TreeMap.Raw.WF.insert ht)

theorem foldl_insert_getElem? {l : List (String × Json)} {t : Std.TreeMap.Raw String Json compare} (ht : t.WF)
    {k : String} {w : Json} (h : (l.foldl (init := t) fun acc p => acc.insert p.1 p.2)[k]? = some w) :
    t[k]? = some w ∨ ∃ k', (k', w) ∈ l := by
  induction l generalizing t with
  | nil => exact .inl h
  | cons p l ih =>
    rcases ih (Std.TreeMap.Raw.WF.insert ht) h with h | ⟨k', hk⟩
    · rw [Std.TreeMap.Raw.getElem?_insert ht] at h
      split at h
      · cases h
        exact .inr ⟨p.1, List.mem_cons_self⟩
      · exact .inl h
    · exact .inr ⟨k', List.mem_cons_of_mem _ hk⟩

/-- **A rebuilt object's values are the fields' values.** -/
theorem mem_normObj {l : List (String × Json)} {p : String × Json} (h : p ∈ (normObj l).toList) :
    ∃ k, (k, p.2) ∈ l := by
  rw [toList_normObj] at h
  have h' := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some (k := p.1) (v := p.2) Std.TreeMap.Raw.WF.ofList).1 h
  rw [Std.TreeMap.Raw.Equiv.getElem?_eq Std.TreeMap.Raw.WF.ofList
    (wf_foldl_insert Std.TreeMap.Raw.WF.emptyc) Std.TreeMap.Raw.ofList_equiv_foldl] at h'
  rcases foldl_insert_getElem? Std.TreeMap.Raw.WF.emptyc h' with h | h
  · simp at h
  · exact h

/-! ## `norm` -/

/-- **`norm` is the same on equivalent values.** -/
theorem norm_congr : ∀ a b : Json, JsonEquiv a b → Flock.norm a = Flock.norm b := by
  refine json_ind ?_ ?_ ?_ ?_ ?_ ?_
  · intro b h; cases b <;> first | rfl | (simp [JsonEquiv, jsonForm] at h)
  · intro c b h; cases b <;> simp [JsonEquiv, jsonForm] at h; subst h; rfl
  · intro n b h; cases b <;> simp [JsonEquiv, jsonForm] at h; subst h; rfl
  · intro s b h; cases b <;> simp [JsonEquiv, jsonForm] at h; subst h; rfl
  · intro a ih b h
    cases b with
    | arr b =>
      obtain ⟨hs, he⟩ := (jsonEquiv_arr_iff a b).1 h
      rw [norm_arr, norm_arr]
      congr 1
      apply Array.ext (by simpa using hs)
      intro i h₁ h₂
      simp only [Array.getElem_map]
      exact ih _ (Array.getElem_mem _) _ (he i (by simpa using h₁) (by simpa using h₂))
    | obj n => simp [JsonEquiv, jsonForm] at h
    | _ => simp [JsonEquiv, jsonForm] at h
  · intro m ih b h
    cases b with
    | obj n =>
      rw [JsonEquiv, jsonForm_obj, jsonForm_obj, JsonForm.obj.injEq, FieldForms.ofList_inj] at h
      rw [norm_obj, norm_obj]
      congr 2
      generalize m.toList = L at ih h ⊢
      generalize n.toList = L' at h
      induction L generalizing L' with
      | nil => cases L' <;> simp_all
      | cons p L ihL =>
        cases L' with
        | nil => simp at h
        | cons q L' =>
          simp only [List.map_cons, List.cons.injEq, Prod.mk.injEq] at h ⊢
          exact ⟨⟨h.1.1, ih p List.mem_cons_self _ h.1.2⟩,
            ihL (fun p' hp' => ih p' (List.mem_cons_of_mem _ hp')) _ h.2⟩
    | arr b => simp [JsonEquiv, jsonForm] at h
    | _ => simp [JsonEquiv, jsonForm] at h

/-- **`norm` is idempotent.** -/
theorem norm_norm : ∀ a : Json, Flock.norm (Flock.norm a) = Flock.norm a := by
  refine json_ind rfl (fun _ => rfl) (fun _ => rfl) (fun _ => rfl) ?_ ?_
  · intro a ih
    rw [norm_arr, norm_arr, Array.map_map]
    congr 1
    exact Array.map_congr_left fun x hx => ih x hx
  · intro m ih
    rw [norm_obj, norm_obj]
    congr 1
    refine Eq.trans ?_ (normObj_toList_normObj _)
    congr 1
    refine Eq.trans ?_ (List.map_id _)
    refine List.map_congr_left fun p hp => ?_
    obtain ⟨k, hk⟩ := mem_normObj hp
    obtain ⟨q, hq, he⟩ := List.mem_map.1 hk
    simp only [Prod.mk.injEq] at he
    show (p.1, Flock.norm p.2) = p
    rw [← he.2, ih q hq, he.2]

/-- **Rebuilt values are equivalent exactly when equal.** -/
theorem norm_eq_of_jsonEquiv {a b : Json} (h : JsonEquiv (Flock.norm a) (Flock.norm b)) :
    Flock.norm a = Flock.norm b := by
  rw [← norm_norm a, ← norm_norm b]
  exact norm_congr _ _ h

/-- **Fixed points of `norm` that are equivalent are equal.** -/
theorem eq_of_jsonEquiv_norm {a b : Json} (ha : Flock.norm a = a) (hb : Flock.norm b = b) (h : JsonEquiv a b) :
    a = b := by
  rw [← ha, ← hb]
  exact norm_congr _ _ h

/-- **Two rebuilt optional values the draw check passes on are equal.** -/
theorem eq_of_rel_norm {a b : Option Json} (h : Option.Rel JsonEquiv a b) (ha : ∀ d, a = some d → Flock.norm d = d)
    (hb : ∀ d, b = some d → Flock.norm d = d) : a = b := by
  cases h with
  | none => rfl
  | some h => exact congrArg some (eq_of_jsonEquiv_norm (ha _ rfl) (hb _ rfl) h)

/-- **A rebuilt value is well formed**: every object in it is a tree built by insertion into an empty map. -/
theorem wf_norm : ∀ j : Json, Claims.Json.Wf (Flock.norm j) := by
  refine json_ind .null .bool .num .str ?_ ?_
  · intro a ih
    rw [norm_arr]
    refine .arr fun x hx => ?_
    obtain ⟨y, hy, rfl⟩ := Array.mem_map.1 hx
    exact ih y hy
  · intro m ih
    rw [norm_obj]
    refine .obj (wf_normObj _) fun k v hv => ?_
    have hp := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some (wf_normObj _)).2 hv
    obtain ⟨k', hk⟩ := mem_normObj hp
    obtain ⟨q, hq, he⟩ := List.mem_map.1 hk
    simp only [Prod.mk.injEq] at he
    rw [← he.2]
    exact ih q hq

/-- **On a well-formed value, `norm` keeps the form**: a well-formed object's fields are already in key order with one
value per key, so rebuilding it gives the same fields. -/
theorem jsonEquiv_norm : ∀ j : Json, Claims.Json.Wf j → JsonEquiv (Flock.norm j) j := by
  refine json_ind (fun _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl) ?_ ?_
  · intro a ih hw
    cases hw with
    | arr h =>
      rw [norm_arr, jsonEquiv_arr_iff]
      refine ⟨by simp, fun i h₁ h₂ => ?_⟩
      simp only [Array.getElem_map]
      exact ih _ (Array.getElem_mem _) (h _ (Array.getElem_mem _))
  · intro m ih hw
    cases hw with
    | obj hm h =>
      have hM : (m.map fun _ v => Flock.norm v).WF := ⟨⟨Std.DTreeMap.Internal.Impl.WF.map hm.out.out⟩⟩
      have hL : m.toList.map (fun p => (p.1, Flock.norm p.2)) = (m.map fun _ v => Flock.norm v).toList :=
        (Std.TreeMap.Raw.toList_map (f := fun _ v => Flock.norm v)).symm
      rw [norm_obj, JsonEquiv, jsonForm_obj, jsonForm_obj, toList_normObj, hL, toList_ofList_toList hM, ← hL,
        List.map_map]
      congr 2
      refine List.map_congr_left fun p hp => ?_
      have hv : m.get? p.1 = some p.2 := (Std.TreeMap.Raw.mem_toList_iff_getElem?_eq_some hm).1 hp
      simp only [Function.comp_apply]
      rw [ih p hp (h _ _ hv)]

/-- **Alike values are well formed** on the right (`Claims.Json.Same` holds only of well-formed trees). -/
theorem same_wf_right {a b : Json} (h : Claims.Json.Same a b) : Claims.Json.Wf b := by
  induction h with
  | null => exact .null
  | bool b => exact .bool b
  | num n => exact .num n
  | str s => exact .str s
  | arr hs _ ih =>
    refine .arr fun x hx => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hx
    exact ih i (hs ▸ hi) hi
  | obj hm hn hk _ ih =>
    refine .obj hn fun k v hv => ?_
    have hs := hk k
    rw [hv, Option.isSome_some] at hs
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.1 hs
    exact ih k a v ha hv

end FlockDeploy
