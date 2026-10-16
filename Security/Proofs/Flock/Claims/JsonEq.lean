import Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas

/-!
# `Json` equivalence Lean can reason through

Lean's `==` on `Json` is a `partial def` (opaque to proofs), so a check by it establishes nothing. `JsonEquiv` compares
objects as the service's draw equality does (Python's dict equality: the same keys with equivalent values, whatever
their order) and arrays elementwise in order (`jsonEquiv_obj_iff`, on objects whose trees are well formed, as
`Json.parse` builds them; `jsonEquiv_arr_iff`), recursively; scalars are equal as values (a number as its mantissa and
exponent, as `canon` reads the integers it accepts). It is equality of forms (`jsonForm`: an object as its fields in
the order of its tree, key order in a well-formed one), so `drawEq`, `JsonEquiv` on optional values, decides it
(`drawEq_iff`).
-/

namespace FlockDeploy

open Lean (Json JsonNumber)
open Std.DTreeMap.Internal (Impl)

mutual
/-- A `Json` value's form: an array as its elements' forms, an object as its fields with their values' forms. -/
inductive JsonForm where
  | null
  | bool (b : Bool)
  | num (n : JsonNumber)
  | str (s : String)
  | arr (a : JsonForms)
  | obj (fs : FieldForms)
  deriving DecidableEq

/-- A list of forms. -/
inductive JsonForms where
  | nil
  | cons (a : JsonForm) (as : JsonForms)
  deriving DecidableEq

/-- A list of fields, each a key and its value's form. -/
inductive FieldForms where
  | nil
  | cons (k : String) (v : JsonForm) (fs : FieldForms)
  deriving DecidableEq
end

/-- The forms of a list. -/
def JsonForms.ofList : List JsonForm → JsonForms
  | [] => .nil
  | a :: as => .cons a (ofList as)

/-- The fields of a list. -/
def FieldForms.ofList : List (String × JsonForm) → FieldForms
  | [] => .nil
  | (k, v) :: fs => .cons k v (ofList fs)

theorem JsonForms.ofList_inj {a b : List JsonForm} : ofList a = ofList b ↔ a = b := by
  induction a generalizing b with
  | nil => cases b <;> simp [ofList]
  | cons a as ih => cases b <;> simp [ofList, ih]

theorem FieldForms.ofList_inj {a b : List (String × JsonForm)} : ofList a = ofList b ↔ a = b := by
  induction a generalizing b with
  | nil => cases b <;> simp [ofList]
  | cons p as ih =>
    cases b with
    | nil => simp [ofList]
    | cons q bs => obtain ⟨k, v⟩ := p; obtain ⟨k', v'⟩ := q; simp [ofList, ih, and_assoc]

mutual
/-- **A value's form**: an object's fields in the order of its tree (`jsonForm_obj`: `toList`'s, key order in a
well-formed tree). -/
def jsonForm : Json → JsonForm
  | .null => .null
  | .bool b => .bool b
  | .num n => .num n
  | .str s => .str s
  | .arr ⟨a⟩ => .arr (.ofList (jsonForms a))
  | .obj ⟨⟨t⟩⟩ => .obj (.ofList (fieldForms t))

/-- `jsonForm` elementwise. -/
def jsonForms : List Json → List JsonForm
  | [] => []
  | a :: as => jsonForm a :: jsonForms as

/-- An object's fields with their values' forms, in the order of its tree. -/
def fieldForms : Impl String (fun _ => Json) → List (String × JsonForm)
  | .leaf => []
  | .inner _ k v l r => fieldForms l ++ (k, jsonForm v) :: fieldForms r
end

/-- **Equivalent values**: equal forms. -/
def JsonEquiv (a b : Json) : Prop := jsonForm a = jsonForm b

instance instDecidableRelJsonEquiv : DecidableRel JsonEquiv := fun a b =>
  inferInstanceAs (Decidable (jsonForm a = jsonForm b))

theorem JsonEquiv.refl (a : Json) : JsonEquiv a a := rfl

theorem JsonEquiv.symm {a b : Json} (h : JsonEquiv a b) : JsonEquiv b a := Eq.symm h

theorem JsonEquiv.trans {a b c : Json} (h₁ : JsonEquiv a b) (h₂ : JsonEquiv b c) : JsonEquiv a c := Eq.trans h₁ h₂

theorem jsonForms_eq (a : List Json) : jsonForms a = a.map jsonForm := by
  induction a with
  | nil => rfl
  | cons a as ih => simp [jsonForms, ih]

theorem fieldForms_eq (t : Impl String (fun _ => Json)) :
    fieldForms t = (Impl.Const.toList t).map fun p => (p.1, jsonForm p.2) := by
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map]
  induction t with
  | leaf => rfl
  | inner _ k v l r ihl ihr => simp [fieldForms, ihl, ihr]

/-- An object's form is its fields' (`toList`'s) with their values' forms. -/
theorem jsonForm_obj (m : Std.TreeMap.Raw String Json compare) :
    jsonForm (.obj m) = .obj (.ofList (m.toList.map fun p => (p.1, jsonForm p.2))) := by
  obtain ⟨⟨t⟩⟩ := m
  simp only [jsonForm, fieldForms_eq]
  rfl

/-- **Arrays are equivalent elementwise, in order.** -/
theorem jsonEquiv_arr_iff (a b : Array Json) :
    JsonEquiv (.arr a) (.arr b) ↔
      a.size = b.size ∧ ∀ i (h₁ : i < a.size) (h₂ : i < b.size), JsonEquiv a[i] b[i] := by
  obtain ⟨a⟩ := a
  obtain ⟨b⟩ := b
  simp only [JsonEquiv, jsonForm, JsonForm.arr.injEq, JsonForms.ofList_inj, jsonForms_eq]
  constructor
  · intro h
    refine ⟨by simpa using congrArg List.length h, fun i h₁ h₂ => ?_⟩
    have := congrArg (·[i]?) h
    simpa [List.getElem?_eq_getElem h₁, List.getElem?_eq_getElem h₂] using this
  · rintro ⟨hl, h⟩
    apply List.ext_getElem (by simpa using hl)
    intro i h₁ h₂
    rw [List.getElem_map, List.getElem_map]
    exact h i (by simpa using h₁) (by simpa using h₂)

theorem wf_mapForm {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) (f : String → Json → JsonForm) :
    (m.map f).WF :=
  ⟨⟨Impl.WF.map hm.out.out⟩⟩

theorem toList_mapForm (m : Std.TreeMap.Raw String Json compare) :
    m.toList.map (fun p => (p.1, jsonForm p.2)) = (m.map fun _ v => jsonForm v).toList :=
  (Std.TreeMap.Raw.toList_map (f := fun _ v => jsonForm v)).symm

theorem option_map_eq_iff {α β : Type} (f : α → β) (a b : Option α) :
    a.map f = b.map f ↔ a.isSome = b.isSome ∧ ∀ x y, a = some x → b = some y → f x = f y := by
  cases a <;> cases b <;> simp

/-- **Well-formed objects are equivalent as finite maps**: the same keys, and equivalent values at each. -/
theorem jsonEquiv_obj_iff {m n : Std.TreeMap.Raw String Json compare} (hm : m.WF) (hn : n.WF) :
    JsonEquiv (.obj m) (.obj n) ↔ (∀ k, (m.get? k).isSome = (n.get? k).isSome) ∧
      ∀ k a b, m.get? k = some a → n.get? k = some b → JsonEquiv a b := by
  have hM := wf_mapForm hm (fun _ v => jsonForm v)
  have hN := wf_mapForm hn (fun _ v => jsonForm v)
  have key : JsonEquiv (.obj m) (.obj n) ↔ ∀ k, (m.get? k).map jsonForm = (n.get? k).map jsonForm := by
    rw [JsonEquiv, jsonForm_obj, jsonForm_obj, JsonForm.obj.injEq, FieldForms.ofList_inj, toList_mapForm,
      toList_mapForm, ← Std.TreeMap.Raw.equiv_iff_toList_eq hM hN]
    constructor
    · intro h k
      have := Std.TreeMap.Raw.Equiv.getElem?_eq (k := k) hM hN h
      simpa [Std.TreeMap.Raw.getElem?_map hm, Std.TreeMap.Raw.getElem?_map hn] using this
    · intro h
      exact Std.TreeMap.Raw.Equiv.of_forall_getElem?_eq hM hN fun k => by
        simpa [Std.TreeMap.Raw.getElem?_map hm, Std.TreeMap.Raw.getElem?_map hn] using h k
  rw [key]
  simp only [option_map_eq_iff, JsonEquiv]
  exact ⟨fun h => ⟨fun k => (h k).1, fun k a b => (h k).2 a b⟩, fun h k => ⟨h.1 k, h.2 k⟩⟩

/-- `JsonEquiv` on optional values: both absent, or both present and equivalent. -/
def drawEq : Option Json → Option Json → Bool
  | none, none => true
  | some a, some b => decide (JsonEquiv a b)
  | _, _ => false

/-- **`drawEq` passes exactly on equivalent values.** -/
theorem drawEq_iff (a b : Option Json) : drawEq a b = true ↔ Option.Rel JsonEquiv a b := by
  rcases a with _ | a <;> rcases b with _ | b
  · exact ⟨fun _ => .none, fun _ => rfl⟩
  · exact ⟨fun h => by simp [drawEq] at h, fun h => by cases h⟩
  · exact ⟨fun h => by simp [drawEq] at h, fun h => by cases h⟩
  · simp only [drawEq, decide_eq_true_eq]
    exact ⟨.some, fun h => by cases h; assumption⟩

theorem drawEq_refl (a : Option Json) : drawEq a a = true :=
  (drawEq_iff a a).2 (by cases a <;> constructor; exact JsonEquiv.refl _)

end FlockDeploy
