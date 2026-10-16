import Lean.Data.Json.Basic

/-!
# `Json` equality Lean can reason through

Lean's `==` on `Json` is a `partial def` (opaque to proofs), so a check by it establishes nothing. `jsonEq` compares two
values constructor by constructor, an object by its tree of fields as stored, and passes exactly when they are equal
(`jsonEq_iff`); `drawEq` is it on optional values (`drawEq_iff`). It is stricter than `==` only on objects: `==` compares
fields by key, `jsonEq` also by the order the parser inserted them.
-/

namespace FlockDeploy

open Lean (Json)
open Std.DTreeMap.Internal (Impl)

mutual
/-- Structural equality of two `Json` values. -/
def jsonEq : Json → Json → Bool
  | .null, .null => true
  | .bool a, .bool b => a == b
  | .num a, .num b => decide (a = b)
  | .str a, .str b => decide (a = b)
  | .arr ⟨a⟩, .arr ⟨b⟩ => jsonsEq a b
  | .obj ⟨⟨a⟩⟩, .obj ⟨⟨b⟩⟩ => fieldsEq a b
  | _, _ => false

/-- `jsonEq` elementwise. -/
def jsonsEq : List Json → List Json → Bool
  | [], [] => true
  | a :: as, b :: bs => jsonEq a b && jsonsEq as bs
  | _, _ => false

/-- `jsonEq` on two trees of an object's fields, node by node. -/
def fieldsEq : Impl String (fun _ => Json) → Impl String (fun _ => Json) → Bool
  | .leaf, .leaf => true
  | .inner n k v l r, .inner n' k' v' l' r' =>
    decide (n = n') && decide (k = k') && jsonEq v v' && fieldsEq l l' && fieldsEq r r'
  | _, _ => false
end

mutual
theorem eq_of_jsonEq : ∀ a b : Json, jsonEq a b = true → a = b
  | .null, .null, _ => rfl
  | .bool a, .bool b, h => by simp [jsonEq] at h; rw [h]
  | .num a, .num b, h => by simp [jsonEq] at h; rw [h]
  | .str a, .str b, h => by simp [jsonEq] at h; rw [h]
  | .arr ⟨a⟩, .arr ⟨b⟩, h => by rw [eq_of_jsonsEq a b (by simpa [jsonEq] using h)]
  | .obj ⟨⟨a⟩⟩, .obj ⟨⟨b⟩⟩, h => by rw [eq_of_fieldsEq a b (by simpa [jsonEq] using h)]
  | .null, .bool _, h | .null, .num _, h | .null, .str _, h | .null, .arr _, h | .null, .obj _, h => by
    simp [jsonEq] at h
  | .bool _, .null, h | .bool _, .num _, h | .bool _, .str _, h | .bool _, .arr _, h | .bool _, .obj _, h => by
    simp [jsonEq] at h
  | .num _, .null, h | .num _, .bool _, h | .num _, .str _, h | .num _, .arr _, h | .num _, .obj _, h => by
    simp [jsonEq] at h
  | .str _, .null, h | .str _, .bool _, h | .str _, .num _, h | .str _, .arr _, h | .str _, .obj _, h => by
    simp [jsonEq] at h
  | .arr _, .null, h | .arr _, .bool _, h | .arr _, .num _, h | .arr _, .str _, h | .arr _, .obj _, h => by
    simp [jsonEq] at h
  | .obj _, .null, h | .obj _, .bool _, h | .obj _, .num _, h | .obj _, .str _, h | .obj _, .arr _, h => by
    simp [jsonEq] at h

theorem eq_of_jsonsEq : ∀ a b : List Json, jsonsEq a b = true → a = b
  | [], [], _ => rfl
  | a :: as, b :: bs, h => by
    simp only [jsonsEq, Bool.and_eq_true] at h
    rw [eq_of_jsonEq a b h.1, eq_of_jsonsEq as bs h.2]
  | [], _ :: _, h | _ :: _, [], h => by simp [jsonsEq] at h

theorem eq_of_fieldsEq : ∀ a b : Impl String (fun _ => Json), fieldsEq a b = true → a = b
  | .leaf, .leaf, _ => rfl
  | .inner n k v l r, .inner n' k' v' l' r', h => by
    simp only [fieldsEq, Bool.and_eq_true, decide_eq_true_eq] at h
    obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
    subst h1 h2
    rw [eq_of_jsonEq v v' h3, eq_of_fieldsEq l l' h4, eq_of_fieldsEq r r' h5]
  | .leaf, .inner .., h | .inner .., .leaf, h => by simp [fieldsEq] at h
end

mutual
theorem jsonEq_self : ∀ a : Json, jsonEq a a = true
  | .null => rfl
  | .bool a => by simp [jsonEq]
  | .num a => by simp [jsonEq]
  | .str a => by simp [jsonEq]
  | .arr ⟨a⟩ => by simpa [jsonEq] using jsonsEq_self a
  | .obj ⟨⟨a⟩⟩ => by simpa [jsonEq] using fieldsEq_self a

theorem jsonsEq_self : ∀ a : List Json, jsonsEq a a = true
  | [] => rfl
  | a :: as => by simp [jsonsEq, jsonEq_self a, jsonsEq_self as]

theorem fieldsEq_self : ∀ a : Impl String (fun _ => Json), fieldsEq a a = true
  | .leaf => rfl
  | .inner n k v l r => by simp [fieldsEq, jsonEq_self v, fieldsEq_self l, fieldsEq_self r]
end

/-- **`jsonEq` passes exactly on equal values.** -/
theorem jsonEq_iff (a b : Json) : jsonEq a b = true ↔ a = b :=
  ⟨eq_of_jsonEq a b, fun h => h ▸ jsonEq_self a⟩

/-- `jsonEq` on optional values: both absent, or both present and `jsonEq`. -/
def drawEq : Option Json → Option Json → Bool
  | none, none => true
  | some a, some b => jsonEq a b
  | _, _ => false

/-- **`drawEq` passes exactly on equal values.** -/
theorem drawEq_iff (a b : Option Json) : drawEq a b = true ↔ a = b := by
  cases a <;> cases b <;> simp [drawEq, jsonEq_iff]

end FlockDeploy
