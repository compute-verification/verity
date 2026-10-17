import Flock.Canon
import Std.Data.TreeMap.Raw.Lemmas

/-!
# What the record's round trip needs from the JSON library

`Json.parse`, `Json.compress`, `Json`'s `==` and the verifier's `canon` are `partial` definitions: Lean runs them, and no
proof unfolds them. So the session record's round trip (the prover's `compress`, then `Record.decodeWith`'s, `liveOf`'s
and `recordDraw`'s `parse`) is proved from their specification, `JsonSpec`, a named assumption the record's theorems take
as a hypothesis.

* **Alike values** (`Same`): equal leaves, arrays alike element by element, and objects that are well-formed trees with
  one key set and alike values at each key. A Lean object is a tree whose shape is the order its keys were inserted in:
  `compress` prints the keys in sorted order and `parse` inserts them in that order, so a parsed object is another tree
  than the one printed, alike at every key lookup (`Same.getObjVal`).
* **Printable values** (`Good`): every object a well-formed tree, every number an integer (exponent `0`). `compress`
  prints a number in its shortest form, so `1.50` comes back as `1.5`, another `JsonNumber`; the record's numbers are all
  `Lean.toJson` of naturals.
* **`JsonSpec`**: `parse` reads back what `compress` printed of a printable value, and what `canon` printed; `parse` builds
  only well-formed trees; `==` holds of alike values; `canon` is the same on alike values, and returns on printable ones.
-/

namespace FlockComplete.JsonRT

open Lean (Json JsonNumber)
open Flock

/-- **Alike JSON values**: equal leaves, arrays alike element by element, objects (both well-formed trees) with one key
set and alike values at each key. -/
inductive Same : Json → Json → Prop
  | null : Same .null .null
  | bool (b : Bool) : Same (.bool b) (.bool b)
  | num (n : JsonNumber) : Same (.num n) (.num n)
  | str (s : String) : Same (.str s) (.str s)
  | arr {a b : Array Json} (hs : a.size = b.size)
      (he : ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i]) : Same (.arr a) (.arr b)
  | obj {m n : Std.TreeMap.Raw String Json} (hm : m.WF) (hn : n.WF) (hk : ∀ k, (m.get? k).isSome = (n.get? k).isSome)
      (hv : ∀ k a b, m.get? k = some a → n.get? k = some b → Same a b) : Same (.obj m) (.obj n)

/-- **A printable JSON value**: every object a well-formed tree, every number an integer. -/
inductive Good : Json → Prop
  | null : Good .null
  | bool (b : Bool) : Good (.bool b)
  | num (m : Int) : Good (.num ⟨m, 0⟩)
  | str (s : String) : Good (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Good x) : Good (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hm : m.WF) (h : ∀ k v, m.get? k = some v → Good v) : Good (.obj m)

/-- **A well-formed JSON value**: every object a well-formed tree. -/
inductive Wf : Json → Prop
  | null : Wf .null
  | bool (b : Bool) : Wf (.bool b)
  | num (n : JsonNumber) : Wf (.num n)
  | str (s : String) : Wf (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Wf x) : Wf (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hm : m.WF) (h : ∀ k v, m.get? k = some v → Wf v) : Wf (.obj m)

/-- **The JSON library's specification**, what the record's round trip assumes of the `partial` definitions:
`Json.parse` reads back `Json.compress`'s and `canon`'s output, and builds only well-formed trees; `Json`'s `==` holds of
alike values; `canon` is the same on alike values, and returns on printable ones. -/
structure JsonSpec : Prop where
  parse_compress : ∀ j, Good j → ∃ j', Json.parse j.compress = .ok j' ∧ Same j j'
  parse_canon : ∀ j s, Good j → canon j = .ok s → ∃ j', Json.parse s = .ok j' ∧ Same j j'
  parse_wf : ∀ s j, Json.parse s = .ok j → Wf j
  beq_same : ∀ a b, Same a b → (a == b) = true
  canon_same : ∀ a b, Same a b → canon a = canon b
  canon_ok : ∀ j, Good j → ∃ s, canon j = .ok s

/-! ## Alike values -/

theorem Same.symm : ∀ {a b : Json}, Same a b → Same b a
  | _, _, .null => .null
  | _, _, .bool b => .bool b
  | _, _, .num n => .num n
  | _, _, .str s => .str s
  | _, _, .arr hs he => .arr hs.symm fun i h₁ h₂ => (he i h₂ h₁).symm
  | _, _, .obj hm hn hk hv => .obj hn hm (fun k => (hk k).symm) fun k a b ha hb => (hv k b a hb ha).symm

theorem same_str {s : String} {j : Json} (h : Same (.str s) j) : j = .str s := by cases h; rfl

theorem same_null {j : Json} (h : Same .null j) : j = .null := by cases h; rfl

theorem same_bool {b : Bool} {j : Json} (h : Same (.bool b) j) : j = .bool b := by cases h; rfl

theorem same_num {n : JsonNumber} {j : Json} (h : Same (.num n) j) : j = .num n := by cases h; rfl

theorem same_arr {a : Array Json} {j : Json} (h : Same (.arr a) j) :
    ∃ b : Array Json, j = .arr b ∧ a.size = b.size ∧ ∀ (i : Nat) (h₁ : i < a.size) (h₂ : i < b.size), Same a[i] b[i] := by
  cases h with
  | arr hs he => exact ⟨_, rfl, hs, he⟩

theorem same_obj {m : Std.TreeMap.Raw String Json} {j : Json} (h : Same (.obj m) j) :
    ∃ n : Std.TreeMap.Raw String Json, j = .obj n ∧ ∀ k, (m.get? k).isSome = (n.get? k).isSome ∧
      ∀ a b, m.get? k = some a → n.get? k = some b → Same a b := by
  cases h with
  | obj _ _ hk hv => exact ⟨_, rfl, fun k => ⟨hk k, hv k⟩⟩

/-- **A key's value in alike objects**: the value at a key of one is alike the other's at that key. -/
theorem Same.getObjVal {j j' : Json} (h : Same j j') {k : String} {v : Json} (hv : j.getObjVal? k = .ok v) :
    ∃ v', j'.getObjVal? k = .ok v' ∧ Same v v' := by
  cases j with
  | obj m =>
    obtain ⟨n, rfl, hn⟩ := same_obj h
    simp only [Json.getObjVal?] at hv ⊢
    cases hm : m.get? k with
    | none => rw [hm] at hv; cases hv
    | some a =>
      rw [hm] at hv
      cases hv
      have hs := (hn k).1
      rw [hm] at hs
      cases hn' : n.get? k with
      | none => rw [hn'] at hs; cases hs
      | some b => exact ⟨b, rfl, (hn k).2 _ _ hm hn'⟩
  | _ => cases hv

/-- **A missing key of alike objects** is missing from both. -/
theorem Same.getObjVal_none {j j' : Json} (h : Same j j') {k : String} {m : Std.TreeMap.Raw String Json}
    (hj : j = .obj m) (hk : m.get? k = none) : ∃ e, j'.getObjVal? k = .error e := by
  subst hj
  obtain ⟨n, rfl, hn⟩ := same_obj h
  have hs := (hn k).1
  rw [hk] at hs
  simp only [Json.getObjVal?]
  cases hn' : n.get? k with
  | none => exact ⟨_, rfl⟩
  | some b => rw [hn'] at hs; cases hs

end FlockComplete.JsonRT
