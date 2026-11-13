/-
One comparison of two JSON values that a proof can read: `jsonSame a b` is true exactly when `a` and `b` are the same
value (`jsonSame_iff`). A check compares by it, never by `Lean.Json`'s `==`, which is a `partial def` that neither the
kernel nor a lemma can unfold.

It is structural equality of trees: constructor by constructor, a number by its mantissa and exponent, an array element
by element, and an object by its tree, node by node. So on objects it is stricter than value equality: the same fields
in another insertion order compare false, where `==`, which reads an object as a map, says true. That is safe for
soundness, since it refuses more. It is right for completeness only where both sides come from one parse of one
canonical text, or from literals (a string, a number, a Boolean, `null`, an array of those). Two values from two sources
are compared by their canonical texts (`Flock.canon`), never by `jsonSame`.

The rule: compare canonical texts, or trees of the same canonical parse.
-/
module

public import Lean.Data.Json.Basic

@[expose] public section

namespace Core

open Lean (Json)

/-- The tree under a JSON object: `Json.obj` holds a `Std.TreeMap.Raw`, a structure around this. -/
abbrev JsonTree := Std.DTreeMap.Internal.Impl String (fun _ => Json)

mutual
/-- **Whether `a` and `b` are the same JSON value** (`jsonSame_eq`, `jsonSame_refl`): structural equality of trees, so on
objects it is stricter than value equality, and the same fields in another insertion order compare false. It is for
trees of the same canonical parse and for a value against a literal; two values from two sources are compared by their
canonical texts. Total by structural recursion: an array is compared as its list (`jsonSameList`) and an object as its
tree (`jsonSameTree`), which are the two types `Json` holds itself in. -/
def jsonSame : Json → Json → Bool
  | .null, .null => true
  | .bool a, .bool b => a == b
  | .num a, .num b => a.mantissa == b.mantissa && a.exponent == b.exponent
  | .str a, .str b => a == b
  | .arr ⟨a⟩, .arr ⟨b⟩ => jsonSameList a b
  | .obj ⟨⟨a⟩⟩, .obj ⟨⟨b⟩⟩ => jsonSameTree a b
  | _, _ => false

/-- `jsonSame` element by element, on lists of one length. -/
def jsonSameList : List Json → List Json → Bool
  | [], [] => true
  | a :: as, b :: bs => jsonSame a b && jsonSameList as bs
  | _, _ => false

/-- `jsonSame` node by node, on trees of one shape: the size each node records, its key, its value and its two
subtrees. -/
def jsonSameTree : JsonTree → JsonTree → Bool
  | .leaf, .leaf => true
  | .inner sa ka va la ra, .inner sb kb vb lb rb =>
    sa == sb && ka == kb && jsonSame va vb && jsonSameTree la lb && jsonSameTree ra rb
  | _, _ => false
end

mutual
/-- **What `jsonSame` accepts is one value**: a check that passes on `jsonSame a b` passes only where `a = b`. -/
theorem jsonSame_eq : ∀ a b : Json, jsonSame a b = true → a = b
  | .null, b, h => by cases b <;> simp_all [jsonSame]
  | .bool _, b, h => by cases b <;> simp_all [jsonSame]
  | .num x, b, h => by
    cases b with
    | num y =>
      obtain ⟨xm, xe⟩ := x; obtain ⟨ym, ye⟩ := y
      simp only [jsonSame, Bool.and_eq_true, beq_iff_eq] at h
      rw [h.1, h.2]
    | _ => simp [jsonSame] at h
  | .str _, b, h => by cases b <;> simp_all [jsonSame]
  | .arr ⟨x⟩, b, h => by
    cases b with
    | arr y => obtain ⟨y⟩ := y; simp only [jsonSame] at h; rw [jsonSameList_eq x y h]
    | _ => simp [jsonSame] at h
  | .obj ⟨⟨x⟩⟩, b, h => by
    cases b with
    | obj y => obtain ⟨⟨y⟩⟩ := y; simp only [jsonSame] at h; rw [jsonSameTree_eq x y h]
    | _ => simp [jsonSame] at h

theorem jsonSameList_eq : ∀ a b : List Json, jsonSameList a b = true → a = b
  | [], b, h => by cases b <;> simp_all [jsonSameList]
  | x :: xs, b, h => by
    cases b with
    | nil => simp [jsonSameList] at h
    | cons y ys =>
      simp only [jsonSameList, Bool.and_eq_true] at h
      rw [jsonSame_eq x y h.1, jsonSameList_eq xs ys h.2]

theorem jsonSameTree_eq : ∀ a b : JsonTree, jsonSameTree a b = true → a = b
  | .leaf, b, h => by cases b <;> simp_all [jsonSameTree]
  | .inner sa ka va la ra, b, h => by
    cases b with
    | leaf => simp [jsonSameTree] at h
    | inner sb kb vb lb rb =>
      simp only [jsonSameTree, Bool.and_eq_true, beq_iff_eq] at h
      obtain ⟨⟨⟨⟨hs, hk⟩, hv⟩, hl⟩, hr⟩ := h
      rw [hs, hk, jsonSame_eq va vb hv, jsonSameTree_eq la lb hl, jsonSameTree_eq ra rb hr]
end

mutual
/-- **Every value is the same as itself**, an object whose tree no parser builds too: what an honest run's proof needs
of a comparison its check makes. -/
theorem jsonSame_refl : ∀ a : Json, jsonSame a a = true
  | .null => by simp [jsonSame]
  | .bool _ => by simp [jsonSame]
  | .num _ => by simp [jsonSame]
  | .str _ => by simp [jsonSame]
  | .arr ⟨x⟩ => by simp only [jsonSame]; exact jsonSameList_refl x
  | .obj ⟨⟨x⟩⟩ => by simp only [jsonSame]; exact jsonSameTree_refl x

theorem jsonSameList_refl : ∀ a : List Json, jsonSameList a a = true
  | [] => by simp [jsonSameList]
  | x :: xs => by simp only [jsonSameList, Bool.and_eq_true]; exact ⟨jsonSame_refl x, jsonSameList_refl xs⟩

theorem jsonSameTree_refl : ∀ a : JsonTree, jsonSameTree a a = true
  | .leaf => by simp [jsonSameTree]
  | .inner _ _ v l r => by
    simp only [jsonSameTree, Bool.and_eq_true, beq_self_eq_true, true_and]
    exact ⟨⟨jsonSame_refl v, jsonSameTree_refl l⟩, jsonSameTree_refl r⟩
end

/-- `jsonSame` decides equality. -/
theorem jsonSame_iff {a b : Json} : jsonSame a b = true ↔ a = b :=
  ⟨jsonSame_eq a b, fun h => h ▸ jsonSame_refl a⟩

/-- Against a string, `jsonSame` is the match a check wrote by hand before it. -/
theorem jsonSame_str (j : Json) (s : String) :
    jsonSame j (.str s) = (match j with | .str t => t == s | _ => false) := by
  cases j <;> rfl

example : jsonSame (.str "draw") (.str "draw") = true := by decide
example : jsonSame (.str "draw") (.str "other") = false := by decide
example : jsonSame (.arr #[.str "draw"]) (.str "draw") = false := by decide
example : jsonSame .null (.bool false) = false := by decide
example : jsonSame (.arr #[.null, .bool true, .num ⟨7, 0⟩]) (.arr #[.null, .bool true, .num ⟨7, 0⟩]) = true := by decide
example : jsonSame (.arr #[.null, .bool true]) (.arr #[.null, .bool true, .num ⟨7, 0⟩]) = false := by decide
example : jsonSame (.num ⟨10, 1⟩) (.num ⟨1, 0⟩) = false := by decide

/-- The object `{"a": null, "b": true}` as the tree that inserting `a` and then `b` builds. -/
private def exampleAB : Json := .obj ⟨⟨.inner 2 "a" .null .leaf (.inner 1 "b" (.bool true) .leaf .leaf)⟩⟩

/-- The same two fields as the tree that inserting `b` and then `a` builds. -/
private def exampleBA : Json := .obj ⟨⟨.inner 2 "b" (.bool true) (.inner 1 "a" .null .leaf .leaf) .leaf⟩⟩

example : jsonSame exampleAB exampleAB = true := by decide
-- One map and two trees: false here, where `==` says true.
example : jsonSame exampleAB exampleBA = false := by decide

end Core
