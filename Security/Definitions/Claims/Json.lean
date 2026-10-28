import Mathlib

/-!
# The JSON library's specification: completeness's JSON assumption

The honest prover sends its session record as JSON, and the verifier reads it back: the prover's printer (`print`)
prints it, then `Json.parse` reads it, and what the verifier compares is `Json`'s `==` and the canonical printer's
output (`canon`, the verifier's `Flock.canon`). `Json.parse` and `==` are `partial` definitions: Lean runs them, and
the kernel sees each as an opaque constant, so no proof unfolds them. `JsonSpec` states what completeness needs of them
and of `canon`, over JSON values that are alike (`Json.Same`), printable (`Json.Good`), ASCII (`Json.Ascii`) and
well-formed (`Json.Wf`).

* **Alike values** (`Same`): equal leaves, arrays alike element by element, and objects that are well-formed trees with
  one key set and alike values at each key. A Lean object is a tree whose shape is the order its keys were inserted in:
  the printer prints the keys in sorted order and `parse` inserts them in that order, so a parsed object is another
  tree than the one printed, alike at every key lookup.
* **Printable values** (`Good`): every object a well-formed tree, every number an integer (exponent `0`). The printer
  prints a number in its shortest form, so `1.50` comes back as `1.5`, another `JsonNumber`.
* **ASCII values** (`Ascii`): every string and key ASCII without DEL, the only ones `canon` prints.

Mathlib only, as `Definitions/` is: the printers are parameters, which the completeness proofs instantiate at the honest
prover's `FlockComplete.printJ` (a total printer of the bytes `Json.compress` prints, so proofs read its length) and
`Flock.canon`.
-/

namespace Claims

open Lean (Json JsonNumber)

namespace Json

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

/-- **An ASCII JSON value**: every string and every object key ASCII without DEL (each character below `0x7f`), the
values the canonical printer prints. -/
inductive Ascii : Json → Prop
  | null : Ascii .null
  | bool (b : Bool) : Ascii (.bool b)
  | num (n : JsonNumber) : Ascii (.num n)
  | str {s : String} (h : ∀ c ∈ s.toList, c.toNat < 0x7f) : Ascii (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Ascii x) : Ascii (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hk : ∀ p ∈ m.toList, ∀ c ∈ p.1.toList, c.toNat < 0x7f)
      (hv : ∀ p ∈ m.toList, Ascii p.2) : Ascii (.obj m)

/-- **A well-formed JSON value**: every object a well-formed tree. -/
inductive Wf : Json → Prop
  | null : Wf .null
  | bool (b : Bool) : Wf (.bool b)
  | num (n : JsonNumber) : Wf (.num n)
  | str (s : String) : Wf (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Wf x) : Wf (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hm : m.WF) (h : ∀ k v, m.get? k = some v → Wf v) : Wf (.obj m)

end Json

/-- **`JsonSpec`: the JSON library does what its source says**, at the prover's printer `print` and the canonical
printer `canon`: `Json.parse` reads back what `print` printed of a printable value, and what `canon` printed; `parse`
builds only well-formed trees; `Json`'s `==` holds of alike values; `canon` is the same on alike values, and returns on
printable ASCII ones.

It can't be proved: `Json.parse` and `Json`'s `==` are `partial` definitions, opaque to the kernel, so no proof unfolds
them (the two fields on `canon` alone are theorems at `Flock.canon`). It affects completeness only, never soundness: it
says that the honest prover's record reads back, and no soundness guarantee takes it. -/
structure JsonSpec (print : Json → String) (canon : Json → Except String String) : Prop where
  parse_print : ∀ j, Json.Good j → ∃ j', Json.parse (print j) = .ok j' ∧ Json.Same j j'
  parse_canon : ∀ j s, Json.Good j → canon j = .ok s → ∃ j', Json.parse s = .ok j' ∧ Json.Same j j'
  parse_wf : ∀ s j, Json.parse s = .ok j → Json.Wf j
  beq_same : ∀ a b, Json.Same a b → (a == b) = true
  canon_same : ∀ a b, Json.Same a b → canon a = canon b
  canon_ok : ∀ j, Json.Good j → Json.Ascii j → ∃ s, canon j = .ok s

end Claims
