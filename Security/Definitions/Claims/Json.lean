import Mathlib

/-!
# The JSON library's specification: completeness's JSON assumption

The honest prover sends its session record as JSON, and the verifier reads it back: the prover's printer (`print`)
prints it, then the verifier's parser (`parse`) reads it, and what the verifier compares is `Json`'s `==` and the
canonical printer's output (`canon`, the verifier's `Flock.canon`). `JsonSpec` states what completeness still assumes
of them, over JSON values that are alike (`Json.Same`) and printable (`Json.Good`); the proofs read well-formed values
(`Json.Wf`) too.

* **Alike values** (`Same`): equal leaves, arrays alike element by element, and objects that are well-formed trees with
  one key set and alike values at each key. A Lean object is a tree whose shape is the order its keys were inserted in:
  the printer prints the keys in sorted order and `parse` inserts them in that order, so a parsed object is another
  tree than the one printed, alike at every key lookup.
* **Printable values** (`Good`): every object a well-formed tree, every number an integer (exponent `0`). The printer
  prints a number in its shortest form, so `1.50` comes back as `1.5`, another `JsonNumber`.

Mathlib only, as `Definitions/` is: the parser and the printers are parameters, which the completeness proofs
instantiate at the verifier's `Flock.JsonParse.parse` (total: `Lean.Json.parse` with the same values and refusals),
the honest prover's `FlockComplete.printJ` (a total printer of the bytes `Json.compress` prints, so proofs read its
length) and `Flock.canon`.
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

/-- **A well-formed JSON value**: every object a well-formed tree. -/
inductive Wf : Json → Prop
  | null : Wf .null
  | bool (b : Bool) : Wf (.bool b)
  | num (n : JsonNumber) : Wf (.num n)
  | str (s : String) : Wf (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Wf x) : Wf (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hm : m.WF) (h : ∀ k v, m.get? k = some v → Wf v) : Wf (.obj m)

end Json

/-- **`JsonSpec`: the JSON library does what its source says**, at the verifier's parser `parse`, the prover's printer
`print` and the canonical printer `canon`: `parse` reads back what `print` printed of a printable value, and what
`canon` printed; `Json`'s `==` holds of alike values.

It affects completeness only, never soundness: it says that the honest prover's record reads back, and no soundness
guarantee takes it. What the parser and `canon` do apart from each other is proved, not assumed: the parser builds
well-formed trees (`FlockComplete.JsonRT.parse_wf`), and `canon` is the same on alike values and returns on printable
ones (`FlockComplete.JsonRT.canon_same`, `canon_ok`). `Json`'s `==` is a `partial` definition, opaque to the kernel, so
no proof unfolds it. -/
structure JsonSpec (parse : String → Except String Json) (print : Json → String)
    (canon : Json → Except String String) : Prop where
  parse_print : ∀ j, Json.Good j → ∃ j', parse (print j) = .ok j' ∧ Json.Same j j'
  parse_canon : ∀ j s, Json.Good j → canon j = .ok s → ∃ j', parse s = .ok j' ∧ Json.Same j j'
  beq_same : ∀ a b, Json.Same a b → (a == b) = true

end Claims
