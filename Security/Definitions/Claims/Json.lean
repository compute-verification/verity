import Mathlib

/-!
# The JSON library's specification: completeness's JSON assumption

The honest prover sends its session record as JSON, and the verifier reads it back: the prover's printer prints it,
the verifier's parser reads it, and what the verifier compares is `Json`'s `==` and the canonical printer's output.
`JsonSpec` states what completeness still assumes of them, over JSON values that are alike (`Json.Same`): that `==`
holds of them. The proofs read printable (`Json.Good`) and well-formed (`Json.Wf`) values too.

* **Alike values** (`Same`): equal leaves, arrays alike element by element, and objects that are well-formed trees with
  one key set and alike values at each key. A Lean object is a tree whose shape is the order its keys were inserted in:
  the printer prints the keys in sorted order and `parse` inserts them in that order, so a parsed object is another
  tree than the one printed, alike at every key lookup.
* **Printable values** (`Good`): every object a well-formed tree, every number an integer (exponent `0`). The printer
  prints a number in its shortest form, so `1.50` comes back as `1.5`, another `JsonNumber`.

Mathlib only, as `Definitions/` is: the comparison is a parameter, which the completeness proofs instantiate at
`BEq.beq`. What the verifier's parser (`Flock.JsonParse.parse`, total) reads back of the honest prover's printer
(`FlockComplete.printJ`) and the canonical printer (`Flock.canon`) is proved, not assumed
(`Proofs/Flock/Claims/Complete/Json/RoundTrip.lean`).
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

/-- **`JsonSpec`: `Json`'s `==` holds of alike values**, at the comparison `beq` (instantiated at `BEq.beq`, Lean's
`==` on `Json`).

It affects completeness only, never soundness: it says that the honest prover's record reads back, and no soundness
guarantee takes it. `Json`'s `==` is a `partial` definition, opaque to the kernel, so no proof unfolds it. What the
verifier's parser and printers do is proved, not assumed: the parser builds well-formed trees
(`FlockComplete.JsonRT.parse_wf`) and reads back an alike value from the prover's printer and from `canon`
(`FlockComplete.JsonRT.parse_print`, `parse_canon`), and `canon` is the same on alike values and returns on printable
ones (`FlockComplete.JsonRT.canon_same`, `canon_ok`). -/
structure JsonSpec (beq : Json → Json → Bool) : Prop where
  beq_same : ∀ a b, Json.Same a b → beq a b = true

end Claims
