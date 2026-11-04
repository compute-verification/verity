import Definitions.Claims.Json
import Lean.Data.Json
import Std.Data.TreeMap.Raw.Basic

/-!
# JSON values alike, printable and well formed

What the completeness proofs read of the record's JSON values:

* **Alike values** (`Same`): equal leaves, arrays alike element by element, and objects that are well-formed trees with
  one key set and alike values at each key. A Lean object is a tree whose shape is the order its keys were inserted in:
  the printer prints the keys in sorted order and the parser inserts them in that order, so a parsed object is another
  tree than the one printed, alike at every key lookup.
* **Printable values** (`Good`): every object a well-formed tree, every number an integer (exponent `0`). The printer
  prints a number in its shortest form, so `1.50` comes back as `1.5`, another `JsonNumber`.
* **ASCII values** (`Ascii`, `Definitions/Claims/Json.lean`): every string and key ASCII without DEL, the only ones
  `Flock.canon` prints.
* **Well-formed values** (`Wf`): every object a well-formed tree.

The verifier's parser `Flock.JsonParse.parse` reads back an alike value from what the honest prover's printer and
`Flock.canon` print of a printable value (`RoundTrip.lean`), so the round trip rests on no assumption.
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

end Claims
