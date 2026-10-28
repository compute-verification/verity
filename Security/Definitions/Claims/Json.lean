import Mathlib

/-!
# ASCII JSON values: what the canonical printer prints

`Flock.canon` prints a string or an object key only when it is ASCII without DEL (every character below `0x7f`, the
domain on which canonical JSON is RFC 8785's and Python's), and refuses any other. Completeness reads that a value it
prints is such a value (`Json.Ascii`): the honest prover's draw, `Hello` and coin tree are, so `canon` prints them.

Mathlib only, as `Definitions/` is.
-/

namespace Claims

open Lean (Json JsonNumber)

namespace Json

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

end Json

end Claims
