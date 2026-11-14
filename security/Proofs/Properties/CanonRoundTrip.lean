module

public import Properties.Composition
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Proofs.Flock.Claims.Complete.Json.Hex
import all Init.Data.String.Defs

/-!
# `Core.canonicalJson` round trip: a printable value is its own canonical JSON

`Core.canonicalJson b` (`verity/protocols/interface/CoreProtocol/Protocol.lean`) accepts bytes `b` when every byte is
below `0x7f`, `b` is UTF-8, `Flock.JsonParse.parse` reads a value from it, and `Flock.canon` writes that value back as
`b` byte for byte. This file shows, for the honest registration bytes `θ := s.toUTF8` with `s` the text `Flock.canon`
writes of a value `j`, that `canonicalJson θ` returns a value, and names it.

What the value is. `Flock.JsonParse.parse` rebuilds every object by inserting its members into the empty tree in key
order, so `parse (canon j)` returns a value **alike** to `j` (`Claims.Json.Same`: both trees well formed, the same keys
present, and each key's values alike), not a value the kernel sees as definitionally equal to `j`. For leaves, numbers,
strings and arrays of those `Same` is equality; for objects it is equality only up to the tree's shape, which the
verifier's readers never see — they read a parsed object only through `get?`/`getObjVal?`, which `Same` preserves at
every key. So the round trip is stated up to `Same`, the equivalence the readers use; a caller that reads the result
through `get?` has what it needs, and one that needs the literal `j` builds `j` in the parser's insertion order.

Where the bytes stay ASCII. `Flock.canon` escapes every character at or above `0x7f` as a `\uXXXX` sequence
(`FlockComplete.JsonRT.canon_ascii`), so `s` is ASCII without DEL whatever `j` holds, and `θ`'s bytes are all below
`0x7f` (`toUTF8_all_lt`). Hence the round trip holds for every `Claims.Json.Good` value (every object a well-formed
tree, every number an integer), and the stricter `Plain` class the honest witness is built from is one such.
-/

@[expose] public section

namespace Properties.CanonRoundTrip

open Lean (Json JsonNumber)
open Claims.Json (Good Same Ascii)

/-! ## The canonical text's bytes are ASCII without DEL, and read back as the text -/

/-- **An ASCII-without-DEL string's UTF-8 bytes are all below `0x7f`.** Each character below `0x7f` has a one-byte
encoding, its own code point. -/
theorem toUTF8_all_lt {s : String} (h : Ascii s) : s.toUTF8.data.all (· < 0x7f) = true := by
  rw [Array.all_eq_true_iff_forall_mem]
  intro b hb
  rw [← Array.mem_toList_iff] at hb
  have hd : s.toUTF8.data.toList = s.toList.flatMap String.utf8EncodeChar := by
    conv_lhs => rw [← String.ofList_toList (s := s)]
    rw [String.toUTF8_eq_toByteArray, String.toByteArray_ofList, List.utf8Encode,
      List.toList_data_toByteArray]
  rw [hd] at hb
  obtain ⟨c, hc, hbc⟩ := List.mem_flatMap.1 hb
  have hc7 : c.toNat < 0x7f := h c hc
  have hcv : c.val.toNat < 127 := hc7
  have hsz : c.utf8Size = 1 := by
    rw [Char.utf8Size_eq_one_iff, UInt32.le_iff_toNat_le]
    simp only [UInt32.reduceToNat]
    omega
  rw [String.utf8EncodeChar_eq_singleton hsz, List.mem_singleton] at hbc
  subst hbc
  simp only [decide_eq_true_eq]
  refine UInt8.lt_iff_toNat_lt.2 ?_
  simp only [UInt32.toNat_toUInt8, UInt8.reduceToNat]
  omega

/-- **A string is read back from its own UTF-8 bytes.** -/
theorem fromUTF8?_toUTF8 (s : String) : String.fromUTF8? s.toUTF8 = some s := by
  have h : s.toUTF8.IsValidUTF8 := by simpa using s.isValidUTF8
  simp only [String.fromUTF8?, h, dite_true]
  rfl

/-! ## The round trip -/

/-- **A printable value round-trips through `Core.canonicalJson`.** `Flock.canon` writes `j` as a text `s` whose bytes
are ASCII without DEL and read back as `s`, and `canonicalJson s.toUTF8` returns a value `j'` alike to `j`
(`Claims.Json.Same`). For an object the parser's `j'` has the same members as `j` but need not be the same tree, so the
result is named up to `Same`, which the verifier's `get?`/`getObjVal?` readers preserve. -/
theorem good_canonRoundTrip {j : Json} (h : Good j) :
    ∃ s j', Flock.canon j = .ok s ∧ Flock.JsonParse.parse s = .ok j' ∧ Same j j' ∧ Ascii s ∧
      s.toUTF8.data.all (· < 0x7f) = true ∧ String.fromUTF8? s.toUTF8 = some s ∧
      Core.canonicalJson s.toUTF8 = .ok j' := by
  obtain ⟨s, hcanon, hascii⟩ := FlockComplete.JsonRT.canon_ok j h
  obtain ⟨j', hparse, hsame⟩ := FlockComplete.JsonRT.parse_canon j s h hcanon
  have hall : s.toUTF8.data.all (· < 0x7f) = true := toUTF8_all_lt hascii
  have hfrom : String.fromUTF8? s.toUTF8 = some s := fromUTF8?_toUTF8 s
  have hcj : Flock.canon j' = .ok s := (FlockComplete.JsonRT.canon_same j j' hsame).symm.trans hcanon
  refine ⟨s, j', hcanon, hparse, hsame, hascii, hall, hfrom, ?_⟩
  unfold Core.canonicalJson
  simp only [hall, hfrom, hparse, hcj, bind, Except.bind, pure, Except.pure, bne_self_eq_false,
    Bool.false_eq_true, reduceIte]

/-! ## `Plain`: the class the honest witness is built from -/

/-- A string the parser reads back unescaped and `canon` writes as itself: printable ASCII, no quote or backslash. Its
`canon` text is `"…"` with the characters verbatim, and every byte of it is below `0x7f`. -/
def PlainStr (s : String) : Prop := ∀ c ∈ s.toList, 0x20 ≤ c.toNat ∧ c.toNat < 0x7f ∧ c ≠ '"' ∧ c ≠ '\\'

/-- **A value built from printable pieces.** Strings are printable ASCII (`PlainStr`), numbers are natural-number
numerals, arrays hold `Plain` values, and objects are well-formed trees with printable keys and `Plain` values. Every
`Plain` value is `Claims.Json.Good`, so it round-trips (`plain_canonRoundTrip`). -/
inductive Plain : Json → Prop
  | null : Plain .null
  | bool (b : Bool) : Plain (.bool b)
  | num (n : Nat) : Plain (.num ⟨(n : Int), 0⟩)
  | str {s : String} (h : PlainStr s) : Plain (.str s)
  | arr {a : Array Json} (h : ∀ x ∈ a, Plain x) : Plain (.arr a)
  | obj {m : Std.TreeMap.Raw String Json} (hm : m.WF)
      (hk : ∀ k, (m.get? k).isSome = true → PlainStr k) (h : ∀ k v, m.get? k = some v → Plain v) :
      Plain (.obj m)

/-- **Every `Plain` value is `Good`.** -/
theorem plain_good : ∀ {j : Json}, Plain j → Good j
  | _, .null => .null
  | _, .bool b => .bool b
  | _, .num n => .num (n : Int)
  | _, .str _ => .str _
  | _, .arr h => .arr fun x hx => plain_good (h x hx)
  | _, .obj hm _ h => .obj hm fun k v hv => plain_good (h k v hv)

/-- **A `Plain` value round-trips through `Core.canonicalJson`**, up to `Same` (`good_canonRoundTrip`). -/
theorem plain_canonRoundTrip {j : Json} (h : Plain j) :
    ∃ s j', Flock.canon j = .ok s ∧ Flock.JsonParse.parse s = .ok j' ∧ Same j j' ∧ Ascii s ∧
      s.toUTF8.data.all (· < 0x7f) = true ∧ String.fromUTF8? s.toUTF8 = some s ∧
      Core.canonicalJson s.toUTF8 = .ok j' :=
  good_canonRoundTrip (plain_good h)

/-! ## Convenience: building `Plain` -/

/-- **A natural-number numeral is `Plain`.** -/
theorem plain_nat (n : Nat) : Plain (.num ⟨(n : Int), 0⟩) := .num n

/-- **A printable-ASCII string is `Plain`.** -/
theorem plain_str {s : String} (h : PlainStr s) : Plain (.str s) := .str h

/-- **An object is `Plain`** when it is a well-formed tree with printable keys and `Plain` values. -/
theorem plain_obj {m : Std.TreeMap.Raw String Json} (hm : m.WF)
    (hk : ∀ k, (m.get? k).isSome = true → PlainStr k) (h : ∀ k v, m.get? k = some v → Plain v) :
    Plain (.obj m) := .obj hm hk h

/-- **An object given as a key–value list is `Plain`** (`Lean.Json.mkObj`, the `.obj` of `Std.TreeMap.Raw.ofList`), which
is how the honest witness writes its objects (keys in `canon`'s strictly sorted order). The tree is well formed for free
(`Std.TreeMap.Raw.WF.ofList`); the keys and values are read, as the verifier reads them, through the built map's `get?`. -/
theorem plain_mkObj (l : List (String × Json))
    (hk : ∀ k, ((Std.TreeMap.Raw.ofList l).get? k).isSome = true → PlainStr k)
    (h : ∀ k v, (Std.TreeMap.Raw.ofList l).get? k = some v → Plain v) :
    Plain (Json.mkObj l) := by
  unfold Lean.Json.mkObj
  exact plain_obj Std.TreeMap.Raw.WF.ofList hk h

/-! ## Convenience: printable strings -/

/-- **Every lowercase-hex digit is printable** (`0`–`9`, `a`–`f`): a byte in `0x30…0x39 ∪ 0x61…0x66`, below `0x7f` and
neither quote nor backslash. -/
theorem plainStr_hd : ∀ d < 16,
    0x20 ≤ (Flock.Hex.hd d).toNat ∧ (Flock.Hex.hd d).toNat < 0x7f ∧
      Flock.Hex.hd d ≠ '"' ∧ Flock.Hex.hd d ≠ '\\' := by
  decide +kernel

/-- **A byte string's lowercase-hex rendering is printable** (`Flock.hexOf`): every character is a hex digit. -/
theorem plainStr_hexOf (bs : ByteArray) : PlainStr (Flock.hexOf bs) := by
  intro c hc
  rw [Flock.Hex.hexOf_toList] at hc
  obtain ⟨x, -, hx⟩ := List.mem_flatMap.1 hc
  have hx256 : x.toNat < 256 := x.toNat_lt
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl
  · exact plainStr_hd _ (by omega)
  · exact plainStr_hd _ (by omega)

/-- **A byte string's hex rendering is a `Plain` string.** -/
theorem plain_hexOf (bs : ByteArray) : Plain (.str (Flock.hexOf bs)) := .str (plainStr_hexOf bs)

end Properties.CanonRoundTrip
