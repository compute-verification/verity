/-
Canonical JSON and hex read at any size. `Core.canonicalJson`, `Flock.canon` and `Flock.unhex` go through
`String.toList`, whose runtime (`lean_string_data`) counts characters in 32 bits and builds a list cell per character:
a string of `n ≥ 2^32` characters becomes its first `n mod 2^32`, so `canonicalJson` refuses a canonical text holding
one ("not canonical JSON") and `Flock.unhex` reads a prefix of one, and below that a string costs about 30 bytes of
memory per character. A protocol's case carries its evidence as such a string (PoUS's `C*`, gigabytes of hex), so its
readers use these, which give the same answers at any size, reading the text's bytes in place:

* `parseUpTo`, `parse`, `read`: `canonicalJson`'s answer, the JSON it accepts or the reason it refuses, from one pass
  that accepts exactly the texts `Flock.canon` writes and copies each string out once;
* `unhex`: `Flock.unhex`'s answer, into a byte array of the right size;
* `writes`: whether `Flock.canon j` is `s`, or `canon`'s error, without writing it.

The `#guard`s at the end hold each to the function it stands in for on texts at every edge of canonical JSON, and
`CoreProtocol.Cli`'s hold `parseUpTo` and `parse` to `canonicalJson`. This imports only what `Core.Protocol` does, so
anything that may import that (a composition's decoder) may import this.
-/
import Lean.Data.Json
import Flock.Canon

namespace Core.Canonical

open Lean (Json JsonNumber)
open Flock (V)

/-- Byte `i` of `s`, `0` past its end: canonical JSON never holds a raw `0`, so a read past the end refuses. -/
@[inline] def byte (s : String) (i : Nat) : UInt8 :=
  if h : i < s.utf8ByteSize then String.Internal.getUTF8Byte s i h else 0

private def QUOTE : UInt8 := 34
private def BACKSLASH : UInt8 := 92
private def COMMA : UInt8 := 44
private def COLON : UInt8 := 58
private def LBRACKET : UInt8 := 91
private def RBRACKET : UInt8 := 93
private def LBRACE : UInt8 := 123
private def RBRACE : UInt8 := 125

/-- A lowercase hex digit's value, `16` for any other byte. -/
@[inline] def nibble (b : UInt8) : UInt8 :=
  if 48 ≤ b && b ≤ 57 then b - 48 else if 97 ≤ b && b ≤ 102 then b - 87 else 16

@[inline] private def hexDigit (x : UInt8) : UInt8 := if x < 10 then 48 + x else 87 + x

/-- The closing quote of a string whose body starts at `i`, and whether the body holds an escape: `none` unless the bytes
before `stop` are `Flock.canonStr`'s of some string. Those are raw bytes from `0x20` on but `"` and `\`, and `canonStr`'s
escapes exactly: `\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`, and `\u00xx` in lowercase for every other control character.
`fuel` bounds the steps: `stop - i + 1` is enough. -/
def strEnd (s : String) (stop : Nat) : (fuel i : Nat) → (esc : Bool) → Option (Nat × Bool)
  | 0, _, _ => none
  | fuel + 1, i, esc =>
    if stop ≤ i then none
    else
      let b := byte s i
      if b == QUOTE then some (i, esc)
      else if b == BACKSLASH then
        let e := byte s (i + 1)
        if stop ≤ i + 1 then none
        else if e == QUOTE || e == BACKSLASH || e == 110 || e == 114 || e == 116 || e == 98 || e == 102 then
          strEnd s stop fuel (i + 2) true
        else if e == 117 && i + 5 < stop && byte s (i + 2) == 48 && byte s (i + 3) == 48 then
          let h := byte s (i + 4)
          let l := nibble (byte s (i + 5))
          let v := (if h == 49 then 16 else 0) + l
          if (h == 48 || h == 49) && l < 16 && v != 8 && v != 9 && v != 10 && v != 12 && v != 13 then
            strEnd s stop fuel (i + 6) true
          else none
        else none
      else if b < 0x20 then none
      else strEnd s stop fuel (i + 1) esc

/-- The string whose canonical form runs from the quote at `a - 1` to the quote at `e`: its bytes as they are when it
holds no escape, else as Lean's JSON reads that form. -/
def strValue (s : String) (a e : Nat) (esc : Bool) : Option String :=
  if !esc then some (String.Pos.Raw.extract s ⟨a⟩ ⟨e⟩)
  else match Json.parse (String.Pos.Raw.extract s ⟨a - 1⟩ ⟨e + 1⟩) with
    | .ok (.str t) => some t
    | _ => none

/-- A run of decimal digits from `i`, read onto `acc`: its value and where it ends. -/
def digits (s : String) (stop : Nat) : (fuel i acc : Nat) → Nat × Nat
  | 0, i, acc => (acc, i)
  | fuel + 1, i, acc =>
    let b := byte s i
    if i < stop && 48 ≤ b && b ≤ 57 then digits s stop fuel (i + 1) (10 * acc + (b - 48).toNat) else (acc, i)

/-- An integer as `Flock.canon` writes it, `0` or an optional `-`, a nonzero digit and digits: its value and end. -/
def int (s : String) (stop i : Nat) : Option (Int × Nat) :=
  let neg := i < stop && byte s i == 45
  let j := if neg then i + 1 else i
  let b := byte s j
  if stop ≤ j then none
  else if b == 48 then if neg then none else some (0, j + 1)
  else if 49 ≤ b && b ≤ 57 then
    let (v, k) := digits s stop (stop - j + 1) j 0
    some (if neg then -(v : Int) else v, k)
  else none

/-- The ASCII word `w` written in `s` from `i`: where it ends. -/
def lit (s : String) (stop i : Nat) (w : String) : Option Nat :=
  let n := w.utf8ByteSize
  if i + n ≤ stop && (List.range n).all (fun k => byte s (i + k) == byte w k) then some (i + n) else none

mutual
/-- The JSON value whose canonical form starts at `i`, and its end; `none` unless one does. Objects and arrays are built
as Lean's JSON parser builds them (in the text's order, from `∅` and `#[]`), so the value is the one it reads. -/
def value (s : String) (stop : Nat) : (fuel i : Nat) → Option (Json × Nat)
  | 0, _ => none
  | fuel + 1, i =>
    if stop ≤ i then none
    else
      let b := byte s i
      if b == LBRACE then
        if i + 1 < stop && byte s (i + 1) == RBRACE then some (.obj ∅, i + 2) else obj s stop fuel (i + 1) ∅ none
      else if b == LBRACKET then
        if i + 1 < stop && byte s (i + 1) == RBRACKET then some (.arr #[], i + 2) else arr s stop fuel (i + 1) #[]
      else if b == QUOTE then
        match strEnd s stop (stop - i + 1) (i + 1) false with
        | some (e, esc) => (strValue s (i + 1) e esc).map fun t => (.str t, e + 1)
        | none => none
      else if b == 116 then (lit s stop i "true").map fun k => (.bool true, k)
      else if b == 102 then (lit s stop i "false").map fun k => (.bool false, k)
      else if b == 110 then (lit s stop i "null").map fun k => (.null, k)
      else (int s stop i).map fun (n, k) => (.num (JsonNumber.fromInt n), k)

/-- An array's elements from `i` on (after `[` or `,`), onto `acc`. -/
def arr (s : String) (stop : Nat) : (fuel i : Nat) → Array Json → Option (Json × Nat)
  | 0, _, _ => none
  | fuel + 1, i, acc =>
    match value s stop fuel i with
    | none => none
    | some (v, k) =>
      let b := byte s k
      if k < stop && b == RBRACKET then some (.arr (acc.push v), k + 1)
      else if k < stop && b == COMMA then arr s stop fuel (k + 1) (acc.push v)
      else none

/-- An object's members from `i` on (after `{` or `,`), onto `kvs`, each key after `prev` in `String`'s order (the order
`Flock.canon` sorts them in, so a key out of order or repeated is not canonical). -/
def obj (s : String) (stop : Nat) :
    (fuel i : Nat) → Std.TreeMap.Raw String Json compare → Option String → Option (Json × Nat)
  | 0, _, _, _ => none
  | fuel + 1, i, kvs, prev =>
    if stop ≤ i || byte s i != QUOTE then none
    else match strEnd s stop (stop - i + 1) (i + 1) false with
      | none => none
      | some (e, esc) =>
        match strValue s (i + 1) e esc with
        | none => none
        | some k =>
          let sorted := match prev with
            | some p => decide (p < k)
            | none => true
          if !sorted || stop ≤ e + 1 || byte s (e + 1) != COLON then none
          else match value s stop fuel (e + 2) with
            | none => none
            | some (v, m) =>
              let b := byte s m
              if m < stop && b == RBRACE then some (.obj (kvs.insert k v), m + 1)
              else if m < stop && b == COMMA then obj s stop fuel (m + 1) (kvs.insert k v) (some k)
              else none
end

/-- Every number in `j` is an integer: exactly when `Flock.canon j` doesn't fail. -/
def integers : Json → Bool
  | .num n => n.exponent == 0
  | .arr a => a.attach.all fun x => integers x.1
  | .obj kvs => kvs.toArray.attach.all fun p => integers p.1.2
  | _ => true
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact Flock.sizeOf_lt_of_mem_toArray p.2

/-- `canonicalJson`'s reason for refusing `t`, a text no canonical JSON is. -/
def refusal (t : String) : String :=
  match Json.parse t with
  | .error e => s!"not JSON ({e})"
  | .ok j =>
    if integers j then "not canonical JSON (compact, keys sorted, integers only)" else "canonical JSON: a non-integer number"

/-- `canonicalJson`'s answer on the first `stop` bytes of `s` (a whole number of its characters): the JSON when they are
its canonical form, else the reason. Only a refusal copies them. -/
def parseUpTo (s : String) (stop : Nat) : V Json :=
  match value s stop (2 * stop + 2) 0 with
  | some (j, k) => if k == stop then pure j else throw (refusal (String.Pos.Raw.extract s 0 ⟨stop⟩))
  | none => throw (refusal (String.Pos.Raw.extract s 0 ⟨stop⟩))

/-- `canonicalJson s.toUTF8`. -/
def parse (s : String) : V Json := parseUpTo s s.utf8ByteSize

/-- `canonicalJson b`, with the text it read: `b` as a string, which a caller checking an encoding against it
(`writes`) keeps in place of the JSON. -/
def read (b : ByteArray) : V (Json × String) := do
  let some s := String.fromUTF8? b | throw "not UTF-8"
  pure (← parse s, s)

/-- `Flock.unhex s`'s bytes from `i` on, after `acc`: `none` at a byte that isn't a lowercase hex digit. -/
def unhexFrom (s : String) (n : Nat) : (fuel i : Nat) → ByteArray → Option ByteArray
  | 0, _, _ => none
  | fuel + 1, i, acc =>
    if n ≤ i then some acc
    else
      let h := nibble (byte s i)
      let l := nibble (byte s (i + 1))
      if h < 16 && l < 16 then unhexFrom s n fuel (i + 2) (acc.push (h * 16 + l)) else none

/-- `Flock.unhex s`: an odd number of characters is "odd-length hex", and any character but `0-9a-f` (any non-ASCII
one among them) "not lowercase hex". -/
def unhex (s : String) : V ByteArray :=
  let n := s.utf8ByteSize
  if s.length % 2 != 0 then throw "odd-length hex"
  else if n != s.length then throw "not lowercase hex"
  else match unhexFrom s n (n / 2 + 1) 0 (ByteArray.emptyWithCapacity (n / 2)) with
    | some b => pure b
    | none => throw "not lowercase hex"

/-- Where `Flock.canonStr t`'s body from `t`'s byte `k` on, then its closing quote, ends when written in `s` from `i`.
`canonStr` escapes only ASCII bytes and copies every other character's bytes, which are all from `0x80` on. -/
def strWrites (s t : String) (stop : Nat) : (fuel k i : Nat) → Option Nat
  | 0, _, _ => none
  | fuel + 1, k, i =>
    if t.utf8ByteSize ≤ k then (if i < stop && byte s i == QUOTE then some (i + 1) else none)
    else
      let b := byte t k
      let two (c : UInt8) : Option Nat :=
        if i + 1 < stop && byte s i == BACKSLASH && byte s (i + 1) == c then some (i + 2) else none
      let next : Option Nat :=
        if b == QUOTE then two QUOTE
        else if b == BACKSLASH then two BACKSLASH
        else if b == 10 then two 110
        else if b == 13 then two 114
        else if b == 9 then two 116
        else if b == 8 then two 98
        else if b == 12 then two 102
        else if b < 0x20 then
          if i + 5 < stop && byte s i == BACKSLASH && byte s (i + 1) == 117 && byte s (i + 2) == 48 &&
              byte s (i + 3) == 48 && byte s (i + 4) == hexDigit (b / 16) && byte s (i + 5) == hexDigit (b % 16) then
            some (i + 6)
          else none
        else if i < stop && byte s i == b then some (i + 1)
        else none
      match next with
      | some i => strWrites s t stop fuel (k + 1) i
      | none => none

/-- Where `Flock.canonStr t` ends when written in `s` from `i`. -/
def strAt (s : String) (stop : Nat) (t : String) (i : Nat) : Option Nat :=
  if i < stop && byte s i == QUOTE then strWrites s t stop (t.utf8ByteSize + 1) 0 (i + 1) else none

/-- Where `Flock.canon j` ends when written in `s` from `i`, for `j` whose numbers are integers (`integers`). -/
def writesAt (s : String) (stop : Nat) : Json → Nat → Option Nat
  | .null, i => lit s stop i "null"
  | .bool b, i => lit s stop i (if b then "true" else "false")
  | .num n, i => lit s stop i (toString n.mantissa)
  | .str t, i => strAt s stop t i
  | .arr a, i =>
    if stop ≤ i || byte s i != LBRACKET then none
    else
      let r := a.attach.foldl (init := some (i + 1, true)) fun acc x =>
        match acc with
        | none => none
        | some (k, first) =>
          let k := if first then some k else if k < stop && byte s k == COMMA then some (k + 1) else none
          match k with
          | none => none
          | some k => (writesAt s stop x.1 k).map fun m => (m, false)
      match r with
      | some (k, _) => if k < stop && byte s k == RBRACKET then some (k + 1) else none
      | none => none
  | .obj kvs, i =>
    if stop ≤ i || byte s i != LBRACE then none
    else
      let items := kvs.toArray.attach.qsort (fun a b => a.1.1 < b.1.1)
      let r := items.foldl (init := some (i + 1, true)) fun acc p =>
        match acc with
        | none => none
        | some (k, first) =>
          let k := if first then some k else if k < stop && byte s k == COMMA then some (k + 1) else none
          match k with
          | none => none
          | some k =>
            match strAt s stop p.1.1 k with
            | none => none
            | some k =>
              if k < stop && byte s k == COLON then (writesAt s stop p.1.2 (k + 1)).map fun m => (m, false) else none
      match r with
      | some (k, _) => if k < stop && byte s k == RBRACE then some (k + 1) else none
      | none => none
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact Flock.sizeOf_lt_of_mem_toArray p.2

/-- `Flock.canon j == s`, or `canon j`'s error: whether `s` is `j`'s canonical JSON, compared in place. -/
def writes (j : Json) (s : String) : V Bool :=
  if integers j then pure (writesAt s s.utf8ByteSize j 0 == some s.utf8ByteSize)
  else throw "canonical JSON: a non-integer number"

/-! Each against the function it stands in for. -/

#guard (read (ByteArray.mk #[0xff])).toOption.isNone
#guard match read "{\"a\":[1]}".toUTF8 with | .ok (_, s) => s == "{\"a\":[1]}" | .error _ => false

private def HEX : List String := ["", "00", "0", "0g", "ab", "AB", "é0", "éé", "0é", "1234abcdef", "ff00", "f", "zz"]

#guard HEX.all fun t => match unhex t, Flock.unhex t with
  | .ok a, .ok b => a.toList == b.toList
  | .error a, .error b => a == b
  | _, _ => false

private def JSONS : List Json := [Json.null, .bool true, .bool false, .num 0, .num (-7), .num ⟨15, 1⟩, .str "",
  .str "a\"b\\c\nd\re\tf\x08g\x0ch\x01i\x1f~", .arr #[], .arr #[.num 1, .str "x", .arr #[.null]],
  Json.mkObj [], Json.mkObj [("b", .num 1), ("a", .arr #[.bool false]), ("e", .str "\x00")],
  Json.mkObj [("k", .num ⟨1, 2⟩)]]

#guard JSONS.all fun j => JSONS.all fun i =>
  match Flock.canon i with
  | .ok s => match writes j s, Flock.canon j with
    | .ok w, .ok c => w == (c == s)
    | .error a, .error b => a == b
    | _, _ => false
  | .error _ => true
#guard JSONS.all fun j => match Flock.canon j, writes j "" with
  | .ok _, .ok w => !w
  | .error a, .error b => a == b
  | _, _ => false

end Core.Canonical
