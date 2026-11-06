import Lean.Data.Json.Basic

/-!
# The verifier's JSON parser: total, and evaluable by the kernel

`parse s` reads `s` as `Lean.Json.parse s` does (Lean 4.34's `Lean.Json.Parser`): the same `Json` value, objects built by
`Std.TreeMap.Raw.insert` in the order their keys arrive (a key named twice keeps its last value), and on a refusal the
same string, `offset N: why`, at the same byte offset and with the same reason. `Lean.Json.parse` is a `partial def`, an
opaque constant to the kernel, so no proof could show that any text parses. `parse` reads `s.toList` by structural
recursion on a fuel bound, keeping the byte offset beside it, so the kernel unfolds it (`decide +kernel`) in time linear
in the text, and proofs can reason about it.

Two places differ from the source, neither on an input the verifier reads:
* `Lean.Json.Parser` caps a number's decimals and positive exponent at `USize.size`, which the kernel can't evaluate
  (`System.Platform.numBits` is opaque); here the cap is `2^64`, its value on the 64-bit machines the verifier runs on.
* Each loop step and each nesting level takes one unit of fuel, `2 * s.toList.length + 2` in all. Every loop step consumes a
  character and every second nesting level does, so the fuel never runs out; were it to, the text would be refused
  (`offset N: nesting too deep`), never accepted.

`parse` is the one JSON reader of C-Flock's Lean (`tests/test_flock_json.py` refuses a call of `Lean.Json.parse` in
`verity/core/service/flock`). It is `@[irreducible]`, so that elaboration treats it as the opaque constant proofs were
written against; `unfold` and the kernel see its body.
-/

namespace Flock.JsonParse

open Lean (Json JsonNumber)

/-- A parser's outcome: a value with the characters after it and their byte offset, or a refusal at a byte offset. -/
inductive R (α : Type) where
  | ok (rest : List Char) (o : Nat) (a : α)
  | err (o : Nat) (why : String)

/-- `Std.Internal.Parsec.Error.eof`'s reason. -/
def EOF : String := "unexpected end of input"

/-- `Lean.Json.Parser`'s `USize.size`, at 64 bits. -/
def USIZE : Nat := 2 ^ 64

/-- `ws`: skip tabs, newlines, carriage returns and spaces. -/
def ws : List Char → Nat → List Char × Nat
  | [], o => ([], o)
  | c :: cs, o => if c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ' then ws cs (o + 1) else (c :: cs, o)

/-- `hexChar`: one hex digit's value. -/
def hexChar : List Char → Nat → R Nat
  | [], o => .err o EOF
  | c :: cs, o =>
    if '0' ≤ c ∧ c ≤ '9' then .ok cs (o + 1) (c.val - '0'.val).toNat
    else if 'a' ≤ c ∧ c ≤ 'f' then .ok cs (o + 1) (c.val - 'a'.val + 10).toNat
    else if 'A' ≤ c ∧ c ≤ 'F' then .ok cs (o + 1) (c.val - 'A'.val + 10).toNat
    else .err (o + c.utf8Size) "invalid hex character"

/-- `n` hex digits' value, most significant first. -/
def hexN : Nat → Nat → List Char → Nat → R Nat
  | 0, acc, cs, o => .ok cs o acc
  | n + 1, acc, cs, o => match hexChar cs o with
    | .ok cs o d => hexN n (16 * acc + d) cs o
    | .err o e => .err o e

/-- `finishSurrogatePair low`: the low half `\uDC00`–`\uDFFF` after the high half `low`, as one character; `none` where
the source's `attempt` fails, which falls back to `'\ufffd'` before the second escape. -/
def finishPair (low : Nat) : List Char → Nat → Option (List Char × Nat × Char)
  | '\\' :: 'u' :: d :: cs, o =>
    if d = 'd' ∨ d = 'D' then
      match hexN 3 0 cs (o + 3) with
      | .ok cs o v =>
        if v < 0xC00 then none
        else
          let cv := (low % 0x400) * 0x400 + v % 0x400 + 0x10000
          if cv.isValidChar then some (cs, o, Char.ofNat cv) else none
      | .err _ _ => none
    else none
  | _, _ => none

/-- `escapedChar`: the character an escape after `\` stands for. -/
def escapedChar : List Char → Nat → R Char
  | [], o => .err o EOF
  | c :: cs, o =>
    match c with
    | '\\' => .ok cs (o + 1) '\\'
    | '"' => .ok cs (o + 1) '"'
    | '/' => .ok cs (o + 1) '/'
    | 'b' => .ok cs (o + 1) '\x08'
    | 'f' => .ok cs (o + 1) '\x0c'
    | 'n' => .ok cs (o + 1) '\n'
    | 'r' => .ok cs (o + 1) '\x0d'
    | 't' => .ok cs (o + 1) '\t'
    | 'u' => match hexN 4 0 cs (o + 1) with
      | .ok cs o v =>
        if v < 0xD800 then .ok cs o (Char.ofNat v)
        else if v < 0xE000 then
          if v < 0xDC00 then match finishPair v cs o with
            | some (cs', o', c) => .ok cs' o' c
            | none => .ok cs o '\ufffd'
          else .ok cs o '\ufffd'
        else .ok cs o (Char.ofNat v)
      | .err o e => .err o e
    | _ => .err (o + c.utf8Size) "illegal \\u escape"

/-- `strCore`: a string's characters after its opening quote, through its closing one. -/
def str : Nat → String → List Char → Nat → R String
  | 0, _, _, o => .err o "nesting too deep"
  | _ + 1, _, [], o => .err o EOF
  | k + 1, acc, c :: cs, o =>
    if c = '"' then .ok cs (o + 1) acc
    else if c = '\\' then match escapedChar cs (o + 1) with
      | .ok cs o e => str k (acc.push e) cs o
      | .err o e => .err o e
    else if 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff then str k (acc.push c) cs (o + c.utf8Size)
    else .err (o + c.utf8Size) "unexpected character in string"

/-- `natCoreNumDigits`: decimal digits' value and their count, up to the first non-digit. -/
def digits : List Char → Nat → Nat → Nat → List Char × Nat × Nat × Nat
  | [], o, acc, n => ([], o, acc, n)
  | c :: cs, o, acc, n =>
    if '0' ≤ c ∧ c ≤ '9' then digits cs (o + 1) (10 * acc + (c.val - '0'.val).toNat) (n + 1) else (c :: cs, o, acc, n)

/-- `natNonZero`, `natNumDigits`, `natMaybeZero`: a run of digits whose first passes `ok`, else `expected why`. -/
def natAfter (ok : Char → Bool) (why : String) : List Char → Nat → R (Nat × Nat)
  | [], o => .err o EOF
  | c :: cs, o =>
    if ok c then
      let (cs, o, acc, n) := digits (c :: cs) o 0 0
      .ok cs o (acc, n)
    else .err o ("expected " ++ why)

/-- A digit. -/
def isDigit (c : Char) : Bool := '0' ≤ c ∧ c ≤ '9'

/-- `exponent`: a number's exponent part, if any. -/
def exponent (v : JsonNumber) : List Char → Nat → R JsonNumber
  | e :: cs, o =>
    if e = 'e' ∨ e = 'E' then
      match cs with
      | [] => .err (o + 1) EOF
      | '-' :: cs => match natAfter isDigit "0-9" cs (o + 2) with
        | .ok cs o (n, _) => .ok cs o (v.shiftr n)
        | .err o e => .err o e
      | c :: cs' =>
        let (cs, o) := if c = '+' then (cs', o + 2) else (c :: cs', o + 1)
        match natAfter isDigit "0-9" cs o with
        | .ok cs o (n, _) => if n > USIZE then .err o "exp too large" else .ok cs o (v.shiftl n)
        | .err o e => .err o e
    else .ok (e :: cs) o v
  | [], o => .ok [] o v

/-- `num`: a number, from its sign through its exponent. -/
def num (cs : List Char) (o : Nat) : R JsonNumber :=
  let (sign, cs, o) : Int × List Char × Nat := match cs with
    | '-' :: cs => (-1, cs, o + 1)
    | cs => (1, cs, o)
  let whole : R Nat := match cs with
    | [] => .err o EOF
    | '0' :: cs => .ok cs (o + 1) 0
    | cs => match natAfter (fun c => '1' ≤ c ∧ c ≤ '9') "1-9" cs o with
      | .ok cs o (n, _) => .ok cs o n
      | .err o e => .err o e
  let dec : R JsonNumber := match whole with
    | .err o e => .err o e
    | .ok ('.' :: cs) o whole =>
      match natAfter isDigit "digit" cs (o + 1) with
      | .ok cs o (n, d) =>
        if d > USIZE then .err o "too many decimals"
        else .ok cs o (JsonNumber.mk (sign * ((whole * (10 ^ d : Nat) + n : Nat) : Int)) d)
      | .err o e => .err o e
    | .ok cs o whole => .ok cs o (JsonNumber.fromInt (sign * whole))
  match dec with
  | .err o e => .err o e
  | .ok cs o v => exponent v cs o

/-- `pstring w`: the characters `w` next, else `expected: w`. -/
def lit : List Char → List Char → Nat → Option (List Char × Nat)
  | [], cs, o => some (cs, o)
  | c :: w, d :: cs, o => if c = d then lit w cs (o + c.utf8Size) else none
  | _ :: _, [], _ => none

/-- A literal `true`, `false` or `null`, and the whitespace after it. -/
def word (w : String) (j : Json) (cs : List Char) (o : Nat) : R Json :=
  match lit w.toList cs o with
  | some (cs, o) => let (cs, o) := ws cs o; .ok cs o j
  | none => .err o ("expected: " ++ w)

mutual

/-- `anyCore`: one value and the whitespace after it. -/
def value : Nat → List Char → Nat → R Json
  | 0, _, o => .err o "nesting too deep"
  | _ + 1, [], o => .err o EOF
  | k + 1, c :: cs, o =>
    if c = '[' then
      match ws cs (o + 1) with
      | ([], o) => .err o EOF
      | (']' :: cs, o) => let (cs, o) := ws cs (o + 1); .ok cs o (.arr #[])
      | (cs, o) => match array k #[] cs o with
        | .ok cs o a => .ok cs o (.arr a)
        | .err o e => .err o e
    else if c = '{' then
      match ws cs (o + 1) with
      | ([], o) => .err o EOF
      | ('}' :: cs, o) => let (cs, o) := ws cs (o + 1); .ok cs o (.obj ∅)
      | (cs, o) => match object k ∅ cs o with
        | .ok cs o m => .ok cs o (.obj m)
        | .err o e => .err o e
    else if c = '"' then match str k "" cs (o + 1) with
      | .ok cs o t => let (cs, o) := ws cs o; .ok cs o (.str t)
      | .err o e => .err o e
    else if c = 'f' then word "false" (.bool false) (c :: cs) o
    else if c = 't' then word "true" (.bool true) (c :: cs) o
    else if c = 'n' then word "null" .null (c :: cs) o
    else if c = '-' ∨ ('0' ≤ c ∧ c ≤ '9') then match num (c :: cs) o with
      | .ok cs o n => let (cs, o) := ws cs o; .ok cs o (.num n)
      | .err o e => .err o e
    else .err o "unexpected input"

/-- `arrayCore`: an array's elements after its `[`, through its `]` and the whitespace after it. -/
def array : Nat → Array Json → List Char → Nat → R (Array Json)
  | 0, _, _, o => .err o "nesting too deep"
  | k + 1, acc, cs, o => match value k cs o with
    | .err o e => .err o e
    | .ok [] o _ => .err o EOF
    | .ok (c :: cs) o hd =>
      if c = ']' then let (cs, o) := ws cs (o + 1); .ok cs o (acc.push hd)
      else if c = ',' then let (cs, o) := ws cs (o + 1); array k (acc.push hd) cs o
      else .err (o + c.utf8Size) "unexpected character in array"

/-- `objectCore`: an object's members after its `{`, through its `}` and the whitespace after it. -/
def object : Nat → Std.TreeMap.Raw String Json → List Char → Nat → R (Std.TreeMap.Raw String Json)
  | 0, _, _, o => .err o "nesting too deep"
  | _ + 1, _, [], o => .err o EOF
  | k + 1, kvs, '"' :: cs, o => match str k "" cs (o + 1) with
    | .err o e => .err o e
    | .ok cs o key => match ws cs o with
      | ([], o) => .err o EOF
      | (':' :: cs, o) =>
        let (cs, o) := ws cs (o + 1)
        match value k cs o with
        | .err o e => .err o e
        | .ok [] o _ => .err o EOF
        | .ok (c :: cs) o v =>
          if c = '}' then let (cs, o) := ws cs (o + 1); .ok cs o (kvs.insert key v)
          else if c = ',' then let (cs, o) := ws cs (o + 1); object k (kvs.insert key v) cs o
          else .err (o + c.utf8Size) "unexpected character in object"
      | (_, o) => .err o "expected :"
  | _ + 1, _, _ :: _, o => .err o "expected \""

end

/-- `parse` on the text's characters. -/
def parseList (cs : List Char) : Except String Json :=
  let (rest, o) := ws cs 0
  match value (2 * cs.length + 2) rest o with
  | .ok [] _ v => .ok v
  | .ok (_ :: _) o _ => .error ("offset " ++ toString o ++ ": expected end of input")
  | .err o e => .error ("offset " ++ toString o ++ ": " ++ e)

/-- **`Lean.Json.parse s`, total**: one value between optional whitespace, and nothing after it. The kernel is slow to
take a string literal apart (`String.toList`), so a proof that evaluates one reads it through `parse_eq`. -/
@[irreducible] def parse (s : String) : Except String Json := parseList s.toList

theorem parse_eq (s : String) : parse s = parseList s.toList := by unfold parse; rfl

end Flock.JsonParse
