/-!
Where Python's `json.loads` reads a text otherwise than Lean's `Json.parse`, for the verdicts that read JSON the Python
reads with `json`: a number with a fraction or an exponent is a Python `float` (never an `int`, and `inf` past the double
range) where `Json.parse` reads `1e16` as a natural and expands every exponent into its mantissa, `NaN`, `Infinity` and
`-Infinity` are floats where `Json.parse` refuses the text, and a lone surrogate escape is kept where `Json.parse` writes
U+FFFD. `scan` puts `null` in place of each float, which fails every check a float fails (a natural, a string, a
Boolean, an array, an object), and says whether the text holds a value Python's canonical `json.dumps` (`allow_nan=False`,
then UTF-8) cannot write.
-/

namespace PousCheck.Prim.PyJson

/-- The decimal digits of `2^1024 − 2^970`, the least value Python's `float` rounds to `inf`. -/
def OVERFLOW : List Char := (Nat.repr (2 ^ 1024 - 2 ^ 970)).toList

/-- Whether `float(token)` is infinite, the token a number with a fraction or an exponent: its value `0.ds × 10^mag` is
at least `2^1024 − 2^970`. -/
def overflows (intDigits fracDigits expDigits : List Char) (negExp : Bool) : Bool :=
  let ds := (intDigits ++ fracDigits).dropWhile (· == '0')
  if ds.isEmpty then false
  else if expDigits.length > 20 then !negExp
  else
    let e : Nat := expDigits.foldl (fun (n : Nat) c => 10 * n + (c.toNat - 48)) 0
    let e : Int := if negExp then -(e : Int) else e
    let mag : Int := (ds.length : Int) + e - (fracDigits.length : Int)
    let top : Int := OVERFLOW.length
    if mag ≠ top then mag > top
    else
      let n := max ds.length OVERFLOW.length
      let pad (cs : List Char) := cs ++ List.replicate (n - cs.length) '0'
      decide ((String.ofList (pad ds)) ≥ String.ofList (pad OVERFLOW))

/-- A number token Python's `json` reads (`-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`, all of `cs`) as a float, and whether
that float is infinite. -/
def float? (cs : List Char) : Option Bool := Id.run do
  let cs := match cs with | '-' :: r => r | r => r
  let (int, cs) := cs.span Char.isDigit
  if !(int == ['0'] || (!int.isEmpty && int.head? != some '0')) then return none
  let mut cs := cs
  let mut frac : Option (List Char) := none
  if let '.' :: r := cs then
    let (d, r) := r.span Char.isDigit
    if d.isEmpty then return none
    (cs, frac) := (r, some d)
  let mut exp : Option (Bool × List Char) := none
  if let e :: r := cs then
    if e == 'e' || e == 'E' then
      let (neg, r) := match r with | '-' :: r => (true, r) | '+' :: r => (false, r) | r => (false, r)
      let (d, r) := r.span Char.isDigit
      if d.isEmpty then return none
      (cs, exp) := (r, some (neg, d))
  if !cs.isEmpty || (frac.isNone && exp.isNone) then return none
  let (neg, e) := exp.getD (false, [])
  return some (overflows int (frac.getD []) e neg)

def hexValue (cs : List Char) : Option Nat :=
  if cs.length == 4 && cs.all Char.isHexDigit then
    some (cs.foldl (fun n c => 16 * n + (if c.isDigit then c.toNat - 48 else (c.toLower.toNat - 87))) 0)
  else none

def numberChar (c : Char) : Bool := c.isDigit || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E'

/-- `scan`'s pass, `fuel` at least `cs.length`: copies `cs` to `out`, inside a string when `inString`. -/
def go : Nat → List Char → Bool → Array Char → Bool → Array Char × Bool
  | 0, _, _, out, bad => (out, bad)
  | _, [], _, out, bad => (out, bad)
  | n + 1, c :: cs, true, out, bad =>
    if c == '\\' then
      match cs with
      | 'u' :: rest =>
        let u := hexValue (rest.take 4)
        let low (r : List Char) := match r with
          | '\\' :: 'u' :: r => (hexValue (r.take 4)).any fun v => 0xDC00 ≤ v && v ≤ 0xDFFF
          | _ => false
        let high := u.any fun v => 0xD800 ≤ v && v ≤ 0xDBFF
        if high && low (rest.drop 4) then
          go n (rest.drop 10) true (out ++ (c :: 'u' :: rest.take 10).toArray) bad
        else
          let lone := high || u.any fun v => 0xDC00 ≤ v && v ≤ 0xDFFF
          go n rest true (out.push c |>.push 'u') (bad || lone)
      | d :: rest => go n rest true (out.push c |>.push d) bad
      | [] => (out.push c, bad)
    else go n cs (c != '"') (out.push c) bad
  | n + 1, c :: cs, false, out, bad =>
    let null := out ++ "null".toList.toArray
    if c == '"' then go n cs true (out.push c) bad
    else if c == 'N' && cs.take 2 == ['a', 'N'] then go n (cs.drop 2) false null true
    else if c == 'I' && cs.take 7 == "nfinity".toList then go n (cs.drop 7) false null true
    else if c == '-' && cs.take 8 == "Infinity".toList then go n (cs.drop 8) false null true
    else if numberChar c then
      let (tok, rest) := (c :: cs).span numberChar
      match float? tok with
      | some inf => go n rest false null (bad || inf)
      | none => go n rest false (out ++ tok.toArray) bad
    else go n cs false (out.push c) bad

/-- The text with `null` in place of every value Python's `json` reads as a float, and whether the text holds a value
Python's canonical `json.dumps` cannot write: `NaN`, `±Infinity`, a number past the double range, or a lone surrogate. -/
def scan (text : String) : String × Bool :=
  let (out, bad) := go text.length text.toList false #[] false
  (String.ofList out.toList, bad)

#guard scan "{\"a\":1e16,\"b\":[1.5,-0,10,NaN,-Infinity],\"c\":\"1e5\\\"2.0\"}" ==
  ("{\"a\":null,\"b\":[null,-0,10,null,null],\"c\":\"1e5\\\"2.0\"}", true)
#guard scan "[1e308,1.7976931348623157e308,1.7976931348623158e308,17976931348623158e292]" == ("[null,null,null,null]", false)
#guard (scan "[1.7976931348623159e308]").2 && (scan "[-1e309]").2 && (scan "[1e99999999999999999999999]").2
#guard (scan s!"[{Nat.repr (2 ^ 1024 - 2 ^ 970)}.0]").2 && !(scan s!"[{Nat.repr (2 ^ 1024 - 2 ^ 970 - 1)}.99999]").2
#guard !(scan "[0e99999999999999999999999,1e-99999999999999999999999]").2
#guard (scan "[\"\\ud800\"]").2 && (scan "[\"\\udc00x\"]").2 && !(scan "[\"\\ud83d\\ude00\"]").2
#guard scan "[05,1.,-,1e]" == ("[05,1.,-,1e]", false)

end PousCheck.Prim.PyJson
