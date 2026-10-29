/-
JSON as Python's `canonical_json` writes it (`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`),
the form the audit record is compared and digested in: keys sorted, strings escaped to ASCII, integers as written, and a
number with a fraction or a negative exponent (`Json.parse` gives it a positive decimal exponent) as Python's `repr` of
the double nearest it: the shortest digits that read back to it, closest to it, an even last digit on a tie, in fixed
notation unless the decimal point is more than 16 places right or 4 left of the first digit. What a Python float can't be
read back as (`-0.0`, a float of 10¹⁶ or more, which `Json.parse` reads as an integer) is written as the integer or `0.0`.
-/
import Lean.Data.Json
import CoreProtocol.Ratio

namespace SampledProofs.PyJson

open Lean (Json JsonNumber)

private def pos (i : Int) : Nat := if 0 ≤ i then i.toNat else 0
private def neg (i : Int) : Nat := if i < 0 then (-i).toNat else 0

/-- `x · 2^b · 10^t` as a fraction of naturals. -/
private def scaled (x : Nat) (b t : Int) : Nat × Nat := (x * 2 ^ pos b * 10 ^ pos t, 2 ^ neg b * 10 ^ neg t)

private def lt (a b : Nat × Nat) : Bool := a.1 * b.2 < b.1 * a.2

/-- The double nearest the positive ratio `n / d`, rounding half to even: `(m, e)` with value `m · 2^e`, `m < 2^53`
(normal when `m ≥ 2^52`), or none past the largest double. -/
def nearest (n d : Nat) : Option (Nat × Int) :=
  if n == 0 || d == 0 then some (0, 0) else
  let l0 : Int := (n.log2 : Int) - d.log2
  let l := if lt (n, d) (scaled 1 l0 0) then l0 - 1 else l0
  let e := max (l - 52) (-1074)
  let (num, den) := (n * 2 ^ neg e, d * 2 ^ pos e)
  let q := num / den
  let r := num % den
  let q := if 2 * r > den || (2 * r == den && q % 2 == 1) then q + 1 else q
  let (m, e) := if q == 2 ^ 53 then (2 ^ 52, e + 1) else (q, e)
  if e > 971 then none else some (m, e)

/-- `repr`'s digits and decimal point for the double `m · 2^e`: the value is `0.digits × 10^point`. -/
def shortest (m : Nat) (e : Int) : String × Int := Id.run do
  if m == 0 then return ("0", 1)
  let t := e - 2
  let v := 4 * m
  let hi := 4 * m + 2
  let lo := if m == 2 ^ 52 && e > -1074 then 4 * m - 1 else 4 * m - 2
  let inclusive := m % 2 == 0
  let val := scaled v t 0
  -- the decimal point: 10^(k-1) ≤ value < 10^k
  let mut k : Int := ((toString val.1).length : Int) - (toString val.2).length
  for _ in [0:4] do
    if lt val (scaled 1 0 (k - 1)) then k := k - 1
    else if !lt val (scaled 1 0 k) then k := k + 1
  let inside := fun (c : Nat × Nat) =>
    let (a, b) := (scaled lo t 0, scaled hi t 0)
    if inclusive then !lt c a && !lt b c else lt a c && lt c b
  for n in [1:18] do
    let q := k - n
    let d0 := (v * 2 ^ pos t * 10 ^ neg q) / (2 ^ neg t * 10 ^ pos q)
    let c0 := scaled d0 0 q
    let c1 := scaled (d0 + 1) 0 q
    let pick : Option Nat := match inside c0, inside c1 with
      | true, true =>
        -- the closer to the value, an even last digit on a tie
        let twice := scaled (2 * v) t 0
        let mid := scaled (2 * d0 + 1) 0 q
        if lt twice mid then some d0 else if lt mid twice then some (d0 + 1)
        else some (if d0 % 2 == 0 then d0 else d0 + 1)
      | true, false => some d0
      | false, true => some (d0 + 1)
      | false, false => none
    if let some d := pick then
      let s := toString d
      let point := q + s.length
      let stripped := (s.toList.reverse.dropWhile (· == '0')).reverse
      return (String.ofList stripped, point)
  return ("0", 1)

/-- Python's `repr` of a float from its digits and decimal point. -/
def reprDigits (negative : Bool) (digits : String) (point : Int) : String :=
  let sign := if negative then "-" else ""
  let n : Int := digits.length
  if point ≤ -4 || point > 16 then
    let x := point - 1
    let mant := match digits.toList with
      | c :: [] => String.singleton c
      | c :: rest => String.singleton c ++ "." ++ String.ofList rest
      | [] => "0"
    let ex := toString x.natAbs
    sign ++ mant ++ "e" ++ (if x < 0 then "-" else "+") ++ (if ex.length < 2 then "0" ++ ex else ex)
  else if point ≤ 0 then sign ++ "0." ++ "".pushn '0' (neg point) ++ digits
  else if point < n then sign ++ String.ofList (digits.toList.take point.toNat) ++ "." ++ String.ofList (digits.toList.drop point.toNat)
  else sign ++ digits ++ "".pushn '0' (point - n).toNat ++ ".0"

/-- Python's `repr(float(n / d))` for a ratio with a nonzero denominator; none past the largest double. -/
def reprRatio (negative : Bool) (n d : Nat) : Option String := do
  let (m, e) ← nearest n d
  let (digits, point) := shortest m e
  return reprDigits (negative && m != 0) digits point

def reprOf (r : Core.Ratio) : Option String := reprRatio false r.num r.den

private def hex4 (n : Nat) : String :=
  let s := String.ofList (Nat.toDigits 16 n)
  "".pushn '0' (4 - s.length) ++ s

/-- A string as `json.dumps(..., ensure_ascii=True)` writes it. -/
def str (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    out := out ++ match c with
      | '"' => "\\\""
      | '\\' => "\\\\"
      | '\n' => "\\n"
      | '\r' => "\\r"
      | '\t' => "\\t"
      | '\x08' => "\\b"
      | '\x0c' => "\\f"
      | c =>
        if ' ' ≤ c && c ≤ '~' then String.singleton c
        else if c.val < 0x10000 then "\\u" ++ hex4 c.val.toNat
        else
          let v := c.val.toNat - 0x10000
          "\\u" ++ hex4 (0xD800 + v / 1024) ++ "\\u" ++ hex4 (0xDC00 + v % 1024)
  return out ++ "\""

def number (x : JsonNumber) : Except String String :=
  if x.exponent == 0 then pure (toString x.mantissa) else
  match reprRatio (x.mantissa < 0) x.mantissa.natAbs (10 ^ x.exponent) with
  | some s => pure s
  | none => throw "a number past the largest double"

/-- `canonical_json` of a value nested at most `depth` deep (Python's own limit is about a thousand). -/
def write : Nat → Json → Except String String
  | 0, _ => throw "JSON nested too deep"
  | _ + 1, .null => pure "null"
  | _ + 1, .bool b => pure (if b then "true" else "false")
  | _ + 1, .num x => number x
  | _ + 1, .str s => pure (str s)
  | d + 1, .arr xs => do return "[" ++ ",".intercalate (← xs.toList.mapM (write d)) ++ "]"
  | d + 1, .obj kvs => do
    let items ← kvs.toArray.toList.mapM fun (k, v) => do return str k ++ ":" ++ (← write d v)
    return "{" ++ ",".intercalate items ++ "}"

def DEPTH : Nat := 1000

def canonical (j : Json) : Except String String := write DEPTH j

end SampledProofs.PyJson
