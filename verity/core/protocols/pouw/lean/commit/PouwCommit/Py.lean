import Lean.Data.Json

/-!
# How the Python reads its JSON

The coins' JSON forms are read as Python reads them: `bytes.fromhex` (two hex digits a byte, either case, ASCII whitespace
between bytes), and a label shown in a reason as `repr` shows it. `pyRepr` is CPython's for every ASCII string; above
ASCII it escapes C1 controls, U+00A0 and U+00AD and keeps every other code point, where CPython also escapes the rest of
Unicode's non-printable ones.
-/

namespace PouwCommit

open Lean

private def hexVal (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else if 'A' ≤ c && c ≤ 'F' then some (c.toNat - 'A'.toNat + 10)
  else none

private def pySpace (c : Char) : Bool := c == ' ' || c == '\t' || c == '\n' || c == '\r' || c.toNat == 11 || c.toNat == 12

/-- `bytes.fromhex`. -/
def pyFromHex (s : String) : Except String ByteArray := go s.toList ByteArray.empty
where
  go : List Char → ByteArray → Except String ByteArray
    | [], acc => pure acc
    | c :: cs, acc =>
      if pySpace c then go cs acc
      else match cs with
        | d :: ds => match hexVal c, hexVal d with
          | some a, some b => go ds (acc.push (16 * a + b).toUInt8)
          | _, _ => throw s!"not hex: {s}"
        | [] => throw s!"not hex: {s}"

private def hexDigits (w n : Nat) : String :=
  String.ofList ((List.range w).reverse.map fun i => Nat.digitChar (n / 16 ^ i % 16))

/-- `repr` of a string (module docstring). -/
def pyRepr (s : String) : String :=
  let q := if s.contains '\'' && !s.contains '"' then '"' else '\''
  let body := s.toList.foldl (fun acc c =>
    let n := c.toNat
    acc ++ if c == q || c == '\\' then String.ofList ['\\', c]
      else if c == '\t' then "\\t" else if c == '\n' then "\\n" else if c == '\r' then "\\r"
      else if n < 32 || (127 ≤ n && n ≤ 160) || n == 173 then "\\x" ++ hexDigits 2 n
      else c.toString) ""
  q.toString ++ body ++ q.toString

/-- A JSON integer written without fraction or exponent. -/
def jsonInt? : Json → Option Int
  | .num ⟨m, 0⟩ => some m
  | _ => none

def field (j : Json) (k : String) : Except String Json :=
  match j with
  | .obj kv => match kv.get? k with
    | some v => pure v
    | none => throw s!"missing {k}"
  | _ => throw s!"not an object, reading {k}"

def hexField (j : Json) (k : String) : Except String ByteArray := do
  match ← field j k with
  | .str s => pyFromHex s
  | _ => throw s!"{k} is not a string"

end PouwCommit
