import PouwWindow.Entry
import PouwCommit.Py
import Lean.Data.Json

/-!
# A call's record (`window.py`, `record_digest` and `record_problem`)

The entry binds its call's record by the SHA-512 of its canonical JSON (`canonical_json`: keys sorted, compact,
`ensure_ascii`, so every character outside space … `~` is a `\uXXXX` escape, above U+FFFF a surrogate pair), and
`recordProblem` holds the record's `shape` [m, k, n], `tile0` and `keep` to the entry. A record is any JSON value, so
`recordProblem` reads it as Python does: an integer where Python computes is an integer or a boolean (Python's `True` is 1),
and the reasons print values as Python's `str` does (`pyStr`).
-/

namespace PouwWindow

open Lean (Json)

private def hex4 (n : Nat) : String :=
  String.ofList ((List.range 4).reverse.map fun k => Nat.digitChar (n / 16 ^ k % 16))

/-- `json.dumps(s, ensure_ascii=True)` of a string. -/
def pyQuote (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    let n := c.toNat
    out := if c == '"' then out ++ "\\\"" else if c == '\\' then out ++ "\\\\" else if c == '\n' then out ++ "\\n"
      else if c == '\r' then out ++ "\\r" else if c == '\t' then out ++ "\\t" else if n == 8 then out ++ "\\b"
      else if n == 12 then out ++ "\\f"
      else if n < 0x20 || n ≥ 0x7f then
        if n < 0x10000 then out ++ "\\u" ++ hex4 n
        else out ++ "\\u" ++ hex4 (0xD800 + (n - 0x10000) / 0x400) ++ "\\u" ++ hex4 (0xDC00 + (n - 0x10000) % 0x400)
      else out.push c
  return out ++ "\""

/-- How deep `pyCanon` and `pyRepr` nest: Python's own recursion limit (1,000 frames) stops `json.dumps` well before. -/
def pyDepth : Nat := 10000

private def pyCanonAux : Nat → Json → Except String String
  | 0, _ => throw "canonical JSON: nested too deep"
  | _ + 1, .null => pure "null"
  | _ + 1, .bool b => pure (if b then "true" else "false")
  | _ + 1, .num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "canonical JSON: a non-integer number"
  | _ + 1, .str s => pure (pyQuote s)
  | d + 1, .arr a => do pure ("[" ++ ",".intercalate (← a.toList.mapM (pyCanonAux d)) ++ "]")
  | d + 1, .obj kvs => do
    let items := kvs.toArray.qsort (fun a b => a.1 < b.1)
    let parts ← items.toList.mapM fun (k, v) => do pure (pyQuote k ++ ":" ++ (← pyCanonAux d v))
    pure ("{" ++ ",".intercalate parts ++ "}")

/-- `canonical_json` (`json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)`), refusing a non-integer
number, which Python would write as a float. -/
def pyCanon (j : Json) : Except String String := pyCanonAux pyDepth j

/-- `record_digest`: the SHA-512 of the record's canonical JSON. -/
def recordDigest (record : Json) : Except String ByteArray := do
  pure (Flock.Sha512.hash (← pyCanon record).toUTF8)

private def pyReprAux : Nat → Json → String
  | 0, _ => "..."
  | _ + 1, .null => "None"
  | _ + 1, .bool b => if b then "True" else "False"
  | _ + 1, .num n => if n.exponent == 0 then toString n.mantissa else toString n
  | _ + 1, .str s => PouwCommit.pyRepr s
  | d + 1, .arr a => "[" ++ ", ".intercalate (a.toList.map (pyReprAux d)) ++ "]"
  | d + 1, .obj kvs =>
    "{" ++ ", ".intercalate (kvs.toArray.toList.map fun (k, v) => PouwCommit.pyRepr k ++ ": " ++ pyReprAux d v) ++ "}"

/-- Python's `repr` of a JSON value: an object's items in the order Lean holds them, which is sorted. -/
def pyRepr (j : Json) : String := pyReprAux pyDepth j

/-- Python's `str` of a JSON value. -/
def pyStr : Json → String
  | .str s => s
  | j => pyRepr j

/-- A value Python computes with as an integer: an integer, or a boolean. -/
def pyInt? : Json → Option Int
  | .num ⟨m, 0⟩ => some m
  | .bool b => some (if b then 1 else 0)
  | _ => none

/-- An integer and not a boolean (`isinstance(i, int) and not isinstance(i, bool)`). -/
def strictInt? : Json → Option Int
  | .num ⟨m, 0⟩ => some m
  | _ => none

/-- `list(x)` of a JSON value: a list's items, a string's characters, an object's keys; `none` where Python raises. -/
def pyList? : Json → Option (List Json)
  | .arr a => some a.toList
  | .str s => some (s.toList.map fun c => .str c.toString)
  | .obj kvs => some (kvs.toArray.toList.map fun (k, _) => .str k)
  | _ => none

/-- `x[k]` of a JSON object; `none` where Python raises. -/
def pyGet? (x : Json) (k : String) : Option Json :=
  match x with
  | .obj kvs => kvs.get? k
  | _ => none

/-- Strictly increasing. -/
def increasing : List Int → Bool
  | a :: b :: rest => a < b && increasing (b :: rest)
  | _ => true

/-- `-(-n // 64)`: ⌈n/64⌉ in Python's floor division. -/
def ceilTile (n : Int) : Int := -((-n).fdiv tile)

/-- `record_problem`: why a call's record is not its entry's, or `none`. Python's `except (KeyError, TypeError,
ValueError)` is the last reason, reached wherever Python would raise one of them. -/
def recordProblem (e : Entry) (record : Json) : Except String (Option String) := do
  if (← recordDigest record) != e.record then return some "its record is not the one the entry binds"
  let noShape := some "its record has no shape [m, k, n], tile0 and keep {a, b}"
  let some (.arr #[m, _, n]) := pyGet? record "shape" | return noShape
  let (some tile0, some keep) := (pyGet? record "tile0", pyGet? record "keep") | return noShape
  let some nInt := pyInt? n | return noShape
  let cols := ceilTile nInt
  if !(pyInt? tile0 == some e.tile0 && cols == e.cols && pyInt? m == some e.m) then
    return some s!"its tile0, cols and m are not its record's {pyStr tile0}, ⌈{pyStr n}/64⌉ and {pyStr m}"
  for (side, count, top) in [(Side.a, e.nA, m), (Side.b, e.nB, n)] do
    let some rows := (pyGet? keep side.name).bind pyList? | return noShape
    let bad := some s!"its record's side-{side.name} keep is not {count} increasing indices of [0, {pyStr top})"
    if rows.length != count then return bad
    let some ints := rows.mapM strictInt? | return bad
    if !increasing ints then return bad
    match ints.head?, ints.getLast?, pyInt? top with
    | some first, some last, some t => if !(0 ≤ first && last < t) then return bad
    | some _, some _, none => return noShape
    | _, _, _ => pure ()
  return none

end PouwWindow
