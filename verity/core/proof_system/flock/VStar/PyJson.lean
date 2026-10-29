import Lean.Data.Json

/-!
# JSON as Python's `json.dumps(v, sort_keys=True, separators=(",", ":"))` writes it

`ir_lower.netlist` writes `LEAVES` and `circuit.compose` writes `META` this way, not as RFC 8785: every character outside
`' '..'~'` is escaped (`ensure_ascii`), as `\uXXXX` in lowercase hex, a pair of surrogates past the BMP, or the short forms
`\" \\ \b \f \n \r \t`; keys sort by code point. Only integers are accepted as numbers.
-/

namespace VStar

open Lean (Json)

def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

/-- `json.encoder.py_encode_basestring_ascii`. -/
def pyStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    let n := c.toNat
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | c => if n == 8 then out ++ "\\b" else if n == 12 then out ++ "\\f"
             else if 0x20 ≤ n && n ≤ 0x7e then out.push c
             else if n < 0x10000 then out ++ "\\u" ++ hex4 n
             else out ++ "\\u" ++ hex4 (0xd800 + ((n - 0x10000) >>> 10)) ++ "\\u" ++ hex4 (0xdc00 + (n - 0x10000) % 0x400)
  return out ++ "\""

/-- The text at nesting depth at most `fuel`. -/
def pyDumpsAt : Nat → Json → Except String String
  | 0, _ => throw "JSON nested too deeply"
  | _ + 1, .null => pure "null"
  | _ + 1, .bool b => pure (if b then "true" else "false")
  | _ + 1, .num n => if n.exponent == 0 then pure (toString n.mantissa) else throw "JSON: a non-integer number"
  | _ + 1, .str s => pure (pyStr s)
  | k + 1, .arr a => do
    let parts ← a.mapM (pyDumpsAt k)
    pure ("[" ++ ",".intercalate parts.toList ++ "]")
  | k + 1, .obj kvs => do
    let items := kvs.toArray.qsort (fun a b => a.1 < b.1)
    let parts ← items.mapM fun (key, v) => do pure (pyStr key ++ ":" ++ (← pyDumpsAt k v))
    pure ("{" ++ ",".intercalate parts.toList ++ "}")

def pyDumps (j : Json) : Except String String := pyDumpsAt 64 j

end VStar
