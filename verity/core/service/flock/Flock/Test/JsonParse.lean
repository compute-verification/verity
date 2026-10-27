/-
`JsonParse.parse` against `Lean.Json.parse`, the parser it replaces: on every case below and on 20,000 seeded texts, the
two give the same value (compared as `Json.compress` prints it, with each number's mantissa and exponent) or the same
refusal string, byte for byte. The cases cover each refusal of the source (its reason and offset), escapes and surrogate
pairs, numbers, nesting, whitespace, multi-byte characters (offsets are bytes) and trailing input. The kernel evaluates
`parseList` (`decide +kernel`), on a value and on a refusal.
-/
import Flock.JsonParse
import Lean.Data.Json

namespace Flock.Test.JsonParse

open Lean (Json)

/-- A parse as text: the value with every number's mantissa and exponent, or the refusal. -/
partial def shw : Except String Json → String
  | .ok j => "ok " ++ go j
  | .error e => "error " ++ e
where
  go : Json → String
    | .num n => s!"num({n.mantissa},{n.exponent})"
    | .arr a => "[" ++ ",".intercalate (a.toList.map go) ++ "]"
    | .obj m => "{" ++ ",".intercalate (m.toList.map fun (k, v) => Json.compress (.str k) ++ ":" ++ go v) ++ "}"
    | j => j.compress

def agrees (s : String) : Bool := shw (Lean.Json.parse s) == shw (JsonParse.parse s)

def cases : List String := [
  "", " ", "null", " null ", "nul", "nullx", "true", "false", "fals", "tru", "tRue", "[", "]", "[]", "[ ]", "[1,2]", "[1,]",
  "[1 2]", "[,]", "[1,2", "[-]", "[01]", "[[[[[]]]]]", "[1\n,\n2]", "[1,2,3] x",
  "{}", "{ }", "{\"a\":1}", "{\"a\":1,}", "{\"a\" 1}", "{a:1}", "{\"a\":1,\"a\":2}", "{\"b\":1,\"a\":[true,{}]}", "{\"a\"",
  "{\"a\":", "{\"a\":1", "{\"a\":1 \"b\":2}", "{\"a\":1}}", "{\"\":\"\"}", "{\"a\":{\"b\":{\"c\":null}}}",
  "  {\"k\" : [ 1 , 2 ] }  ", "\t\n\r null", "{\"x\":-0.000}",
  "\"abc\"", "\"a\\\"b\"", "\"\\u0041\"", "\"\\ud83d\\ude00\"", "\"\\ud83d\"", "\"\\ud83dx\"", "\"\\udc00\"",
  "\"\\ud83d\\u0041\"", "\"\\ud83d\\uDE00\"", "\"\\ud83d\\ue000\"", "\"\\ud83d\\", "\"\\ud83d\\u", "\"\\ud83d\\ud",
  "\"\\ud83d\\udz00\"", "\"\\uZZZZ\"", "\"\\x\"", "\"\\", "\"abc", "\"a\nb\"", "\"é ü 日本\"", "\"é\u0001\"", "é",
  "[\"日\" x]", "\"\\u00e9\"", "\"\\u12\"", "\"\\u12", "[\"\\/\\b\\f\\n\\r\\t\"]", "\u0001", "\"\u0001\"", "nul\u0000",
  "0", "-0", "01", "-", "-a", "1.", "1.5", "1.50", "1e5", "1E+5", "1e-5", "1e", "1e+", "1e-", "1ex", "-1.25e-3",
  "12345678901234567890", "1.0e400", "0.1", ".5", "+1", "1x", "1e99999999999999999999999", "1.5e3", "-1.5e-0", "1e0",
  "1.2.3", "1ee", "- 1", "-01", "0e", "0.e1", "1.5E-07"]

#guard cases.all agrees

def toks : Array String := #["{", "}", "[", "]", ",", ":", " ", "\n", "\"", "\\", "\\u", "d83d", "de00", "DC00", "0041",
  "u", "a", "é", "日", "0", "1", "9", "-", "+", ".", "e", "E", "true", "false", "null", "\"k\"", "\"v\"", "\t", "\u0001",
  "x", "12", "\\n", "\\\"", "\\/", "fa", "nu", "1e", "\\ud83d", "\\ude00", "\\uDE00", "D", "f"]

/-- Text `seed` of the fuzz: `n` tokens drawn from `toks`. -/
def gen (seed n : Nat) : String := Id.run do
  let mut g := mkStdGen seed
  let mut s := ""
  for _ in [0:n] do
    let (k, g') := randNat g 0 (toks.size - 1)
    g := g'
    s := s ++ toks[k]!
  return s

#guard (List.range 20000).all fun seed => agrees (gen seed (seed % 12 + 1))

/-- `parseList`'s value at a key of an object, or its refusal. -/
def keyOf (cs : List Char) (k : String) : Option Json := match JsonParse.parseList cs with
  | .ok (.obj m) => m.get? k
  | _ => none

example : (match keyOf ['{', '"', 'a', '"', ':', '[', '1', ',', '-', '2', '.', '5', 'e', '1', ']', '}'] "a" with
    | some (.arr a) => a.size == 2
    | _ => false) = true := by decide +kernel

example : (match JsonParse.parseList ['[', '1', ',', ']'] with
    | .error e => e == "offset 3: unexpected input"
    | .ok _ => false) = true := by decide +kernel

end Flock.Test.JsonParse
