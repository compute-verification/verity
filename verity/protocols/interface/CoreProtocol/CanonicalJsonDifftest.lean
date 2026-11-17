module

public import Difftest
public import FlockJsonDifftest
public import CoreProtocol.Protocol
public import CoreProtocol.Cli
meta import CoreProtocol.Cli

/-!
# The canonical reader's difftest: `Core.canonicalJson` under each JSON parser

`core-canonical-json-difftest` serves `table` to the core difftest harness (`tools/difftest`), whose Python half
(`tools/difftest/tests/test_canonical_json_port.py`) holds it to its mirror:

* `canonical-parsers` (`text`, bytes in lowercase hex): `{"core": A, "total": A}`, what `Core.canonicalJson` answers on
  those bytes when their text is read with Lean core's JSON parser and when it is read with the total one
  (`Flock.JsonParse.parse`, the parser it reads with). An answer is `["ok", V]`, `V` the value with nothing lost
  (`FlockJsonDifftest.enc`), or `["error", REASON]`: a refusal is an output, so its text is compared too.
-/

@[expose] public section

namespace Core.CanonicalJsonDifftest

open Lean (Json)
open Flock (V)

/-- `Core.canonicalJson`'s text with its parser a parameter. -/
def canonicalJsonWith (parse : String → Except String Json) (b : ByteArray) : V Json := do
  unless b.data.all (· < 0x7f) do throw "not ASCII without DEL (a byte at or above 0x7f)"
  let some s := String.fromUTF8? b | throw "not UTF-8"
  let j ← match parse s with
    | .ok j => pure j
    | .error e => throw s!"not JSON ({e})"
  if (← Flock.canon j) != s then throw "not canonical JSON (compact, keys sorted, integers only)"
  pure j

/-- Read with the total parser, the copy is `Core.canonicalJson`, so `canonical-parsers`' two answers differ in the
parser alone. -/
theorem canonicalJsonWith_total : canonicalJsonWith Flock.JsonParse.parse = Core.canonicalJson := rfl

def canonicalParsers : _root_.Difftest.Fn := fun i => do
  let b ← Flock.unhex (← FlockJsonDifftest.text i)
  return Json.mkObj [("core", FlockJsonDifftest.answer (canonicalJsonWith Lean.Json.parse b)),
    ("total", FlockJsonDifftest.answer (Core.canonicalJson b))]

def table := _root_.Difftest.table [("canonical-parsers", canonicalParsers)]

/-! `canonicalJson` reads its text with `Flock.JsonParse.parse`. On every text of `TEXTS` (`CoreProtocol.Cli`'s) and
`ADVERSARIAL`, Lean core's `Lean.Json.parse` gives the same value or the same refusal, and `canonicalJson` the same
answer it gives reading with `Lean.Json.parse` (`canonicalJsonWith Lean.Json.parse`). `ADVERSARIAL` holds each class of
text on which two JSON readers commonly differ: duplicate keys, exponents, fractions, leading zeros, negative zero,
escapes and surrogates, whitespace, trailing input, nesting, empty input, literals and non-ASCII. -/

/-- A value with each number's mantissa and exponent and each object's members in its map's order. -/
private def shown : Json → String
  | .null => "null"
  | .bool b => toString b
  | .num n => s!"num({n.mantissa},{n.exponent})"
  | .str s => s.quote
  | .arr a => "[" ++ ",".intercalate (a.attach.map fun x => shown x.1).toList ++ "]"
  | .obj kvs => "{" ++ ",".intercalate (kvs.toArray.attach.map fun p => p.1.1.quote ++ ":" ++ shown p.1.2).toList ++ "}"
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact Flock.sizeOf_lt_of_mem_toArray p.2

private def answerShown : Except String Json → String
  | .ok j => "ok " ++ shown j
  | .error e => "error " ++ e

/-- `mid` inside `n` of `o` and `n` of `c`. -/
private def nest (n : Nat) (o c mid : String) : String :=
  String.join (List.replicate n o) ++ mid ++ String.join (List.replicate n c)

private def ADVERSARIAL : List String := [
  "{\"a\":1,\"a\":2}", "{\"a\":2,\"a\":1}", "{\"a\":1,\"a\":1}", "{\"a\":{\"b\":1},\"a\":{\"b\":1}}", "{\"b\":0,\"a\":0,\"b\":0}",
  "1e2", "1E2", "1e+2", "1e-2", "1e0", "0e0", "-1e2", "1e", "1e+", "1e-", "1E400", "1.0e1", "1e18446744073709551617",
  "1e-18446744073709551617", "100", "1.0", "1.5", "0.0", "-0.0", "1.", ".5", "[1.50]", "00", "007", "-01", "[01]",
  "{\"a\":01}", "0.01", "-0", "[-0]", "-0e0", "{\"a\":-0}", "0", "-1", "\"\\/\"", "\"\\u002f\"", "\"/\"", "\"\\u0041\"",
  "\"A\"", "\"\\u000a\"", "\"\\n\"", "\"\\U0041\"", "\"\\'\"", "\"\\a\"", "\"\\u00\"", "\"\\uGGGG\"", "\"\\u001f\"",
  "\"\\u001F\"", "\"\\ud800\"", "\"\\udfff\"", "\"\\ud800\\udc00\"", "\"\\uD800\\uDC00\"", "\"\\ud800\\ud800\"",
  "\"\\udc00\\ud800\"", "\"\\ud800\\u0041\"", "\"\\ud800x\"", "\"\\ufffd\"", "\"\\ud83d\\ud83d\\ude00\"",
  "\"\\ufffd\\ud83d\\ude00\"", "\"\\udbff\\udfff\"", "\"\\u0000\"", "\"a\u0001b\"", "\"a\tb\"", " 1", "1 ", "\t1", "1\n",
  "\r\n1", "[1 ,2]", "{\"a\" :1}", "\u000b1", "\u000c1", "\u00a01", "1 2", "{}{}", "[]]", "nullnull", "1,", "\"a\"\"b\"",
  "{} x", "0\u0000", "\u0000", "", " ", "\n", "True", "NaN", "Infinity", "-Infinity", "+1", "nulll", "undefined",
  nest 200 "[" "]" "0", nest 200 "{\"a\":" "}" "0", nest 200 "[" "" "", nest 200 "[" "]" "", "\"é\"", "é", "\"日本\"",
  "\"😀\"", "\u007f", "\"\u007f\"", "[\"\\u00e9\"]"]

#guard (TEXTS ++ ADVERSARIAL).all fun t => answerShown (Lean.Json.parse t) == answerShown (Flock.JsonParse.parse t)
#guard (TEXTS ++ ADVERSARIAL).all fun t => answerShown (canonicalJsonWith Lean.Json.parse t.toUTF8) == answerShown (canonicalJson t.toUTF8)
#guard [#[0xff], #[0xc3, 0x28], #[0x22, 0xed, 0xa0, 0x80, 0x22]].all fun b =>
  answerShown (canonicalJsonWith Lean.Json.parse ⟨b⟩) == answerShown (canonicalJson ⟨b⟩)
#guard (ADVERSARIAL.filter fun t => (canonicalJson t.toUTF8).toOption.isSome).length == 16

end Core.CanonicalJsonDifftest
