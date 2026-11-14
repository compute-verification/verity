module

public import Difftest
public import FlockJsonDifftest
public import CoreProtocol.Protocol

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

end Core.CanonicalJsonDifftest
