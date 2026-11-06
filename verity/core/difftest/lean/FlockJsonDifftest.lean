import Difftest
import Flock.JsonParse

/-! The C-Flock verifier's JSON parser, `Flock.JsonParse.parse`, held to its Python mirror
(`tests/test_flock_json_port.py`), and Lean core's `Lean.Json.parse`, which it reads as, held to the same mirror. Each
reads the text of `{"text": T}` and answers `["ok", V]` or `["error", REASON]`: a refusal is an output, so its text is
compared too, and `V` is the parsed value in canonical JSON with nothing lost (`enc`). C-Flock's package has no
dependencies, so its port lives beside the harness. -/

namespace FlockJsonDifftest

open Lean (Json toJson)

/-- A parsed value with nothing lost: `["null"]`, `["bool", b]`, `["num", mantissa, exponent]`, `["str", s]`,
`["arr", [..]]`, and `["obj", [[key, value], ..]]` in the tree's order. -/
def enc : Json → Json
  | .null => Json.arr #[.str "null"]
  | .bool b => Json.arr #[.str "bool", .bool b]
  | .num n => Json.arr #[.str "num", toJson n.mantissa, toJson n.exponent]
  | .str s => Json.arr #[.str "str", .str s]
  | .arr a => Json.arr #[.str "arr", Json.arr (a.attach.map fun x => enc x.1)]
  | .obj kvs => Json.arr #[.str "obj", Json.arr (kvs.toArray.attach.map fun p => Json.arr #[.str p.1.1, enc p.1.2])]
termination_by j => sizeOf j
decreasing_by
  · have := Array.sizeOf_lt_of_mem x.2
    simp only [Json.arr.sizeOf_spec]
    omega
  · exact Flock.sizeOf_lt_of_mem_toArray p.2

/-- A parser's answer. -/
def answer : Except String Json → Json
  | .ok v => Json.arr #[.str "ok", enc v]
  | .error e => Json.arr #[.str "error", .str e]

/-- The text to parse. -/
def text (j : Json) : Except String String :=
  match j.getObjValAs? String "text" with
  | .ok s => .ok s
  | .error _ => .error "text: not a string"

def parse : Difftest.Fn := fun j => return answer (Flock.JsonParse.parse (← text j))

def parseCore : Difftest.Fn := fun j => return answer (Lean.Json.parse (← text j))

def table : String → Option Difftest.Fn := Difftest.table [("parse", parse), ("parse-core", parseCore)]

end FlockJsonDifftest
