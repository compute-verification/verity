module

public import Difftest
public import Base.Bytes
public import Circuits.Program

/-!
# The program decoder's port: `Flock.Program.decode` behind `Difftest.serve`

`program-difftest` serves the decoder of `verity/circuits/PROTOCOL.md` §4, schema v2 with its Boolean map (§4.9), so that
`codec.decode_program` is held to it on the format's vectors (`tests/tools/difftest/test_program_decode_port.py`, its cases
`tests/verity/circuits/decode_difftest.py`'s). `decode` reads `{"descriptor": TEXT}` and answers
`{"ok": true, "program_digest": HEX, "boolean": MAP, "inputs": INPUTS, "definition_digests": {ID: HEX}}` (§1, §4.9, §5.1,
§4.8) or `{"ok": false, "error": CODE}`, CODE the refusal's first `E_` code
or null: a refusal is an output, so which refusal is compared, and not only the fact of one. -/

@[expose] public section

namespace ProgramDifftest

open Lean (Json)

/-- The first `E_[A-Z_]+` of `s` that starts a word (`re.search(r"\bE_[A-Z_]+", s)`), or none. -/
def code? (s : String) : Option String := Id.run do
  let cs := s.toList.toArray
  let word := fun (c : Char) => c.isAlphanum || c == '_'
  for i in [0:cs.size] do
    if cs[i]! == 'E' && cs[i + 1]? == some '_' && (i == 0 || !word cs[i - 1]!) then
      let mut j := i + 2
      while j < cs.size && (cs[j]!.isUpper || cs[j]! == '_') do j := j + 1
      return some (String.ofList (cs.extract i j).toList)
  return none

def decode : Difftest.Fn := fun j => do
  let text ← match j.getObjValAs? String "descriptor" with
    | .ok s => pure s
    | .error _ => throw "descriptor: not a string"
  return match Flock.Program.decode text.toUTF8, (Flock.JsonParse.parse text).toOption with
    | .ok P, some j =>
      let digests := match Flock.Program.definitionDigests j with
        | .ok d => Json.mkObj ((d.toList.mergeSort (fun a b => a.1 ≤ b.1)).map fun (k, v) => (k, Json.str v))
        | .error e => Json.str e
      Json.mkObj [("ok", true), ("program_digest", Flock.hexOf P.sha512),
        ("boolean", Json.mkObj ((P.boolean.toList.mergeSort (fun a b => a.1 ≤ b.1)).map fun (k, v) => (k, Json.str v))),
        ("inputs", j.getObjValD "inputs"), ("definition_digests", digests)]
    | .ok _, none => Json.mkObj [("ok", false), ("error", Json.null)]
    | .error e, _ => Json.mkObj [("ok", false), ("error", match code? e with | some c => Json.str c | none => Json.null)]

def table : String → Option Difftest.Fn := Difftest.table [("decode", decode)]

end ProgramDifftest
