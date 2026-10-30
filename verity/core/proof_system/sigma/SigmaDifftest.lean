import Difftest
import Flock.Sigma

/-!
# The σ-lift check's difftest executable

The functions `tests/test_sigma_difftest.py` holds to the Python: `check` is `Flock.Sigma.check` on a word descriptor and a
Boolean one (`{"word", "bits"}`, each the descriptor's JSON), its report or its refusal's code and place (the reason's text
is the Lean's own), and a descriptor either side's decoder refuses a reject; `bit-offsets` is `Flock.Sigma.bitOff` of a type
at the given leaves.
-/

namespace SigmaDifftest

open Lean (Json)

def check (j : Json) : Except String Json := do
  let (.ok w, .ok b) := (j.getObjVal? "word", j.getObjVal? "bits") | throw "input: {\"word\", \"bits\"}"
  match Flock.Sigma.checkBytes w.compress.toUTF8 b.compress.toUTF8 with
  | .ok r => pure r.toJson
  | .error e =>
    if e.code == "decode" then throw e.why
    pure (Json.mkObj ([("ok", Json.bool false), ("code", Json.str e.code)] ++
      (e.call.map fun c => ("call", Lean.toJson c)).toList ++ (e.operand.map fun g => ("operand", Lean.toJson g)).toList ++
      (e.definition.map fun d => ("definition", Json.str d)).toList))

def bitOffsets (j : Json) : Except String Json := do
  let (.ok t, .ok (.arr js)) := (j.getObjVal? "type", j.getObjVal? "js") | throw "input: {\"type\", \"js\"}"
  let t ← Flock.Program.Ty.ofJson t
  let js ← js.mapM fun x => match x.getNat? with
    | .ok n => pure n
    | .error _ => throw s!"js: {x.compress} is not a leaf index"
  pure (Json.arr (js.map fun k => Lean.toJson (Flock.Sigma.bitOff t k)))

end SigmaDifftest

def main : IO Unit := Difftest.serve (Difftest.table [("check", SigmaDifftest.check), ("bit-offsets", SigmaDifftest.bitOffsets)])
