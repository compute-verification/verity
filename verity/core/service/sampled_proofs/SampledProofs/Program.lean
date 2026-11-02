/-
The verifier's own program, from its descriptor's bytes (`verity-ir/descriptor/v1`): C-Flock's decoding of the whole program
(`Flock.Program.Prog`), whose root Calls `Q_word` and `Q_call` cut, and the template queries' reading of it
(`Flock.TemplateQuery.Desc.ofProg`). Bytes `Flock.Program.decode` refuses are refused.
-/
import Flock.Program
import Flock.TemplateQuery

namespace SampledProofs

open Flock (V)

structure Program where
  desc : Flock.TemplateQuery.Desc
  prog : Flock.Program.Prog
  deriving Inhabited

/-- The program its descriptor's bytes describe. -/
def Program.parse (bytes : ByteArray) : V Program := do
  let prog ← Flock.Program.decode bytes
  return { desc := .ofProg prog, prog }

/-- `partition_object.program_sha512`: SHA-512 of the canonical descriptor without `annotations`. -/
def Program.sha512 (p : Program) : ByteArray := p.desc.sha512

end SampledProofs
