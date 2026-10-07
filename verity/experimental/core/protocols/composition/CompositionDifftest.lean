/-
The difftest's fn over any `Core.Protocol`: `answer` takes one case, `{"commitment":HEX,"evidence":HEX,"params":JSON}`
(what the protocol's `serve` reads), and outputs `Protocol.answer`'s canonical outcome with its reason left out
(`"reason":null`), which a difftest doesn't compare. An input that isn't a case is the port's rejection.
-/
import CoreProtocol.Protocol
import Difftest

namespace Core.CompositionDifftest

open Lean (Json)

def answer (P : Protocol) : Difftest.Fn := fun input => do
  let params ← Flock.canon (← Flock.J.get input "params" "the case")
  let c ← Flock.unhex (← Flock.J.str input "commitment" "the case")
  let e ← Flock.unhex (← Flock.J.str input "evidence" "the case")
  pure { P.answer params.toUTF8 c e with reason := none }.toJson

end Core.CompositionDifftest
