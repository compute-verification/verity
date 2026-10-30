import Definitions.Core.Boolean
import Proofs.Core.Circuit

/-!
# The Boolean profile evaluates gate by gate

A Boolean circuit's evaluation, read as bits, applies each gate's Boolean operation to the bits of the gates it reads.
-/

namespace FlockSoundness.Audit

theorem eval_toCircuit (B : Circuit) (a : B.toCircuit.Transcript) {g : Fin B.N} (h : (B.op g).isInput = false) :
    (B.toCircuit.eval a g).getLsbD 0 = (B.op g).apply fun x => (B.toCircuit.eval a x).getLsbD 0 := by
  rw [B.toCircuit.eval_op a h]
  simp [Circuit.toCircuit]

end FlockSoundness.Audit
