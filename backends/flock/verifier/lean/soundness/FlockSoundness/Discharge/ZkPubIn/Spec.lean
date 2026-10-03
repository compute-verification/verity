import FlockSoundness.Discharge.ZkPubIn.Defs
import FlockSoundness.Discharge.Hidden.Spec

/-!
# What `Stmt.setupHidden` accepts with the verifier's public inputs

`setupHidden_specP`: every statement `Stmt.setupHidden … (some own)` returns on a circuit with public input ports is
its `SpecP`, as `Hidden.setupHidden_spec'` is `Spec` at `publicInputs := none` (where a circuit with public input
ports is refused, `Hidden.checkOwn_none`).
-/

open Flock FlockLevel3
open Lean (Json)

namespace FlockSoundness.Discharge.ZkPubIn

/-- **What `Stmt.setupHidden` accepts with the verifier's public inputs**: a statement it returns on the verifier's own
recomputation `own`, on a circuit whose public input ports `HmIn.parse` reads as `pi`, is its `SpecP`; and the
circuit names a public input port, else `checkOwn` refuses `own`. -/
theorem setupHidden_specP {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    ∃ pi, SpecP tags cf pf coins tables draw own pi st := by
  sorry

end FlockSoundness.Discharge.ZkPubIn
