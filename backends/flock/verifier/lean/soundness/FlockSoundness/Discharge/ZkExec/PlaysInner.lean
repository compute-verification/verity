import FlockSoundness.Discharge.ZkExec.PlaysStatements
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysInner

/-!
# The inner proof's runs

`zk_inner_plays`: `Zk.verifyInner` on seven rows refines the model's compiled inner proof. After `final_c'` (absorbed,
no coin), the executable draws `ε`, absorbs `[ρ, σ]`, draws the ten batching coins (seven rows and the triple's
three), absorbs `τ`'s commitment, draws `β`, absorbs `Y` and draws the `Q` query words; the model draws and receives the
same, in order. What follows the last coin is checks only (`forIn_keeps`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Discharge.ZkExec
