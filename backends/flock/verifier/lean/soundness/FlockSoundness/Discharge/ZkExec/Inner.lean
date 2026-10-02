import FlockSoundness.Discharge.ZkExec.Statements
import FlockSoundness.Refine.Exec

/-!
# flock-verify `--zk`'s inner proof accepts (live/PROTOCOL.md §5)

`verifyInner_ok : ZkInnerOk`: `Zk.verifyInner` accepts only with the pinned sizes, `β ≠ 0`, `τ` opening its
commitment, the batched constraint, and every queried column consistent with `Y` and climbing to the pads' root under
its `hm96` leaf.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine

theorem verifyInner_ok : ZkInnerOk := by
  sorry

end FlockSoundness.Discharge.ZkExec
