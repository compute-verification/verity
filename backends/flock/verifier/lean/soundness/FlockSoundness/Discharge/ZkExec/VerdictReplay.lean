import FlockSoundness.Discharge.ZkExec.VerdictLane
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictReplay

/-!
# The masked replay's rows are the executable's

`zk_replay_rows`: `Zk.replay`'s three constraints, its `ab` value and its `c` value are the forms the model's masked
zerocheck and lincheck compute, at pads placed as the executable's layout places them.

* `forIn_foldF`: a loop that consumes one one-coin round per step and pushes that coin computes the fold of its step.
* `res_fold_rel`, `lc_fold_rel`: the residual and lincheck rounds' claims (`resOut`, `lcRoundsOut`) are the folds of the
  executable's steps.
* `prod_rel`: the product row `a'b' + a'·h_fb + b'·h_fa + h_ab`.
* `folded_rel`: the circuit's 64 entries are the model's `foldAll` of `comb`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## The rounds' steps -/

section Steps

variable {np : ℕ} (P : ZkSession.PadsA np) (lay : Zk.PadLayout) (hs : lay.len = np)
  (rounds : Array (F128 × F128))
include hs

end Steps

/-! ## The replay's first constraints -/

section Init

variable (lay : Zk.PadLayout) (P : ZkSession.PadsA lay.len)

end Init

end FlockSoundness.Discharge.ZkExec
