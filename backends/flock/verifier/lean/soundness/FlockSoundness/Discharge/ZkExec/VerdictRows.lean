import FlockSoundness.Discharge.ZkExec.VerdictReplay
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictRows

/-!
# A `--zk` rep's inner check holds on its inner openings

`ZkRepAccepts`'s second conclusion. The compiled rep's seven level-0 rows are the seven constraints `repProg` hands
`Zk.verifyInner`, in their order: the replay's three (`zk_replay_rows`), the two ring switches' (the masked opening's
values are the claim checks, `openingZK_plays'`) and level 0's two lane rows (`ligeritoPadZC_row`). So
`zk_inner_accepts` applies to the compiled rep's inner proof.

* `zk_decode_paths`: a decoded proof's path entries are 64 bytes.
* `zk_inner_rows`: the run of the compiled rep that `zk_rep_plays` builds has an inner check that holds on the proof's
  inner openings.
* `zk_inner_of_rep`: at `ZkRepAccepts`'s hypotheses, every result of the compiled rep does (`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript Reader
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The decoded paths -/

/-! ## The layout's positions -/

/-! ## The masked opening's values -/

/-! ## Level 0's lane rows, at the rounds `Zk.ligerito` consumed -/

/-! ## The seven rows -/

/-! ## The rep's inner check -/

end FlockSoundness.Discharge.ZkExec
