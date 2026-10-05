import FlockSoundness.Refine.Opening
import FlockSoundness.Discharge.ZkExec.PlaysReplay
import FlockSoundness.Discharge.ZkExec.PlaysLig
import FlockSoundness.Discharge.ZkExec.PlaysInner
import FlockSoundness.Discharge.ZkExec.RunsFrames
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysRep

/-!
# `repProg`'s run is a run of the compiled `--zk` rep

`zk_rep_plays`: the plays half of `--zk`'s `rep_refines`, composed phase by phase as M0's `rep_refines` is. The pads
root is the rep's first message; `zk_replay_plays` gives the masked zerocheck's and lincheck's runs; M0's
`opening_refines` gives the opening's on the two claims at their claim checks and the extras, and `openingZK_plays`
carries it to the masked opening, which recomputes the two claims' values from the switched `s_hat_v` (`opResult_two`);
`zk_ligerito_plays` gives level 0's; `final_c'` is received; and `zk_inner_plays` gives the inner proof's on the seven
rows (the replay's three, `replay_cons_size`, the ring switch's two and the lane phase's two).

A run of a `bind` splits (`plays_bind_inv`), which is how the opening's run is taken apart.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

section Inv

variable {W F : Type} [Inhabited W] {ι : W → F} {α β : Type}

end Inv

/-! ## The masked opening -/

/-! ## The masked replay's constraints -/

/-! ## Runs with a property of their result -/

variable {α β : Type}

/-! ## The rep -/

end FlockSoundness.Discharge.ZkExec
