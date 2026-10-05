import FlockSoundness.Discharge.ZkExec.Coin
import FlockSoundness.Discharge.DecodeZL.LIG
import FlockSoundness.Discharge.DecodeZL.OP
import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Protocol.Discharge.ZkExec.RunsStatements

/-!
# `RunsZC`'s decoder half: the `--zk` rep's frames and the messages they carry (statements)

`RunsZC` (`Coin.lean`) asks each rep for a `RepRun` along `repShapeZ`: a run on its rounds' coins whose messages, cut
at the spine's coins, are the decoder's groups of the rounds. At M0 the decoder reads each round's frames by its wires
(`DecodeZL.decOf`), and `DecodeZL.repRun_at` glues the phases' chunk lemmas (`zl_carries`, `op_carries`,
`lig_carries`). Under `--zk` the rep's frames (`repFramesZ`) differ from M0's in four places, and the rest is M0's:

* round 0 also absorbs the pads' label and root before the zerocheck's first coin (`zcHeadZ`), so it carries the root;
* Ligerito absorbs `LABEL_OPEN` after the cap, and its last lane round's coin is followed by level 0's hiding frames
  (`l0Fr`: the pair's sums, the pair's two coins `ρ`, `LABEL_CLOSE`, `T'` and `ybar`) before the round's message
  (`ligFramesZ`);
* the inner proof follows Ligerito (`innerFr`): `final_c'`, `ε`, `[ρ, σ]`, `γ`, `τ`'s commitment, `β`, `Y` and the
  queries.

The decoder `decZ` reads round 0's root at its fixed place before the closing squeeze (`rootOf`), and every later round
by its wires (`wRestZ`: M0's after round 0, the opening's, `wLIGZ` and `wInner`).

* `ZkScheduleSeams`: the `--zk` schedule's dimensions are the ones the frames and wires read.
* `ZkRepFrames`: `repProg` (the transcript program `Zk.verifyRep` runs) emits `repFramesZ`, one chunk per round.
* `ZkDecodeSizes`: a decoded `--zk` proof has M0's lengths (`DecodeZL.ProofSizes`), and `Y` fewer than `2^64` words.
* `ZkCarries`: the frames after round 0, read by `wRestZ`, are the messages after the root, cut at the spine's coins.
* `ZkCut`: on rounds that retained `repFramesZ`'s chunks, the rep's messages, cut along `shapeZE` (`repShapeZ` at the
  executable's dimensions), are `decZ`'s groups of the rounds, with no tail: `RepRun`'s cut.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The frames -/

/-! ## The messages -/

/-! ## The wires and the decoder -/

/-! ## Statements -/

end FlockSoundness.Discharge.ZkExec
