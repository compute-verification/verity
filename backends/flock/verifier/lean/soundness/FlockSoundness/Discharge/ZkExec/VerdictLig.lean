import FlockSoundness.Discharge.ZkExec.VerdictStatements
import FlockSoundness.Discharge.ZkExec.VerdictLigMain
import FlockSoundness.Discharge.ZkExec.PlaysRep
import FlockSoundness.Discharge.ZkExec.PlaysDet
import FlockSoundness.Discharge.ZkExec.RunsSeams
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLig

/-!
# `ZkRepAccepts`'s Ligerito half: a `--zk` rep's Ligerito verdict accepts the proof's rows

`ZkRepAccepts` (`VerdictStatements.lean`) has three conclusions at every result of the compiled rep: the Ligerito
verdict accepts the opened rows, the inner check holds, and every requested row verifies (`OpensOKZ`). This file is the
first: `ZkLigAccepts` takes exactly `ZkRepAccepts`'s hypotheses and concludes `pend.lig.verdict os.rows = true`.

The compiled masked phase's Ligerito (`ZkLink.ligeritoPadZC`, then `ZkLink.ligTailC`) is M0's compiled Ligerito
(`Model.recursiveRoundsC`, `Model.finalCheck`) at levels ≥ 1; at level 0 it batches the padded column
`ZkLink.padColumnO`: the witness lanes folded, `ρ₁·row[2^k₀+2r] + ρ₂·row[2^k₀+2r+1]`, and the pads' correction
`Model.padCorr` at `ȳ`, which is the executable's `Zk.padAt` at `PAD_C`.

`zk_ligerito_accepts` (`VerdictLigMain.lean`) is that refinement for one run of `Zk.ligerito`. Here `zk_rep_lig` carries
it through the rep as `zk_rep_plays` builds the rep's run, at the opening's batched values (`OpenRel`: the executable's
`basis` is the model's `bHat` at the opening's `b`), and a run has one result (`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

end FlockSoundness.Discharge.ZkExec
