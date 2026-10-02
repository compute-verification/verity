import FlockSoundness.Discharge.DecodeZL.Wire
import FlockSoundness.Discharge.DecodeZL.Statements
import FlockSoundness.Discharge.DecodeZL.ZL
import FlockSoundness.Discharge.DecodeZL.Sizes
import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Discharge.DecodeZL.Assemble

/-!
# R11b's message half: the per-Kind decoder, the zerocheck's and lincheck's chunks, and `DecodesOn` at the headline

* `Wire.lean`: the decoder (`Wire`, `decRound`, `decOf`) and the chunk lemma's shape (`Carries`, `Carries.append`).
* `Statements.lean`: the frozen statements.
* `ZL.lean`: `zl_carries`. `Sizes.lean`: `decodeProof_sizes`. `Runs.lean`: a decoded repetition's `RepRun`.
  `Assemble.lean`: `decodesOn_exec`.
-/
