import FlockSoundness.Discharge.DecodeZL.Wire
import FlockSoundness.Discharge.DecodeZL.Statements
import FlockSoundness.Discharge.DecodeZL.ZL
import FlockSoundness.Discharge.DecodeZL.Sizes
import FlockSoundness.Discharge.DecodeZL.Runs
import FlockSoundness.Discharge.DecodeZL.Assemble
import FlockSoundness.Discharge.DecodeZL.OP
import FlockSoundness.Discharge.DecodeZL.LIG

/-!
# R11b's message half: the per-Kind decoder, the chunk lemmas, and `DecodesOn` at the headline

* `Wire.lean`: the decoder (`Wire`, `decRound`, `decOf`) and the chunk lemma's shape (`Carries`, `Carries.append`).
* `Statements.lean`: the frozen statements.
* `ZL.lean`: `zl_carries`. `Sizes.lean`: `decodeProof_sizes`. `Runs.lean`: a decoded repetition's `RepRun`.
  `Assemble.lean`: `decodesOn_exec`. `OP.lean`: `op_carries`, the opening's chunk lemma at `wOP`. `LIG.lean`:
  `lig_carries`, Ligerito's chunk lemma at `wLIG`.
-/
