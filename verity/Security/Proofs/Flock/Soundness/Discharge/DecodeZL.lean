import Proofs.Flock.Soundness.Discharge.DecodeZL.Wire
import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements
import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes
import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs
import Proofs.Flock.Soundness.Discharge.DecodeZL.Assemble
import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG

/-!
# R11b's message half: the per-Kind decoder, the chunk lemmas, and `DecodesOn` at the headline

* `Wire.lean`: the decoder (`Wire`, `decRound`, `decOf`) and the chunk lemma's shape (`Carries`, `Carries.append`).
* `Statements.lean`: the frozen statements.
* `ZL.lean`: `zl_carries`. `Sizes.lean`: `decodeProof_sizes`. `Runs.lean`: a decoded repetition's `RepRun`.
  `Assemble.lean`: `decodesOn_exec`. `OP.lean`: `op_carries`, the opening's chunk lemma at `wOP`. `LIG.lean`:
  `lig_carries`, Ligerito's chunk lemma at `wLIG`.
-/
