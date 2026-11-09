module

public import Proofs.Flock.Soundness.Discharge.DecodeZL.Wire
public import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements
public import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL
public import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes
public import Proofs.Flock.Soundness.Discharge.DecodeZL.Runs
public import Proofs.Flock.Soundness.Discharge.DecodeZL.Assemble
public import Proofs.Flock.Soundness.Discharge.DecodeZL.OP
public import Proofs.Flock.Soundness.Discharge.DecodeZL.LIG

/-!
# The decoder premise's message half: the per-Kind decoder, the chunk lemmas, and `DecodesOn` at the headline

* `Wire.lean`: the decoder (`Wire`, `decRound`, `decOf`) and the chunk lemma's shape (`Carries`, `Carries.append`).
* `Statements.lean`: the frozen statements.
* `ZL.lean`: `zl_carries`. `Sizes.lean`: `decodeProof_sizes`. `Runs.lean`: a decoded repetition's `RepRun`.
  `Assemble.lean`: `decodesOn_exec`. `OP.lean`: `op_carries`, the opening's chunk lemma at `wOP`. `LIG.lean`:
  `lig_carries`, Ligerito's chunk lemma at `wLIG`.
-/

@[expose] public section
