module

public import Proofs.Properties.Verity.Sampled.Record.KSByStatement
public import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# `KSByDeployed`'s witnesses (the Lean check's `nontrivial`)

* **The nontrivial lemma** (`ksByDeployed_nontrivial`): `KSBy` at SHA-512 is false at the sampled deployment whose
  verifier accepts everything (`Claims.Witness.Sampling.sksD`, its verifier `accV`), at `correctOk`, where `KSBy` is
  `KS` (`KSBy_correctOk`) and `KS` fails (`Claims.Witness.Sampling.KS_fails`).
* **The honest witness's statement** (`KSByDeployedHonest`): a live session meets every premise of `KSByDeployed` at
  the check `inputOk`, with no hypothesis. Its witness's prover registers nothing before the draw (`Live.Reg := Empty`),
  so the record half (`RecordsLiveZKJA` in `Live.hCust`, `Live.wr`, `Live.hr`) holds vacuously: no honest record is
  exercised, and the witness can't catch a contradiction that appears only with a nonempty record.
-/

@[expose] public section

namespace Claims.Sampling

/-- **`KSByDeployed`'s nontrivial lemma**: its conclusion, `KSBy` at SHA-512, is false at the sampled deployment whose
verifier accepts everything, at the check `correctOk`. -/
theorem ksByDeployed_nontrivial : ¬ KSBy sha512Msg (Claims.Witness.Sampling.sksD FlockDeploy.Coin) correctOk :=
  (KSBy_correctOk sha512Msg _).not.mpr (Claims.Witness.Sampling.KS_fails sha512Msg)

/-- **`KSByDeployed`'s honest witness's statement**: some live session, owner's files and extractor's reveal meet all of
`KSByDeployed`'s premises at the check `inputOk`: `sha512/row/v2` rows, the statement in scope at v2's slots, a
reference draw file, and each checked unit's Flock witness revealing openings that pass `inputOk`. -/
def KSByDeployedHonest : Prop :=
  ∃ (A : Verity.Sampled.Live) (o : Verity.Sampled.Record.Own) (e : Verity.Sampled.Record.Ext),
    Flock.Bridge.RowsV2 A.D.unroll ∧ Verity.Sampled.Record.ScopeSlotsOk A ∧ (∃ d, Verity.Sampled.Record.RefOk A o d) ∧
      SessionOpensBy sha512Msg (Verity.Sampled.Record.record A o e) (fun _ _ _ => A.roots) inputOk

end Claims.Sampling
