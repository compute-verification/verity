module

public import Proofs.Properties.Verity.Sampled.Record.KSByStatement
public import Proofs.Flock.Claims.Witnesses.Sampling

/-!
# `KSByDeployed`'s witnesses (the Lean check's `nontrivial`)

* **The nontrivial lemma** (`ksByDeployed_nontrivial`): `KSBy` at SHA-512 is false at the sampled deployment whose
  verifier accepts everything (`Claims.Witness.Sampling.sksD`, its verifier `accV`), at `correctOk`, where `KSBy` is
  `KS` (`KSBy_correctOk`) and `KS` fails (`Claims.Witness.Sampling.KS_fails`).
-/

@[expose] public section

namespace Claims.Sampling

/-- **`KSByDeployed`'s nontrivial lemma**: its conclusion, `KSBy` at SHA-512, is false at the sampled deployment whose
verifier accepts everything, at the check `correctOk`. -/
theorem ksByDeployed_nontrivial : ¬ KSBy sha512Msg (Claims.Witness.Sampling.sksD FlockDeploy.Coin) correctOk :=
  (KSBy_correctOk sha512Msg _).not.mpr (Claims.Witness.Sampling.KS_fails sha512Msg)

end Claims.Sampling
