module

public import Proofs.Properties.Verity.Sampled.Record.KSByStatement
public import Proofs.Properties.Verity.Sampled.Record.KS

/-!
# `KSByDeployed`, proved

`Claims.Sampling.KSByDeployed` is `Verity.Sampled.Record.ksBy_record`: `Claims.Sampling.KSBy_of` at the record, with
binding, `SessionKS`, `InteriorFixed`, `DrawLaw`, `Refuses` and the ε table's sign discharged at the deployed record.
-/

@[expose] public section

namespace Verity.SecurityProofs

/-- **Sampling's KS at a check on each proof block, at one-stage v2's deployed sampled verifier.** -/
theorem KSByDeployed : Claims.Sampling.KSByDeployed :=
  fun A o e hv2 hscope href Ok hopens => Verity.Sampled.Record.ksBy_record A o e hv2 hscope href Ok hopens

end Verity.SecurityProofs
