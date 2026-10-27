import Definitions.Claims.Sampling
import SampledProofs.Partition

/-!
# Sampled proofs' size classes are the claims' (`SampledProofs.Partition.sizeClasses`)

Code never imports the claims, so the verifier holds its own copy of the public size classes of a proof block's
committed bits: `SampledProofs.Partition.sizeClassMax`, the widest, and `SampledProofs.Partition.sizeClasses`, the powers
of two from 16 up to it, which its refusals read (`SampledProofs.Partition.sizeClass`). `SizeClassesAgree` states that
both are the claims' (`Claims.Sampling.sizeClassMax`, `Claims.Sampling.sizeClasses`), so a change to one copy alone fails
its proof. Both sides are constants, so it takes no premise.
-/

namespace Properties.SampledProofs

/-- **The verifier's size classes are the claims'**: the widest public size class, and the classes themselves. -/
def SizeClassesAgree : Prop :=
  SampledProofs.Partition.sizeClassMax = Claims.Sampling.sizeClassMax ∧
    SampledProofs.Partition.sizeClasses = Claims.Sampling.sizeClasses

end Properties.SampledProofs
