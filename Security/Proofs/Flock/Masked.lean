import Proofs.Flock.Masked.Statement
import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.ZK.Assemble

/-!
# Proofs of C-Flock's masked guarantees

* `Flock.SecurityProofs.MaskedKS : Flock.Guarantees.MaskedKS` is `FlockDeploy.KS.zk_KS_untrim` (G1).
* `Flock.SecurityProofs.MaskedZK : Flock.Guarantees.MaskedZK` is `Claims.ZKMal.Flock.zk_deployZk_masked_exe_split` at
  `FlockDeploy.sha512`, whose `hOf` is SHA-512 (`FlockDeploy.hOf_sha512`) (G2).
-/

namespace Flock.SecurityProofs

/-- **C-Flock's masked record is knowledge sound** at SHA-512 at the deployed relation, with no premise. -/
theorem MaskedKS : Flock.Guarantees.MaskedKS :=
  FlockDeploy.KS.zk_KS_untrim

/-- **C-Flock's deployed masked session is zero knowledge** at SHA-512, at every `nr` and every `0 < t`, with no other
premise. -/
theorem MaskedZK : Flock.Guarantees.MaskedZK :=
  fun nr t ht => Claims.ZKMal.Flock.zk_deployZk_masked_exe_split FlockDeploy.sha512 FlockDeploy.hOf_sha512 nr t ht

end Flock.SecurityProofs
