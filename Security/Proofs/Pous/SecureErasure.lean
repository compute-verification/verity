import Proofs.Pous.SecureErasure.Digest
import Proofs.Pous.SecureErasure.Reduction
import Proofs.Pous.SecureErasure.Band

/-!
# Secure erasure

* `SecureErasure/Digest.lean`: the fill check's digest is almost universal (`PousDigest.digestAU`), ported from
  the older bandwidth-separated proof without its bandwidth assumption.
* `SecureErasure/Reduction.lean`: secure erasure from any sequential audit bound, by counting
  (`PousErase.secure_erasure_meets`).
* `SecureErasure/Band.lean`: the band's audit at `2152` reads and `EraseMeets` at `f = 1/19`, `ε = 2^-128`, for every
  `1 ≤ N ≤ 2^64` segments and at one RTX PRO 6000's 96 GiB (`PousErase.secure_erasure_meets_family`,
  `PousErase.secure_erasure_meets_96GiB`).

None of them uses a named assumption or `TwoTierBandwidth`.
-/
