import Pous.SecurityProofs.SecureErasure.Digest
import Pous.SecurityProofs.SecureErasure.Reduction
import Pous.SecurityProofs.SecureErasure.Band
import Pous.SecurityProofs.SecureErasure.ReductionDraw
import Pous.SecurityProofs.SecureErasure.PagesAudit
import Pous.SecurityProofs.SecureErasure.Pages

/-!
# Secure erasure

* `SecureErasure/Digest.lean`: the fill check's digest is almost universal (`PousDigest.digestAU`), ported from
  PR #1010 without its bandwidth assumption.
* `SecureErasure/Reduction.lean`: secure erasure from any sequential audit bound, by counting
  (`PousErase.secure_erasure_meets`).
* `SecureErasure/Band.lean`: the band's audit at `2152` reads and `EraseMeets` at `f = 1/19`, `ε = 2^-128`, for every
  `1 ≤ N ≤ 2^64` segments and at one RTX PRO 6000's 96 GiB (`PousErase.secure_erasure_meets_family`,
  `PousErase.secure_erasure_meets_96GiB`).
* `SecureErasure/ReductionDraw.lean`: the same reduction over any read draw (`PousErase.secure_erasure_meets_draw`);
  at the uniform draw with replacement it is `Reduction.lean`'s, definitionally (`PousErase.auditSecure_iff_draw`).
* `SecureErasure/PagesAudit.lean`: the keyed page fill's audit at `D = 0` with distinct byte reads
  (`PousErase.page_audit_secure`), each page's Feistel10 an ideal permutation, as P3 idealizes Π₂.
* `SecureErasure/Pages.lean`: the page fill meets `EraseMeetsDistinct` at `f = 1/19`, `ε = 2^-128`, with `3430` reads
  for every `1 ≤ P ≤ 2^64` pages and `3046` reads at `P = 96112` (`PousErase.secure_erasure_pages_family`,
  `PousErase.secure_erasure_pages_96112`).

None of them uses a named assumption or `TwoTierBandwidth`.
-/
