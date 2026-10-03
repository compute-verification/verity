import Pous.SecurityProofs.Bandwidth.Digest
import Pous.SecurityProofs.Bandwidth.Placement
import Pous.SecurityProofs.Bandwidth.Honest

/-!
# The bulk read (placement)

* `Bandwidth/Digest.lean`: the digest is almost universal (`PousBandwidth.digestAU`), and the counting helpers.
* `Bandwidth/Placement.lean`: the placement guarantee from `TwoTierBandwidth` (`PousBandwidth.hbmPlacement`), and its
  form at the threshold `f* = BWpcie / BWdev` (`PousBandwidth.hbmPlacementFStar`).
* `Bandwidth/Honest.lean`: the honest responder satisfies `TwoTierBandwidth` (`PousBandwidth.honest_twoTierBandwidth`)
  and is accepted exactly when `b · h ≤ BWpcie · T` (`PousBandwidth.honest_accepts_iff`).
-/
