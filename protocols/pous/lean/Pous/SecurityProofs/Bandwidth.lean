import Pous.SecurityProofs.Bandwidth.Digest
import Pous.SecurityProofs.Bandwidth.Placement

/-!
# The bulk read (placement)

* `Bandwidth/Digest.lean`: the digest is almost universal (`PousBandwidth.digestAU`), and the counting helpers.
* `Bandwidth/Placement.lean`: the placement guarantee from `TwoTierBandwidth` (`PousBandwidth.hbmPlacement`), and its
  form at the threshold `f* = BWpcie / BWdev` (`PousBandwidth.hbmPlacementFStar`).
-/
