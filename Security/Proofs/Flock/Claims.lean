import Models.Flock.Guarantees.Claims
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Decode
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.KS

/-! `Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated`, by reflexivity: the guarantee is there for its
record, which holds the claims' text.

The claims' lemmas sit under `Proofs.Flock.Claims`: uniform-coin probability (`Prob`), `B`'s soundness and the
finders' Cauchy–Schwarz aggregation (`Finder`, `Aggregate`), the `Bits` corollary of `KS` (`Bits`), `E^rep`'s
reruns and amplification (`ERep`), `ERepCost` (`ERepCost`), and a decoder that meets `ListComplete` for interleaved
Reed–Solomon codes, from ArkLib's Guruswami–Sudan decoder (`Decode`), and that decoder at Flock's level-0 code
(`DecodeFlock`). -/

/-- **The claims are stated** (`Flock.Guarantees.ClaimsStated`). -/
theorem Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated := rfl
