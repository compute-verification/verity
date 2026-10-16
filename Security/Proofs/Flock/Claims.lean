import Models.Flock.Guarantees.Claims
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Bridge
import Proofs.Flock.Claims.Decode
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.ZkDecode
import Proofs.Flock.Claims.DecodeFamily
import Proofs.Flock.Claims.ZkDecodeFlock
import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Claims.Scope
import Proofs.Flock.Claims.DeployOC
import Proofs.Flock.Claims.KS

/-! `Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated`, by reflexivity: the guarantee is there for its
record, which holds the claims' text.

The claims' lemmas sit under `Proofs.Flock.Claims`: uniform-coin probability (`Prob`), `B`'s soundness and the
finders' Cauchy–Schwarz aggregation (`Finder`, `Aggregate`), the `Bits` corollary of `KS` (`Bits`), `E^rep`'s
reruns and amplification (`ERep`), `ERepCost` (`ERepCost`), the bridge from `Claims.run` to the live game
(`Bridge`), and a decoder that meets `ListComplete` at the Johnson
count for interleaved Reed–Solomon codes, from ArkLib's Guruswami–Sudan decoder (`Decode`), that decoder at Flock's
level-0 code (`DecodeFlock`), and at the masked (`--zk`) level-0 code, with its read step (`ZkDecode`); a family of
such decoders read as one (`DecodeFamily`), the masked verifier's deployed code at each `m` it serves
(`ZkDecodeFlock`), a masked run's opened level-0 columns (`ZkCols`), the verifier's statement read as `Claims.Flock.Stmt`
(`Scope`), and Flock's `KS` at `E^rep` from one wait's bound (`KS`). -/

/-- **The claims are stated** (`Flock.Guarantees.ClaimsStated`). -/
theorem Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated := rfl
