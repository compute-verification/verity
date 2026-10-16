import Models.Flock.Guarantees.Claims
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Bridge
import Proofs.Flock.Claims.Numbers
import Proofs.Flock.Claims.NumbersTable
import Proofs.Flock.Claims.NumbersPads
import Proofs.Flock.Claims.NumbersGamma
import Proofs.Flock.Claims.NumbersNMin
import Proofs.Flock.Claims.Decode
import Proofs.Flock.Claims.DecodeFlock
import Proofs.Flock.Claims.ZkDecode
import Proofs.Flock.Claims.DecodeFamily
import Proofs.Flock.Claims.ZkDecodeFlock
import Proofs.Flock.Claims.ZkCols
import Proofs.Flock.Claims.Scope
import Proofs.Flock.Claims.DeployOC
import Proofs.Flock.Claims.DeployInputs
import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Claims.DeployCode
import Proofs.Flock.Claims.RootFindGF128
import Proofs.Flock.Claims.GS

/-! `Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated`, by reflexivity: the guarantee is there for its
record, which holds the claims' text.

The claims' lemmas sit under `Proofs.Flock.Claims`: uniform-coin probability (`Prob`), `B`'s soundness and the
finders' Cauchy–Schwarz aggregation (`Finder`, `Aggregate`), the `Bits` corollary of `KS` (`Bits`), `E^rep`'s
reruns and amplification (`ERep`), `ERepCost` (`ERepCost`), the bridge from `Claims.run` to the live game
(`Bridge`), and a decoder that meets `ListComplete` at the Johnson
count for interleaved Reed–Solomon codes, from ArkLib's Guruswami–Sudan decoder (`Decode`), that decoder at Flock's
level-0 code (`DecodeFlock`), and at the masked (`--zk`) level-0 code, with its read step (`ZkDecode`); a family of
such decoders read as one (`DecodeFamily`), the masked verifier's deployed code at each `m` it serves
(`ZkDecodeFlock`), a masked run's opened level-0 columns (`ZkCols`), and the verifier's statement read as `Claims.Flock.Stmt`
(`Scope`). Flock's numbers at the deployed parameters: `k`, `κ` and `λ` (`Numbers`), `ε_table` (`NumbersTable`), the
pads' domain (`NumbersPads`), the weight and `γ` (`NumbersGamma`), and `nMin` (`NumbersNMin`). The deployed instance of
`Claims.Flock.Deploy`: the hash-parametric verdict over the oracle (`DeployOC`), the statement's bytes, parser and setup
(`DeployInputs`), the verifier in public-coin form (`Deploy`), and the code, columns and numbers that complete the
plain and masked instances (`DeployCode`: `FlockDeploy.deployPlain`, `FlockDeploy.deployZk`). A counted
Guruswami–Sudan decoder with its `DecoderComplexity`, over Berlekamp's root finder in GF(2^128) (`RootFindGF128`), is
`GS`; `GS.Radius` states the list-decoding spec in its radius form (`Flock.ListWithin`) and proves that decoder meets it
at Flock's level-0 code. -/

/-- **The claims are stated** (`Flock.Guarantees.ClaimsStated`). -/
theorem Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated := rfl
