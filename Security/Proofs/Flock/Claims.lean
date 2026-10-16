import Models.Flock.Guarantees.Claims
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.ERep
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Sampling.Draws
import Proofs.Flock.Claims.Sampling.ExecBernoulli
import Proofs.Flock.Claims.Sampling.Live
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Escape

/-! `Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated`, by reflexivity: the guarantee is there for its
record, which holds the claims' text.

The claims' lemmas sit under `Proofs.Flock.Claims`: uniform-coin probability (`Prob`), `B`'s soundness and the
finders' Cauchy–Schwarz aggregation (`Finder`, `Aggregate`), the `Bits` corollary of `KS` (`Bits`), `E^rep`'s
reruns and amplification (`ERep`), and `ERepCost` (`ERepCost`). The sampling compiler's lemmas are under
`Proofs.Flock.Claims.Sampling`: each registered draw law's miss bound in `Claims.Pr` (`Sampling.Draws`), the
executable Bernoulli draw's bound (`Sampling.ExecBernoulli`), every law's live draw's (`Sampling.Live`), the sampled
extractor and its reruns (`Sampling.Extract`), the boundary values' binding, `KSLin`'s linear term
(`Sampling.Binding`), and the claims block's escape and miss as the draws' probabilities (`Sampling.Escape`). -/

/-- **The claims are stated** (`Flock.Guarantees.ClaimsStated`). -/
theorem Flock.SecurityProofs.ClaimsStated : Flock.Guarantees.ClaimsStated := rfl
