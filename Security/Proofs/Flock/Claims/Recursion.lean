import Proofs.Flock.Claims.Recursion.Serve
import Proofs.Flock.Claims.Recursion.Wald
import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Recursion.Transport
import Proofs.Flock.Claims.Recursion.Cost
import Proofs.Flock.Claims.Recursion.CostExp
import Proofs.Flock.Claims.Recursion.Finders
import Proofs.Flock.Claims.Recursion.Hybrid
import Proofs.Flock.Claims.Recursion.KS
import Proofs.Flock.Claims.Recursion.OuterKS
import Proofs.Flock.Claims.Recursion.Inner
import Proofs.Flock.Claims.Recursion.Abort
import Proofs.Flock.Claims.Recursion.InnerAbort
import Proofs.Flock.Claims.Recursion.PlurLin
import Proofs.Flock.Claims.Recursion.BadLin
import Proofs.Flock.Claims.Recursion.KSLin
import Proofs.Flock.Claims.Recursion.Gap
import Proofs.Flock.Claims.Recursion.Registered
import Proofs.Flock.Claims.Recursion.FlockAbort
import Proofs.Flock.Claims.Recursion.AbortWait
import Proofs.Flock.Claims.Recursion.AbortFork
import Proofs.Flock.Claims.Recursion.FlockKSAbort
import Proofs.Flock.Claims.Recursion.FlockKSAcc
import Proofs.Flock.Claims.Recursion.FlockKSDeploy
import Proofs.Flock.Claims.Recursion.FlockKSRecord
import Proofs.Flock.Claims.Recursion.FlockKSOnAcc
import Proofs.Flock.Claims.Recursion.KSStated
import Proofs.Flock.Claims.Recursion.Scope

/-! The ZK-recursion compiler's claims proofs: serving one rerun program's reruns by another's (`Serve`), `E^rep`'s
weighted reruns by Wald's identity (`Wald`), the composed extractor, `E^rep` served by the outer extractor, with its
cost and its knowledge soundness over `KS`/`KSLin` (`Compose`), where the recursive audit's fork-form disjuncts land
among `KS`'s error terms (`Transport`), the recursion's cost claim from the outer extractor's (`Cost`), the outer
extractor's expected reruns by Wald's argument over forks, which proves the cost claim (`CostExp`), its
extraction-based finder `Be` as a fork finder (`Finders`), the composed extractor's failure with the outer extractor
charged at every rerun (`Hybrid`), and the recursion's `KS` from the outer Flock's KS at each session and the inner
protocol's KS at the served runs (`KS`), and that outer KS at each session from Flock's own KS at the outer deployment,
the session's prover simulating the rest of the compiled verifier around it (`OuterKS`); and `ScopeMatches` at the
compiled verifier's parser and scope check (`Scope`). -/
