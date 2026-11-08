module

public import Proofs.Flock.Claims.Recursion.Serve
public import Proofs.Flock.Claims.Recursion.Wald
public import Proofs.Flock.Claims.Recursion.Compose
public import Proofs.Flock.Claims.Recursion.Transport
public import Proofs.Flock.Claims.Recursion.Cost
public import Proofs.Flock.Claims.Recursion.CostExp
public import Proofs.Flock.Claims.Recursion.Finders
public import Proofs.Flock.Claims.Recursion.Hybrid
public import Proofs.Flock.Claims.Recursion.KS
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.Inner
public import Proofs.Flock.Claims.Recursion.Abort
public import Proofs.Flock.Claims.Recursion.InnerAbort
public import Proofs.Flock.Claims.Recursion.PlurLin
public import Proofs.Flock.Claims.Recursion.BadLin
public import Proofs.Flock.Claims.Recursion.KSLin
public import Proofs.Flock.Claims.Recursion.Gap
public import Proofs.Flock.Claims.Recursion.Registered
public import Proofs.Flock.Claims.Recursion.FlockAbort
public import Proofs.Flock.Claims.Recursion.AbortWait
public import Proofs.Flock.Claims.Recursion.AbortFork
public import Proofs.Flock.Claims.Recursion.FlockKSAbort
public import Proofs.Flock.Claims.Recursion.FlockKSAcc
public import Proofs.Flock.Claims.Recursion.FlockKSDeploy
public import Proofs.Flock.Claims.Recursion.FlockKSRecord
public import Proofs.Flock.Claims.KS.PlainAcc
public import Proofs.Flock.Claims.Recursion.KSStated
public import Proofs.Flock.Claims.Recursion.Scope

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

@[expose] public section
