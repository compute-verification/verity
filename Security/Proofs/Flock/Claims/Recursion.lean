import Proofs.Flock.Claims.Recursion.Serve
import Proofs.Flock.Claims.Recursion.Wald
import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Recursion.Transport
import Proofs.Flock.Claims.Recursion.Cost
import Proofs.Flock.Claims.Recursion.CostExp
import Proofs.Flock.Claims.Recursion.Finders
import Proofs.Flock.Claims.Recursion.Hybrid
import Proofs.Flock.Claims.Recursion.KS

/-! The ZK-recursion compiler's claims proofs: serving one rerun program's reruns by another's (`Serve`), `E^rep`'s
weighted reruns by Wald's identity (`Wald`), the composed extractor, `E^rep` served by the outer extractor, with its
cost and its knowledge soundness over `KS`/`KSLin` (`Compose`), where the recursive audit's fork-form disjuncts land
among `KS`'s error terms (`Transport`), the recursion's cost claim from the outer extractor's (`Cost`), the outer
extractor's expected reruns by Wald's argument over forks, which proves the cost claim (`CostExp`), its
extraction-based finder `Be` as a fork finder (`Finders`), the composed extractor's failure with the outer extractor
charged at every rerun (`Hybrid`), and the recursion's `KS` from the outer Flock's KS at each session and the inner
protocol's KS at the served runs (`KS`). -/
