import Proofs.Flock.Claims.Recursion.Serve
import Proofs.Flock.Claims.Recursion.Wald
import Proofs.Flock.Claims.Recursion.Compose
import Proofs.Flock.Claims.Recursion.Transport

/-! The ZK-recursion compiler's claims proofs: serving one rerun program's reruns by another's (`Serve`), `E^rep`'s
weighted reruns by Wald's identity (`Wald`), the composed extractor, `E^rep` served by the outer extractor, with its
cost and its knowledge soundness over `KS`/`KSLin` (`Compose`), and where the recursive audit's fork-form disjuncts land
among `KS`'s error terms (`Transport`). -/
