import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.KS.Flock
import Proofs.Flock.Claims.KS.Game
import Proofs.Flock.Claims.KS.Deploy

/-! Flock's knowledge soundness at `E^rep`, in parts: one wait truncated to `E^rep`'s `K` draws, by a coupling with `k`
independent accepted runs and a Chernoff bound (`Wait`, `ERep_trunc`); the soundness proof's escape events, each a
collision among the verifier's own calls in a fork or in one run, covered by `B` and gathered into one square root
(`Fork`, `Escapes.le`); `Flock.KS` from one wait's bound on the opened columns (`Flock`, `Flock.KS_of_oneWait`); at a
game's verifier (`Bridge.ofGame`), escapes tied to the verifier's log through the decision's calls (`Game`); and the
deployed plain and masked instances, from one wait's bound (`Deploy`, `FlockDeploy.KS.plain_of_oneWait`,
`FlockDeploy.KS.zk_of_oneWait`). -/
