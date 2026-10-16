import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork

/-! Flock's knowledge soundness at `E^rep`, in parts: one wait truncated to `E^rep`'s `K` draws, by a coupling with `k`
independent accepted runs and a Chernoff bound (`Wait`, `ERep_trunc`); and the soundness proof's escape events, each a
collision among the verifier's own calls in a fork or in one run, covered by `B` and gathered into one square root
(`Fork`, `Escapes.le`). -/
