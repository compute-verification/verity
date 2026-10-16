import Proofs.Flock.Claims.ZK.Vacuous
import Proofs.Flock.Claims.ZK.Statement
import Proofs.Flock.Claims.ZK.GK

/-!
# Flock's zero knowledge against a malicious verifier

* `Vacuous`: `ZKmal` as frozen holds exactly at `εzk ≥ 1` on the statements with a witness (`ZKmal_iff`), since the
  simulator does not see the verifier's rounds; so does `Flock.ZK` (`flockZK_iff`).
* `Statement`: the claim in a form a simulator can meet, proposed (`ZKmalR`, `Flock.ZKR`): the protocol's rounds, and the
  simulator's own SHA-512 collisions as the binding term; ZK bits under `StrictBirthday` at `Bzk` (`ZKmalR.bits`).
* `GK`: a committed-coin protocol (`Proto`), its honest prover, its Goldreich–Kahan simulator with `N` rewinds
  (`Proto.simulator`), and the three facts the theorem reads (`PrefixHides`, `SHVZK`, `Binding`).
-/
