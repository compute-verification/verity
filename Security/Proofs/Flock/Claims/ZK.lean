import Proofs.Flock.Claims.ZK.Vacuous
import Proofs.Flock.Claims.ZK.Statement
import Proofs.Flock.Claims.ZK.GK
import Proofs.Flock.Claims.ZK.Interact
import Proofs.Flock.Claims.ZK.Coins
import Proofs.Flock.Claims.ZK.Forced
import Proofs.Flock.Claims.ZK.Rewind
import Proofs.Flock.Claims.ZK.GKProof
import Proofs.Flock.Claims.ZK.GKCost

/-!
# Flock's zero knowledge against a malicious verifier

* `Vacuous`: `ZKmal` as frozen holds exactly at `εzk ≥ 1` on the statements with a witness (`ZKmal_iff`), since the
  simulator does not see the verifier's rounds; so does `Flock.ZK` (`flockZK_iff`).
* `Statement`: the claim in a form a simulator can meet, proposed (`ZKmalR`, `Flock.ZKR`): the protocol's rounds, and the
  simulator's own SHA-512 collisions as the binding term; ZK bits under `StrictBirthday` at `Bzk` (`ZKmalR.bits`).
* `GK`: a committed-coin protocol (`Proto`), its honest prover, its Goldreich–Kahan simulator with `N` rewinds
  (`Proto.simulator`), and the three facts the theorem reads (`PrefixHides`, `SHVZK`, `Binding`).
* `Interact`: what the programs evaluate to: `runStar` and the simulator's runs as one interaction (`runStar_msgs`,
  `spRun_eval`), and two provers that agree on a transcript's histories give it to both (`interact_take_congr`).
* `Coins`: what the coin checks evaluate to: a transcript opens `c` exactly when every opening is answered and passes
  the check (`cls_eq_some_iff`), the forced walk agrees (`cls_eq_some_iff_fcls`), and a mismatch is two checked
  openings within the walk's calls (`fcls_eq_mism`).
* `Forced`: the forced run at `c` with the honest prover's messages at `c` opens all of `c` exactly when the real run
  opens `c`, and is then the real run (`fcls_succ_iff`, `forced_eq_of_cls`, `real_eq_of_fcls`); one that stops before a
  binding failure is a real run that opens no coin vector (`cls_none_of_abort`).
* `Rewind`: the rewinding argument over abstract runs: the first run is within `1/(N+1)` plus a binding failure of the
  first successful rewind (`rewind_close`), and `N` independent samples are within `N·δ` (`Pr_hybrid`).
* `GKProof`: **`Proto.gk_zk`**, the Goldreich–Kahan simulator meets `ZKmalR` at `1/(N+1) + 2·δ_A + 2·N·δ_B` from
  `PrefixHides`, `SHVZK` and `Binding`; a binding failure in the simulator is a collision among its own calls
  (`mm_coll`).
* `GKCost`: its cost, `SimCost` at `simBound` under `CostLe` (`Proto.gk_cost`).
-/
