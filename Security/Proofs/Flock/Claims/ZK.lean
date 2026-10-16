import Proofs.Flock.Claims.ZK.Vacuous

/-!
# Flock's zero knowledge against a malicious verifier

* `Vacuous`: `ZKmal` as frozen holds exactly at `εzk ≥ 1` on the statements with a witness (`ZKmal_iff`), since the
  simulator does not see the verifier's rounds; so does `Flock.ZK` (`flockZK_iff`).
-/
