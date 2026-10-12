import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask
import Proofs.Flock.Soundness.Discharge.FailClosed.Composed
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec
import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegisteredTags
import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies

/-!
# The headlines over what the verifier accepts

`flock-verify verify` refuses every statement whose form the soundness proof does not cover (`Flock.ProvedScope.check`),
so its acceptance (`Verifies`, `VerifiesZK`, and under `--zk --registered F` `VerifiesZKR`) carries the facts about the
statement that the headlines took as hypotheses (`FailClosed/Scope.lean`). The headlines restated over it, with no
hypothesis on the statement's form: `FailClosed/Composed.lean` (public outputs, `--zk`), `FailClosed/ZkHidden.lean`
(hidden outputs, `--zk`), `FailClosed/ZkRegistered.lean` (hidden outputs, `--zk --registered F`, the one form with a
shared-row public file the check passes) and `FailClosed/Exec.lean` (one table, no `--zk`, public or hidden outputs).
Zero knowledge's views take `avoidsMask` from the setup (`FailClosed/Mask.lean`): `verify --zk` refuses a statement whose
Δ or regions meet the mask slot. `FailClosed/One.lean` is the one theorem over all of it (`flock_verify_sound`): its
conclusion splits on the statement and states each case's strongest claim. `FailClosed/Numeric.lean` is its numeric
corollary without `--zk` (`flock_verify_sound_205`): each table's term is `2^-205`, from the range acceptance carries.
`FailClosed/PublicTies.lean` (`flock_verify_public_ties`) states, deterministically, what the verifier's check of the
public file (`checkPublic`) ties its header's frame-v3-sha512 roots, domains and schemas to.
-/
