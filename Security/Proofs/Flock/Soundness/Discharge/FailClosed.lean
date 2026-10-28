import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.FailClosed.Mask
import Proofs.Flock.Soundness.Discharge.FailClosed.Composed
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHidden
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkRegistered
import Proofs.Flock.Soundness.Discharge.FailClosed.Exec
import Proofs.Flock.Soundness.Discharge.FailClosed.General
import Proofs.Flock.Soundness.Discharge.FailClosed.PlainRegistered
import Proofs.Flock.Soundness.Discharge.FailClosed.One
import Proofs.Flock.Soundness.Discharge.FailClosed.Numeric
import Proofs.Flock.Soundness.Discharge.FailClosed.ZkHiddenM1
import Proofs.Flock.Soundness.Discharge.FailClosed.PublicTies

/-!
# The headlines over what the verifier accepts

`flock-verify verify` refuses every statement whose form the soundness proof does not cover (`Flock.ProvedScope.check`),
so its acceptance (`Verifies`, `VerifiesZK`, and with `--registered F` `VerifiesR` and `VerifiesZKR`) carries the facts about the
statement that the headlines took as hypotheses (`FailClosed/Scope.lean`). The headlines restated over it, with no
hypothesis on the statement's form: `FailClosed/Composed.lean` (public outputs, `--zk`), `FailClosed/ZkHidden.lean`
(hidden outputs, `--zk`), `FailClosed/ZkRegistered.lean` (hidden outputs, `--zk --registered F`),
`FailClosed/Exec.lean` (one table, no `--zk`, public or hidden outputs) and `FailClosed/PlainRegistered.lean` (one
table, no `--zk`, hidden outputs, `--registered F`). The registered forms are the ones with a shared-row public file the
check passes, at `hm96-sha512/v1` leaves.
Under hidden outputs these are the headlines where the circuit's wires feed neither its unit nor its outputs; where they
feed either, the statement is in the general form (`Layout.scopeGenH`) and the headlines are `FailClosed/General.lean`'s,
at the template instance's program.
Zero knowledge's views take `avoidsMask` from the setup (`FailClosed/Mask.lean`): `verify --zk` refuses a statement whose
Δ or regions meet the mask slot. `FailClosed/One.lean` is the one theorem over all of it (`flock_verify_sound`): its
conclusion splits on the statement and states each case's strongest claim. `FailClosed/Numeric.lean` is its numeric
corollary without `--zk` (`flock_verify_sound_205`): each table's term is `2^-205`, from the range acceptance carries.
`FailClosed/PublicTies.lean` (`flock_verify_public_ties`) states, deterministically, what the verifier's check of the
public file (`checkPublic`) ties its header's frame-v3-sha512 roots, domains and schemas to.
-/
