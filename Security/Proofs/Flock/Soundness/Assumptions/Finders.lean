import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ

/-!
# cr/sha-512 at the prover's own finders

`flock_verify_sound`'s claims take A2 (`SHA512CRExpected`) and its strict form (`SHA512CRStrict`) only of the finders
built from the prover they bound, since some finder outputs a collision after no evaluations. These name those
instances, each the definition it stands for at the same arguments: `TableCRAt` and `LinkCRAt` without `--zk`,
`TableCRZAt` under `--zk`.
-/

namespace FlockSoundness.Assumptions

/-- **`SHA512CRStrict` at table `j`'s finders in a batched session** (`Discharge.Exec.TableCRL`): the rewinding finders at
budget `qF` and the self-clash finder at budget `qS`, each built from the prover's session strategy `τ`. -/
def TableCRAt := @Discharge.Exec.TableCRL

/-- **A2 at the prover's link finders** (`Discharge.Exec.LinkCRL`): `SHA512CRExpected` for the finder built for each
commit string a draw can read, at the registration and the prover's continuation. -/
def LinkCRAt := @Discharge.Exec.LinkCRL

/-- **`SHA512CRStrict` at table `j`'s finders in a batched `--zk` session** (`Discharge.ZkLink.TableCRZ`), as
`TableCRAt`. -/
def TableCRZAt := @Discharge.ZkLink.TableCRZ

end FlockSoundness.Assumptions
