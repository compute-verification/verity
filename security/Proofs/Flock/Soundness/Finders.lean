module

public import Proofs.Flock.Soundness.Discharge.Exec.StrictCRL
public import Proofs.Flock.Soundness.Discharge.Exec.FlockLinkL
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
import Mathlib.Combinatorics.Matroid.Init

/-!
# cr/sha-512 at the prover's own finders

`flock_verify_sound`'s claims take A2 (`SHA512CRExpected`) and its strict form (`SHA512CRStrict`) only of the finders
built from the prover they bound, since some finder outputs a collision after no evaluations. These name those
instances, each the definition it stands for at the same arguments: `TableCRAt` and `LinkCRAt` without `--zk`,
`TableCRZAt` and `LinkCRZCAt` under `--zk`.

Each is reducible, so a statement that takes one takes the definition it stands for, a conjunction of
`StrictCR.Finder.CR`s or a `∀` of `SHA512CRExpected`s at finders built from the prover. Those are the named assumptions
of `Definitions.Assumptions.Flock`, so a guarantee taking these takes SHA-512's collision resistance and nothing else.
-/

@[expose] public section

namespace FlockSoundness.Assumptions

/-- **`SHA512CRStrict` at table `j`'s finders in a batched session** (`Discharge.Exec.TableCRL`): the rewinding finders at
budget `qF` and the self-clash finder at budget `qS`, each built from the prover's session strategy `τ`. -/
abbrev TableCRAt := @Discharge.Exec.TableCRL

/-- **A2 at the prover's link finders** (`Discharge.Exec.LinkCRL`): `SHA512CRExpected` for the finder built for each
commit string a sample can read, at the registration and the prover's continuation. -/
abbrev LinkCRAt := @Discharge.Exec.LinkCRL

/-- **`SHA512CRStrict` at table `j`'s finders in a batched `--zk` session** (`Discharge.ZkLink.TableCRZ`), as
`TableCRAt`. -/
abbrev TableCRZAt := @Discharge.ZkLink.TableCRZ

/-- **A2 at the prover's link finders in a compiled `--zk` session** (`Discharge.ZkLink.LinkCRZC`), as `LinkCRAt`. -/
abbrev LinkCRZCAt := @Discharge.ZkLink.LinkCRZC

end FlockSoundness.Assumptions
