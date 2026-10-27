import Flock.Desc
import Proofs.Flock.Soundness.Discharge.ZkExec.Event

/-!
# The deployed files (`deployed-files`)

`EndToEndHidden` reads the session's circuit and public files only through the values the verifier parses from them:
the circuit's descriptor `D` and the public file `pub`. `EndToEndHiddenModel` is the guarantee over those values, and
`EndToEndHidden` its corollary at the deployed files, which this premise ties to them.
-/

namespace FlockSoundness.Assumptions.Flock

open FlockSoundness.Discharge.Exec (Inputs)

/-- Premise (`deployed-files`) of `EndToEndHidden`. **The deployed files parse to `D` and `pub`.** The circuit
file and the public file `flock-verify verify --zk` reads are the ones whose SHA-512 digests are `pinC` and `pinP`,
and they parse, under the `--zk` tags, to the descriptor `D` (`Desc.parse`) and the public file `pub`
(`HmOut.loadPublic`, at the circuit `D` unrolls to). A run of `flock-verify` on the pinned files checks it. -/
def DeployedFilesHJ (I : Inputs) (D : _root_.Flock.Desc) (pub : _root_.Flock.Public) (pinC pinP : ByteArray) : Prop :=
  _root_.Flock.Desc.parse (Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D ∧
    _root_.Flock.HmOut.loadPublic (Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pub ∧
    _root_.Flock.Sha512.hash I.circuitFile = pinC ∧ _root_.Flock.Sha512.hash I.publicFile = pinP

end FlockSoundness.Assumptions.Flock
