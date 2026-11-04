import Flock.Desc
import Proofs.Flock.Soundness.Discharge.ZkExec.Event

/-!
# The session's statement file parses to its descriptor

`SampledKSLiveV5` is stated at the descriptor `D` the verifier reads from the session's circuit file. That the file reads
as `D` is one named premise, stated with the verifier's own parse at the session's inputs.
-/

namespace Flock.Bridge.Assumptions

open Flock

/-- **The session's statement file parses to `D`.** Under the tags `verify --zk` sets the session up with (`zkTags I`)
and at the session's lookup tables, the circuit file `I.circuitFile` reads as the descriptor `D` (`Desc.parseStmt`: a
templated statement as `HmOut.parseTmpl` reads it, any other as `Desc.parse` does).

Assumption record. No claim: a hypothesis on the session's inputs that names the descriptor the verifier's own parse
reads from them, and says nothing of the prover, a hash or the world. `SampledKSLiveV5Honest` concludes it at the served
registered session, from `ServedFilesParseV5`. -/
def StatementParses (I : FlockSoundness.Discharge.Exec.Inputs) (D : Desc) : Prop :=
  Desc.parseStmt (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D

end Flock.Bridge.Assumptions
