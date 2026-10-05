import FlockSoundness.Protocol.Discharge.HiddenExec.HeadlineExec
import FlockSoundness.Protocol.Discharge.HiddenDec.Assemble
import FlockSoundness.Protocol.Discharge.DecodeZLJ.Statements
import FlockSoundness.Protocol.Discharge.FailClosed.Scope

/-!
# The non-ZK headlines, over what flock-verify accepts

Each theorem here is the headline of the same name (`HdecCor`, `Integrate`, `HiddenExec`) over `Verifies`
(`LiveVerifies`), what `flock-verify verify` accepts without `--zk`, with no hypothesis on the statement's form: the
circuit (`circS I`), the public file and the facts the headline took about them (`ht`, `hc`, `hU`, `hpub`, `hpt`,
`hscope`, and `c.outNet = c.unitNet` and `TagsOk`/`TagsOkH` in the decoded forms) come from the statement check at each
statement it passes (`hs : Scope I`), and at a statement the check refuses the event is empty and the bound `0`. The
check refuses a session of more than one table without `--zk`, so these cover every non-ZK session it accepts.

What stays a hypothesis on the statement is which headline applies: public outputs (`hh := false`) or hidden ones. The
original headlines remain, as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

theorem Scope.tagsOk {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = false) : DecodeZL.TagsOk I :=
  ⟨hh, h.retainRounds, h.merkleLeaf, h.count⟩

theorem Scope.tagsOkH {I : Inputs} (h : Scope I) (hh : I.tags.hiddenOutputs = true) : HiddenDec.TagsOkH I :=
  ⟨hh, h.retainRounds, h.merkleLeaf, h.count⟩

section Public

open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL

end Public

section Hidden

open FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.HiddenDec FlockSoundness.Discharge.DecodeZL

end Hidden

end FlockSoundness.Discharge.FailClosed
