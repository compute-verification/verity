import FlockSoundness.Protocol.Discharge.ZkHidden.View
import FlockSoundness.Protocol.Discharge.FailClosed.Scope

/-!
# The hidden-output `--zk` headlines, over what flock-verify accepts

Each theorem here is the `ZkHidden` headline of the same name over `VerifiesZK` (`LiveVerifiesZK`), what
`flock-verify verify --zk` accepts, with no hypothesis on the statement's form, as `FailClosed.Composed` is for public
outputs: the circuit (`circZ I`), the public file and the facts the `ZkHidden` headline took about them come from the
statement check at each statement it passes (`hs : ScopeZ I`), and at a statement the check refuses the event is empty.

What stays a hypothesis on the statement is which headline applies: hidden outputs (`hh`) and, for the stratified law,
one table (`hn`); `zk_session_soundHJ` covers every number of tables. Zero knowledge's `avoidsMask` comes from each
table's setup (`DrawSetupZHJ.avoidsMask`): `verify --zk` refuses a statement whose Δ or regions meet the mask slot
(`Zk.maskFree`, D12). The `ZkHidden` headlines remain, as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind HiddenExec HiddenDec
  Composed Placed StrictCR DecodeZL ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

theorem ScopeZ.tagsOkH {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) (hn : I.count ≤ 1) :
    TagsOkH I :=
  ⟨hh, h.retainRounds, h.merkleLeaf, hn⟩

theorem ScopeZ.tagsOkZHJ {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = true) : TagsOkZHJ I :=
  ⟨hh, h.retainRounds, h.merkleLeaf⟩

variable {mPts n : ℕ} {Reg : Type}

section

variable (P : LigProver)

end

end FlockSoundness.Discharge.FailClosed
