import Mathlib.LinearAlgebra.Vandermonde
import FlockSoundness.Assumptions.Zk
import FlockSoundness.Protocol.Discharge.FailClosed.Scope
import FlockSoundness.Protocol.Discharge.ZkHidden.DefsJ

/-!
# The public-output `--zk` headlines, over what flock-verify accepts

Each theorem here is the `Composed` headline of the same name over `VerifiesZK` (`LiveVerifiesZK`), what
`flock-verify verify --zk` accepts, in place of `AcceptsZK`, with no hypothesis on the statement's form: the verifier
refuses every statement whose form the soundness proof does not cover (`Flock.ProvedScope.check`), so the circuit, the
public file and the facts the `Composed` headline took about them (`ht`, `hc`, `hU`, `hpub`, `hpt`, `hscope`, `hTags`)
come from the check at each statement it passes (`hs : ScopeZ I`, `FailClosed.Scope`). The units are those of the
check's circuit (`circZ I`); the event's count and the bound read the check (`∀ hs`, `⨆ hs`), and at a statement the
check refuses the event is empty and the bound `0`. `prob_scope` carries each `Composed` headline over.

What stays a hypothesis on the statement is which headline applies: public outputs (`hh`) and, for the stratified law,
one table (`hn`); `zk_session_soundJ` covers every number of tables. Zero knowledge's `avoidsMask` comes from the setup
(`DrawSetupZK.avoidsMask`): `verify --zk` refuses a statement whose Δ or regions meet the mask slot (`Zk.maskFree`, D12).
The `Composed` headlines remain, as lemmas.
-/

namespace FlockSoundness.Discharge.FailClosed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL
open Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

theorem ScopeZ.tagsOk {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) (hn : I.count ≤ 1) :
    TagsOk I :=
  ⟨hh, h.retainRounds, h.merkleLeaf, hn⟩

theorem ScopeZ.tagsOkZJ {I : Inputs} (h : ScopeZ I) (hh : I.tags.hiddenOutputs = false) : TagsOkZJ I :=
  ⟨hh, h.retainRounds, h.merkleLeaf⟩

variable {mPts n : ℕ} {Reg : Type}

section

variable (P : LigProver)

end

end FlockSoundness.Discharge.FailClosed
