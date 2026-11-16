module

public import Proofs.Properties.Verity.Sampled.HonestSession.Instance
public import Proofs.Properties.Verity.Sampled.HonestSession.CircuitFileProof
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanonFacts
public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramPop
public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramSetup
public import Proofs.Properties.Verity.Sampled.HonestSession.PublicFileReads

/-!
# `remote_soundness_deployedReady`'s premises at an instance that runs a session

`Instance.HonestSession.inst` registers a hidden-output circuit and its public file at `θ₀` and runs a session there, the
one `liveOf` chooses among those v2 sets up on them at `live`'s bound: zero-knowledge (`verify --zk`'s tags, hidden
outputs), one table, its schedule at least two levels deep. The witness shows the instance and the session's shape, not
an accepted play.
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

open Flock Instance.HonestSession in
/-- **`remote_soundness_deployedReady`'s premises at an instance that runs a session**: the two it takes beside the
statements on the committed files (`Vocab.WiresSound`, `GateARow`) hold at `HonestSession.inst`, whose base publishes
`⊤` for every subprotocol: the coin commitment's families hold the family v2's parameters name (`families_pinned`), and
PoUW's target holds at `⊤` (`target_top`). The instance runs a session at `θ₀` on the registered circuit and public
files, a zero-knowledge session of one table with hidden outputs whose schedule has at least two levels, unless those
two files are a SHA-512 collision. The circuit file's facts (`CircuitFileProof.circuitFileFacts`), the canonical-JSON
reader's (`JsonCanon.jsonFacts`, in `Params`), the public file's (`publicFileFacts`) and the program's
(`ProgramPop.programFacts`) are proved, and the setup's are `ProgramSetup.setupFacts_prog` over the public file's four
reads (`PublicFileReads`). The collision disjunct is there because v2 finds the public file among the registered files by
its digest, so a session is set up on those files only where the two files' SHA-512s differ, and the kernel does not
evaluate SHA-512; the pair is the two explicit files, `circuitFile` and `publicFile`. This shows the instance and the
session's shape, not an accepted play. -/
theorem remote_soundness_deployedReady_honestSession :
    let hC := CircuitFileProof.circuitFileFacts
    let hP := publicFileFacts hC
    let hS := ProgramSetup.setupFacts_prog hP (pubD_n hC) (pubD_units_indices hC) (pubD_digest_size hC)
      (pubI_units_indices hC)
    let hPr := ProgramPop.programFacts
    (inst hC hP hS hPr).FamiliesPinned .remote ∧
      (inst hC hP hS hPr).base.PouwSound .remote ∧
      ((∃ A, (inst hC hP hS hPr).session θ₀ = some A ∧ A.I.circuitFile = circuitFile ∧
          A.I.publicFile = publicFile ∧ A.I.tags.hiddenOutputs = true ∧
          FlockSoundness.Discharge.Composed.nTab A.I = 1 ∧
          2 ≤ (FlockSoundness.Refine.schedOf A.y₀.sch).levels.length) ∨
        Hash.Collision Verity.sha512 circuitFile.data.toList publicFile.data.toList) := by
  intro hC hP hS hPr
  refine ⟨families_pinned hC hP hS hPr .remote, target_top .remote .pouw, ?_⟩
  by_cases hd : Distinct
  · obtain ⟨A, hA, hs⟩ := session_at hC hP hS hPr hd
    have ht : A.I.tags = inputs.tags := Option.some.inj (hs.tags.symm.trans tags_find)
    refine Or.inl ⟨A, hA, ?_, ?_, ?_, ?_, A.y₀.hshape.1⟩
    · rw [hs.circuit, circuit_at]
    · rw [hs.publicFile, public_at hd]
    · rw [ht]; rfl
    · unfold FlockSoundness.Discharge.Composed.nTab
      rw [hs.count]; rfl
  · exact Or.inr (collision hd)

end Verity.Sampled
