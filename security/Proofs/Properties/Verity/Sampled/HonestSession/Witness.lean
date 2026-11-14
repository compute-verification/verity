module

public import Proofs.Properties.Verity.Sampled.HonestSession.Instance

/-!
# `remote_soundness_deployedReady`'s premises at an instance that runs a session

`Instance.HonestSession.inst` registers a hidden-output circuit and its public file at `θ₀` and runs `live` there, the session v2
sets up on them: zero-knowledge (`verify --zk`'s tags, hidden outputs), one table, its schedule at least two levels
deep. The witness shows the instance and the schedule's shape, not an accepted play.
-/

@[expose] public section

namespace Verity.Sampled

open scoped ENNReal

open Instance.HonestSession in
/-- **`remote_soundness_deployedReady`'s premises at an instance that runs a session**: they hold at
`HonestSession.inst`, whose base publishes `⊤` for every subprotocol. It runs `live` at `θ₀`, a zero-knowledge session
of one table with hidden outputs whose schedule has at least two levels, unless the registered circuit and public files
are a SHA-512 collision. The hypotheses are named propositions, none proved here: `CircuitFileFacts`, `PublicFileFacts`,
`SetupFacts` and `ProgramFacts` state what the concrete circuit file, public file, setup and program do, and
`JsonCanon.JsonFacts` the canonical-JSON reader's three facts. The collision disjunct is there because v2 finds the public file among the registered files by
its digest, so `live` is set up on those files only where the two files' SHA-512s differ, and the kernel does not
evaluate SHA-512; the pair is the two explicit files, `circuitFile` and `publicFile`. This shows the instance and the
schedule's shape, not an accepted play. -/
theorem remote_soundness_deployedReady_honestSession
    (hC : CircuitFileFacts) (hP : PublicFileFacts) (hS : SetupFacts) (hPr : ProgramFacts)
    (hJ : JsonCanon.JsonFacts) :
    (inst hC hP hS hPr hJ).SessionsReady ∧ (inst hC hP hS hPr hJ).SessionRegistered .remote ∧
      (inst hC hP hS hPr hJ).base.PouwSound .remote ∧ (inst hC hP hS hPr hJ).base.InferenceOnlySound .remote ∧
      (((inst hC hP hS hPr hJ).session θ₀ = some (live hC hP hS hPr) ∧
          (live hC hP hS hPr).I.tags.hiddenOutputs = true ∧
          FlockSoundness.Discharge.Composed.nTab (live hC hP hS hPr).I = 1 ∧
          2 ≤ (FlockSoundness.Refine.schedOf (live hC hP hS hPr).y₀.sch).levels.length) ∨
        Hash.Collision Verity.sha512 circuitFile.data.toList publicFile.data.toList) := by
  refine ⟨sessions_ready hC hP hS hPr hJ, session_registered hC hP hS hPr hJ, target_top .remote .pouw,
    target_top .remote .inferenceOnly, ?_⟩
  by_cases hd : Distinct
  · refine Or.inl ⟨?_, ?_, ?_, (live hC hP hS hPr).y₀.hshape.1⟩
    · show session hC hP hS hPr θ₀ = some (live hC hP hS hPr)
      unfold session
      rw [ite_eq_left ⟨rfl, hd⟩]
    · rw [live_inputs hC hP hS hPr]; rfl
    · rw [live_inputs hC hP hS hPr]; rfl
  · exact Or.inr (collision hd)

end Verity.Sampled
