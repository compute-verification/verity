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
are a SHA-512 collision. The collision disjunct is there because v2 finds the public file among the registered files by
its digest, so `live` is set up on those files only where the two files' SHA-512s differ, and the kernel does not
evaluate SHA-512; the pair is the two explicit files, `circuitFile` and `publicFile`. This shows the instance and the
schedule's shape, not an accepted play. -/
theorem remote_soundness_deployedReady_honestSession :
    inst.SessionsReady ∧ inst.SessionRegistered .remote ∧ inst.base.PouwSound .remote ∧
      inst.base.InferenceOnlySound .remote ∧
      ((inst.session θ₀ = some live ∧ live.I.tags.hiddenOutputs = true ∧
          FlockSoundness.Discharge.Composed.nTab live.I = 1 ∧
          2 ≤ (FlockSoundness.Refine.schedOf live.y₀.sch).levels.length) ∨
        Hash.Collision Verity.sha512 circuitFile.data.toList publicFile.data.toList) := by
  refine ⟨sessions_ready, session_registered, target_top .remote .pouw, target_top .remote .inferenceOnly, ?_⟩
  by_cases hd : Distinct
  · refine Or.inl ⟨?_, ?_, ?_, live.y₀.hshape.1⟩
    · show session θ₀ = some live
      unfold session
      rw [ite_eq_left ⟨rfl, hd⟩]
    · rw [live_inputs]; rfl
    · rw [live_inputs]; rfl
  · exact Or.inr (collision hd)

end Verity.Sampled
