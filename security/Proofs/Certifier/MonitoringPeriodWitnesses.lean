module

public import Proofs.Certifier.Witnesses
public import Proofs.Certifier.WitnessToys

/-! Two more instances of the pooled monitoring period's guarantee, each at what its honest and nontrivial lemmas leave out. -/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.Witness.SamplingGeneric (notStmt notPart)

/-- The honest lemma concludes neither: `Witnesses` reads the guarantee at a network attester with no connection, where
both hold with nothing to check. -/
theorem PooledLeakAccepted_schedules :
    (∀ s ∈ ({fxQueueSched} : Set (CertifierSchedule (Fin 3))), (RowsFormat.recordFormat fxRows fxRows_wf 4 1).fits s) ∧
      ∀ s ∈ ({fxQueueSched} : Set (CertifierSchedule (Fin 3))), ∀ c, s.cells c ≤ 6 := by
  refine ⟨fun s hs => ?_, fun s hs c => ?_⟩
  · rw [Set.mem_singleton_iff.1 hs]
    exact fxQueue_fits
  · rw [Set.mem_singleton_iff.1 hs]
    fin_cases c <;> decide

/-- There its conditional escape holds only at `q = 0`, which its `hF` refuses at `δ < 1`. -/
theorem PooledLeakAccepted_fails_at_one_unit :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) :=
  pooled_bound_fails

end NetTiming.Witness
