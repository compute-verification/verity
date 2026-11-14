module

public import Proofs.NetworkCertifier.Witnesses
public import Proofs.NetworkCertifier.WitnessToys

/-!
# Two more instances of the pooled monitoring period's guarantee

`PooledLeakAccepted`'s `honest` and `nontrivial` lemmas are in `Witnesses`: its two inputs (`hesc`, `hleak`) at the toy
code, and its bound failing at an audit that always accepts where the incorrect set and the observations are empty.
`PeriodLeakAccepted`'s pair is there too. This file holds two more instances of `PooledLeakAccepted`, each at what
those two leave out:

* `PooledLeakAccepted_schedules`: its premises on the schedules, at `fxQueue`'s alone: each fits the format (`hfit`) and
  sends at most 6 cells a connection (`hcap`), stated as their conjunction. The honest lemma concludes neither:
  `Witnesses` reads the guarantee at a network certifier with no connection, where both hold with nothing to check.
* `PooledLeakAccepted_fails_at_one_unit`: its bound fails at an audit that always accepts, at `δ = 1/2` with one window
  of one step whose incorrect set holds its one unit (`pooled_bound_fails`). There its conditional escape (`hesc`)
  holds only at `q = 0`, which its `hF` refuses at `δ < 1`.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.Witness.SamplingGeneric (notStmt notPart)

/-- **`PooledLeakAccepted`'s schedule premises at the fixture `fxQueue`**, its schedules that fixture's alone: each
fits its rows format (`hfit`), and sends at most 6 cells on each connection (`hcap`). -/
theorem PooledLeakAccepted_schedules :
    (∀ s ∈ ({fxQueueSched} : Set (CertifierSchedule (Fin 3))), (RowsFormat.recordFormat fxRows fxRows_wf 4 1).fits s) ∧
      ∀ s ∈ ({fxQueueSched} : Set (CertifierSchedule (Fin 3))), ∀ c, s.cells c ≤ 6 := by
  refine ⟨fun s hs => ?_, fun s hs c => ?_⟩
  · rw [Set.mem_singleton_iff.1 hs]
    exact fxQueue_fits
  · rw [Set.mem_singleton_iff.1 hs]
    fin_cases c <;> decide

/-- **`PooledLeakAccepted`'s bound fails at an audit that always accepts**, at one window of one step whose incorrect
set holds its one unit, `Fmax = 1`, `δ = 1/2` and `e = e' = 0` (`pooled_bound_fails`). -/
theorem PooledLeakAccepted_fails_at_one_unit :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) :=
  pooled_bound_fails

end NetTiming.Witness
