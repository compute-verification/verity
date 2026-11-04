import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys

/-!
# The monitoring period's honest and nontrivial lemmas

`PooledLeakAccepted` has an `honest` and a `nontrivial` lemma and `PeriodLeakAccepted` a `nontrivial` one, each listed
in the lock's `witnesses`, and none takes a binder over its instance; the lock's `unwitnessed` exempts
`PeriodLeakAccepted`'s `honest` half.

* `PeriodLeakAccepted_nontrivial`: the per-window bound fails at the accepting verifier `tapD` at SHA-512, at
  `δ = 1/8` (`tapD_bound_fails`).
* `PooledLeakAccepted_honest`: its premises on the schedules, at `fxQueue`'s alone: each fits the format (`hfit`) and
  sends at most 6 cells a connection (`hcap`). Stated as their conjunction, not as `fits` alone: the witnesses rule
  reads a statement that is one definition applied by unfolding it once, and `fits` is a field.
* `PooledLeakAccepted_nontrivial`: its bound fails at an audit that always accepts, at `δ = 1/2` with one window of
  one step whose faulty set holds its one unit (`pooled_bound_fails`). There its conditional escape (`hesc`) holds
  only at `q = 0`, which its `hF` refuses at `δ < 1`.
-/

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

local notation "shaD" => FlockDeploy.sha512

/-- **`PeriodLeakAccepted`'s per-window bound fails at the accepting verifier** `tapD`, at SHA-512, at one step at
`δ = 1/8` with no input registered (`tapD_bound_fails`). -/
theorem PeriodLeakAccepted_nontrivial :
    ¬ PrBad shaD (tapD Unit).V
        (RLeakTap shaD (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay)
          (fun _ => 1) (1 / 8) (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => tapOps))
        (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (tapD Unit) (1 / 8) [] +
          Real.sqrt (Claims.Sampling.γ (tapD Unit) [] * PrB shaD (tapD Unit).V silentP []) +
        (Bext shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) Unit silentP []).Pr shaD +
        (Btap shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
          (fun _ => tapOps) Unit silentP []).Pr shaD :=
  tapD_bound_fails shaD

/-- **`PooledLeakAccepted`'s schedule premises at the deployed instance**, its schedules the fixture `fxQueue`'s alone:
each fits its rows format (`hfit`), and sends at most 6 cells on each connection (`hcap`). -/
theorem PooledLeakAccepted_honest :
    (∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits s) ∧
      ∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), ∀ c, s.cells c ≤ 6 := by
  refine ⟨fun s hs => ?_, fun s hs c => ?_⟩
  · rw [Set.mem_singleton_iff.1 hs]
    exact fxQueue_fits
  · rw [Set.mem_singleton_iff.1 hs]
    fin_cases c <;> decide

/-- **`PooledLeakAccepted`'s bound fails at an audit that always accepts**, at one window of one step whose faulty
set holds its one unit, `Fmax = 1`, `δ = 1/2` and `e = e' = 0` (`pooled_bound_fails`). -/
theorem PooledLeakAccepted_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) :=
  pooled_bound_fails

end NetTiming.Witness
