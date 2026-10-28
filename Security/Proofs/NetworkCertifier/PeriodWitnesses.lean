import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.Assumptions.SampledCode

/-!
# The monitoring period's honest and nontrivial lemmas

`PeriodLeakAccepted` and `HorizonLeakAccepted` each have an `honest` and a `nontrivial` lemma, listed in the lock's
`witnesses`, and neither takes a binder over its instance:

* `PeriodLeakAccepted_honest`: under `SampledCodeRuns`, the one named assumption, the premises that are no named
  assumption hold at the deployed instance: `Claims.Sampling.KS` at SHA-512 at the deployed sampling record
  `SamplingDeploy.deploy`, whose commitment and sessions' Flock record are `deploy`'s (`ks_deployed`); its decoder
  within Flock's radius (`Sampling.decoder_meets_zk`); its hm96 commitment binding (`Binding_of_hm96`); and the rows
  format of the Python network certifier's fixture `fxQueue` fitting its schedule (`fxQueue_fits`). A statement is in
  the record's scope. The other premises are bounds and equalities over the instance's parameters, and `TrustedHost`
  (`trusted-host`), a named assumption.
* `PeriodLeakAccepted_nontrivial`: the per-window bound fails at the accepting verifier `tapD` at SHA-512, at
  `δ = 1/8` (`tapD_bound_fails`).
* `HorizonLeakAccepted_honest`: its premises on the schedules, at `fxQueue`'s alone: each fits the format (`hfit`) and
  sends at most 6 cells a connection (`hcap`). Stated as their conjunction, not as `fits` alone: the witnesses rule
  reads a statement that is one definition applied by unfolding it once, and `fits` is a field.
* `HorizonLeakAccepted_nontrivial`: its bound fails at an audit that always accepts, at `δ = 1/2` with one window of
  one step whose faulty set holds its one unit (`horizon_bound_fails`). There its conditional escape (`hesc`) holds
  only at `q = 0`, which its `hF` refuses at `δ < 1`.
-/

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

local notation "relD" => _root_.Flock.Readable.relation
local notation "shaD" => FlockDeploy.sha512

set_option synthInstance.maxSize 1024 in
/-- **`PeriodLeakAccepted`'s premises at the deployed instance**, under `SampledCodeRuns`: `KS` at SHA-512, the decoder
within Flock's radius and the commitment binding at the deployed sampling record, a statement in its scope; and the
fixture `fxQueue`'s schedule fitting its rows format. -/
theorem PeriodLeakAccepted_honest (h : NetTiming.Assumptions.SampledCodeRuns) :
    (∃ c : SamplingDeploy.Code,
      (∃ x, Claims.Sampling.Scope (SamplingDeploy.deploy relD c) x) ∧
        Claims.Sampling.KS shaD (SamplingDeploy.deploy relD c) ∧
        Claims.Flock.ListWithin (SamplingDeploy.deploy relD c).flock.len (SamplingDeploy.deploy relD c).flock.enc
          (SamplingDeploy.deploy relD c).flock.θ (SamplingDeploy.deploy relD c).flock.decode ∧
        (SamplingDeploy.deploy relD c).commit.Binds shaD) ∧
      (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits fxQueueSched := by
  obtain ⟨c, hscope, hfixed, hsessions, hin, hslots, ⟨iroots, hopens, hfix⟩, hprog, hchecks⟩ := h
  refine ⟨⟨c, hscope, ?_, Claims.Sampling.decoder_meets_zk relD _ (SamplingDeploy.deploy_flock relD c), ?_⟩,
    fxQueue_fits⟩
  · exact ks_deployed _ (SamplingDeploy.deploy_flock relD c) (SamplingDeploy.deploy_εtable_nonneg relD c) c.own c.at_
      (SamplingDeploy.row c.place c.bits) c.salt c.π c.layer (SamplingDeploy.deploy_commit relD c)
      (SamplingDeploy.row_places c.place c.bits) hfixed hsessions (SamplingDeploy.innerNumbers_deploy relD c hin)
      hslots iroots hopens hfix hprog hchecks
  · exact Binding_of_hm96 _ c.own c.at_ (SamplingDeploy.row c.place c.bits) c.salt c.π c.layer
      (SamplingDeploy.deploy_commit relD c) (SamplingDeploy.row_places c.place c.bits) shaD

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

/-- **`HorizonLeakAccepted`'s schedule premises at the deployed instance**, its schedules the fixture `fxQueue`'s alone:
each fits its rows format (`hfit`), and sends at most 6 cells on each connection (`hcap`). -/
theorem HorizonLeakAccepted_honest :
    (∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits s) ∧
      ∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), ∀ c, s.cells c ≤ 6 := by
  refine ⟨fun s hs => ?_, fun s hs c => ?_⟩
  · rw [Set.mem_singleton_iff.1 hs]
    exact fxQueue_fits
  · rw [Set.mem_singleton_iff.1 hs]
    fin_cases c <;> decide

/-- **`HorizonLeakAccepted`'s bound fails at an audit that always accepts**, at one window of one step whose faulty
set holds its one unit, `Fmax = 1`, `δ = 1/2` and `e = e' = 0` (`horizon_bound_fails`). -/
theorem HorizonLeakAccepted_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ HorizonGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) :=
  horizon_bound_fails

end NetTiming.Witness
