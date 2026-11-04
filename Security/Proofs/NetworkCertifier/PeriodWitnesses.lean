import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.Assumptions.SampledCode
import Definitions.NetworkCertifier.Guarantees.PeriodOnce

/-!
# The monitoring period's honest and nontrivial lemmas

`PeriodLeakAccepted` and `PooledLeakAccepted` each have an `honest` and a `nontrivial` lemma, and `PeriodLeakOnce` a
`nontrivial` one, listed in the lock's `witnesses`; none takes a binder over its instance:

* `PeriodLeakAccepted_honest`: under `SampledCodeRuns`, the one named assumption, the premises that are no named
  assumption hold at the deployed instance: `Claims.Sampling.KS` at SHA-512 at the deployed sampling record
  `SamplingDeploy.deploy`, whose commitment and sessions' Flock record are `deploy`'s (`ks_deployed`); its decoder
  within Flock's radius (`Sampling.decoder_meets_zk`); its hm96 commitment binding (`Binding_of_hm96`); and the rows
  format of the Python network certifier's fixture `fxQueue` fitting its schedule (`fxQueue_fits`). A statement is in
  the record's scope. The other premises are bounds and equalities over the instance's parameters, and `TrustedHost`
  (`trusted-host`), a named assumption.
* `PeriodLeakAccepted_nontrivial`: the per-window bound fails at the accepting verifier `tapD` at SHA-512, at
  `δ = 1/8` (`tapD_bound_fails`).
* `PeriodLeakOnce_nontrivial`: its conclusion fails at `scopelessD`, whose verifier accepts everything, with the rows
  format `fxRows`, no window and `Θ = {(1, 0)}`. The device `leakyRun` shows the client a cell its record doesn't hold
  on one of two servings, so the two are told apart against the count `#Θ = 1`. `PeriodLeakOnce_untrusted`, beside it
  and unlisted, makes it an exclusion: at that instance every premise but `TrustedHost` holds, and `TrustedHost` fails.
* `PooledLeakAccepted_honest`: its premises on the schedules, at `fxQueue`'s alone: each fits the format (`hfit`) and
  sends at most 6 cells a connection (`hcap`). Stated as their conjunction, not as `fits` alone: the witnesses rule
  reads a statement that is one definition applied by unfolding it once, and `fits` is a field.
* `PooledLeakAccepted_nontrivial`: its bound fails at an audit that always accepts, at `δ = 1/2` with one window of
  one step whose faulty set holds its one unit (`pooled_bound_fails`). There its conditional escape (`hesc`) holds
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

/-- **`PeriodLeakOnce`'s conclusion fails** at `scopelessD`, whose verifier accepts everything, the rows format
`fxRows`, no window and `Θ = {(1, 0)}`: the device `leakyRun`, which shows the client a cell its record doesn't hold on
one of two servings, tells the two apart, so `2` servings are counted against `#Θ = 1`. -/
theorem PeriodLeakOnce_nontrivial :
    ¬ ((∀ m, ScopeAcc (scopelessD Unit) (fun _ => notLay) (fun _ => 1) m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
        PrBad shaD (scopelessD Unit).V
            (RLeakTap shaD (scopelessD Unit).commit (scopelessD Unit).parse (scopelessD Unit).partition 1 1 1
              (fun _ => notLay) (fun _ => 1) 0 (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => []))
            (Claims.Sampling.E (scopelessD Unit) fun _ => [()]) m P ≤
          Claims.Sampling.κ (scopelessD Unit) 0 m +
              Real.sqrt (Claims.Sampling.γ (scopelessD Unit) m * PrB shaD (scopelessD Unit).V P m) +
            (Bext shaD (scopelessD Unit).V (Claims.Sampling.E (scopelessD Unit) fun _ => [()]) Ω P m).Pr shaD +
            (Btap shaD (scopelessD Unit).V (Claims.Sampling.E (scopelessD Unit) fun _ => [()])
              (scopelessD Unit).commit (scopelessD Unit).parse (fun _ => []) Ω P m).Pr shaD) ∧
      ((∀ z ∈ (Set.univ : Set Bool), ∀ (i : Fin 0) s, (scopelessD Unit).parse (Fin.elim0 i) = some s →
          TapOpens shaD (scopelessD Unit).commit notLay 1 (fun _ _ => []) s [] ((leakyRun () (emptySched, z)).2 i)) →
        {z : Bool | z ∈ Set.univ ∧ (emptySched.r, emptySched.D) ∈ ({(1, 0)} : Set (ℕ × ℕ)) ∧
            fxRows.Fits 1 0 emptySched ∧ leakyDec (leakyRun () (emptySched, z)).1 = z ∧
            ∀ i : Fin 0, WindowGood shaD (scopelessD Unit) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅) 1
              (Fin.elim0 i) (fun _ => false) (fun _ => []) (Fin.elim0 i) (Fin.elim0 i)
              ((leakyRun () (emptySched, z)).2 i)}.encard ≤
          ({(1, 0)} : Set (ℕ × ℕ)).encard *
            ((∏ i : Fin 0, (2 ^ (freeInputs (Fin.elim0 i : AuditStatement).prog notLay 1 \ ∅).card *
              (faultySets (Fin.elim0 i : AuditStatement).design ((scopelessD Unit).part (Fin.elim0 i)) 0).card *
              2 ^ ((faultySets (Fin.elim0 i : AuditStatement).design ((scopelessD Unit).part (Fin.elim0 i)) 0).sup
                (faultyBits (Fin.elim0 i : AuditStatement).prog ((scopelessD Unit).part (Fin.elim0 i)))) *
              1 ^ 1) : ℕ) : ℕ∞))) := by
  rintro ⟨-, hc⟩
  have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
    ((Set.encard_le_encard fun z _ => Set.mem_ofPred.2
      ⟨Set.mem_univ z, rfl, emptySched_fits, leakyDec_run z, fun i => i.elim0⟩).trans (hc fun _ _ i => i.elim0))
  simp at h2

/-- **`PeriodLeakOnce_nontrivial`'s instance drops trusted-host alone**: `KS`, binding and the decoder's radius hold
at `scopelessD` (no statement is in its scope), `fxRows` is well formed, `0 ≤ δ` at `δ = 0`, and `leakyRun` doesn't
run the network certifier's code. So without `TrustedHost` the count fails where every other premise holds. -/
theorem PeriodLeakOnce_untrusted :
    Claims.Sampling.KS shaD (scopelessD Unit) ∧
      Claims.Flock.ListWithin (scopelessD Unit).flock.len (scopelessD Unit).flock.enc (scopelessD Unit).flock.θ
        (fun _ => [()]) ∧
      (scopelessD Unit).commit.Binds shaD ∧ fxRows.Wf ∧ (0 : ℝ) ≤ 0 ∧
      ¬ Verity.Assumptions.TrustedHost () leakyRun (fun p =>
        (wardenCode p.1 (noArr p.2), fxRows.record 1 0 p.1 (noArr p.2) (wardenCode p.1 (noArr p.2)))) :=
  ⟨scopelessD_ks shaD, scopelessD_listWithin, scopelessD_binds shaD, fxRows_wf, le_rfl, leakyRun_untrusted⟩

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
