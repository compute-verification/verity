import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.PooledLeak
import Proofs.NetworkCertifier.Assumptions.SampledCode

/-!
# The monitoring period's honest and nontrivial lemmas

`PeriodLeakAccepted` and `PooledLeakAccepted` each have an `honest` and a `nontrivial` lemma, listed in the lock's
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
* `PooledLeakAccepted_honest`: its premises on the schedules, at `fxQueue`'s alone: each fits the format (`hfit`) and
  sends at most 6 cells a connection (`hcap`); and sampled proofs' pooled bound at the tied audits' samples
  (`hpool`, `TiedIncorrectBlocksLe`) at the demo's tied sample, a uniform 169 of 7,000,000 proof blocks, at
  `F_t = 574205` and `ε_t = (1 - q_t)^F_t`, which `tiedIncorrectBlocksLe_of_samples` derives from
  `PooledIncorrectBlocksLe` for an honest prover, whose tied audits hold no incorrect proof block. Stated as their
  conjunction, not as `fits` alone: the witnesses rule reads a statement that is one definition applied by unfolding it
  once, and `fits` is a field.
* `PooledLeakAccepted_nontrivial`: its bound fails at an audit that always accepts, at `δ = 1/2` with one window of
  one step tied to `tiedToy`, whose one proof block is incorrect (`pooled_bound_fails`). There sampled proofs' pooled
  bound (`hpool`) fails: every audit accepts with that proof block incorrect on every coin, and `ε_t = 0`.
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

/-- `TiedIncorrectBlocksLe` at one window whose tied audits draw a uniform `k` of their `n` proof blocks, for a prover
with no incorrect tied proof block: `tiedIncorrectBlocksLe_of_samples` at `ε = 0`. -/
theorem tiedIncorrectBlocksLe_uniform_none (n k : ℕ) (hk : k ≤ n) (F : ℕ) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hq : q * n ≤ k) :
    TiedIncorrectBlocksLe (Set.univ : Set Unit)
      (fun (_ : Unit) (_ : Fin 1) (_ : Fin 1 → {S : Finset (Fin n) // S.card = k}) _ => True)
      (fun _ _ _ => ∅) F ((1 - q) ^ F + ∑ _ : Fin 1, (0 : ℝ)) := by
  have : Nonempty {S : Finset (Fin n) // S.card = k} := Fintype.card_pos_iff.1 (by
    rw [Fintype.card_finset_len, Fintype.card_fin]
    exact Nat.choose_pos hk)
  exact tiedIncorrectBlocksLe_of_samples (m := 1) (Ω := {S : Finset (Fin n) // S.card = k}) (Z := Unit) Set.univ
    (fun _ _ _ _ => True) (fun _ => n) (fun _ => k) (fun _ x => x.1) (fun _ => (uniform_draw n k).1)
    (fun _ => (uniform_draw n k).2) (fun _ _ _ => ∅) (fun _ _ _ _ _ => rfl) (fun _ _ _ => ∅)
    (fun _ _ _ => (Finset.map_empty _).symm) (fun _ => 0) (fun _ _ _ => by simp [Pr]) q hq0 hq1 (fun _ => hq) F

/-- **`PooledLeakAccepted`'s premises at the deployed instance**, its schedules the fixture `fxQueue`'s alone: each fits
its rows format (`hfit`), and sends at most 6 cells on each connection (`hcap`); and sampled proofs' pooled bound at the
tied audits' samples (`hpool`) at the demo's tied sample, a uniform 169 of 7,000,000 proof blocks, at
`F_t = 574205` (`tiedF` at `δ_t = 10^-6` and `q_t = 169/7,000,000`) and `ε_t = (1 - q_t)^F_t`, for an honest prover,
whose tied audits have no incorrect proof block: `tiedIncorrectBlocksLe_of_samples` at `ε = 0`. -/
theorem PooledLeakAccepted_honest :
    (∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits s) ∧
      (∀ s ∈ ({fxQueueSched} : Set (WardenSchedule (Fin 3))), ∀ c, s.cells c ≤ 6) ∧
      TiedIncorrectBlocksLe (Set.univ : Set Unit)
        (fun (_ : Unit) (_ : Fin 1) (_ : Fin 1 → {S : Finset (Fin 7000000) // S.card = 169}) _ => True)
        (fun _ _ _ => ∅) 574205 ((1 - 169 / 7000000) ^ 574205 + ∑ _ : Fin 1, (0 : ℝ)) := by
  refine ⟨fun s hs => ?_, fun s hs c => ?_, ?_⟩
  · rw [Set.mem_singleton_iff.1 hs]
    exact fxQueue_fits
  · rw [Set.mem_singleton_iff.1 hs]
    fin_cases c <;> decide
  · exact tiedIncorrectBlocksLe_uniform_none 7000000 169 (by norm_num) 574205 (169 / 7000000) (by norm_num)
      (by norm_num) (by norm_num)

/-- **`PooledLeakAccepted`'s bound fails at an audit that always accepts**, at one window of one step tied to
`tiedToy`, whose one proof block is incorrect, `Fmax = Ft = 1`, `δ = 1/2`, `ε_t = 0` and `e = e' = 0`
(`pooled_bound_fails`). -/
theorem PooledLeakAccepted_nontrivial :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ => ([] : List Bool)) (fun _ => [tiedToy]) 1 1 (fun _ => ∅) (fun _ => {0})
          (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + 0 + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) :=
  pooled_bound_fails

end NetTiming.Witness
