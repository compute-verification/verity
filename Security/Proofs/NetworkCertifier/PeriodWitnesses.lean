import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.KSWired
import Definitions.NetworkCertifier.Period

/-!
# The monitoring period's honest and nontrivial lemmas

`PooledLeakAccepted` has an `honest` and a `nontrivial` lemma, and `PeriodLeakAccepted` and `PeriodLeakRegistered`
a `nontrivial` one each, listed in the lock's `witnesses`, and none takes a binder over its instance; the lock's
`unwitnessed` exempts `PeriodLeakAccepted`'s and `PeriodLeakRegistered`'s `honest` halves.

* `PeriodLeakAccepted_nontrivial`: the per-window bound fails at the accepting verifier `tapD` at SHA-512, at
  `δ = 1/8` (`tapD_bound_fails`).
* `PeriodLeakRegistered_nontrivial`: its per-window bound fails at `tapD` likewise, with a registration with no
  openings, which opens every registered input when none is registered (`tapD_reg_bound_fails`). Its `honest` half is
  the sampled leg's guarantee at the honest deployment (`Verity.Vocab.SampledProofsSound`), which the lock exempts until
  sampled-audits-sound proves it.
* `not_periodLeakRegisteredUntrusted`, unlisted, its exclusion of `trusted-host`: the statement with `TrustedHost`
  dropped and every other premise kept (`PeriodLeakRegisteredUntrusted`) is false, at `scopelessD` (no statement in
  its scope, so `KS` holds), the well-formed rows format `fxRows`, no window and `Θ = {(1, 0)}`, where the device
  `leakyRun` tells two servings apart against `#Θ = 1`.
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

/-- **`PeriodLeakRegistered`'s per-window bound fails at the accepting verifier** `tapD`, at SHA-512, at one step at
`δ = 1/8` with no input registered and a registration with no openings, which opens every registered input
(`tapD_reg_bound_fails`). -/
theorem PeriodLeakRegistered_nontrivial :
    ¬ PrBad shaD (tapD Unit).V
        (RLeakTap shaD (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay)
          (fun _ => 1) (1 / 8) (fun s _ => AsReg s.prog notLay 1 ∅ (regVal [])) 1 (fun _ => tapOps))
        (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (tapD Unit) (1 / 8) [] +
          Real.sqrt (Claims.Sampling.γ (tapD Unit) [] * PrB shaD (tapD Unit).V silentP []) +
        (Bext shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) Unit silentP []).Pr shaD +
        (Btap shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
          (fun _ => tapOps) Unit silentP []).Pr shaD +
        (Breg shaD (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
          (fun _ => []) Unit silentP []).Pr shaD :=
  tapD_reg_bound_fails shaD

/-- **`PeriodLeakRegistered` without `TrustedHost`**: its statement with the device's run any function of the
schedule and the server, every other hypothesis kept. -/
def PeriodLeakRegisteredUntrusted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Claims.Sampling.Deploy Coin Col T) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S), KSWired H D (fun s => (lay s).wires) →
    ∀ (k : AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T), 0 ≤ δ →
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool)) (Dj : ℕ)
      {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition)
      (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ),
    WindowsPlaced D δ sw Pw β →
    ∀ {Conn Cell Srv Dev Z : Type} (arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K), F.Wf →
    ∀ (tap : Dev)
      (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
      (Θ : Set (ℕ × ℕ)) (Zs : Set Z) (schedOf : Z → WardenSchedule Conn) (srvOf : Z → Srv),
    SchedulesCovered {s | (s.r, s.D) ∈ Θ ∧ F.Fits S n s} Zs schedOf →
    ∀ (dec : (Conn → List (ℕ × CellView Cell)) → Z), SenderDecodes run tap Zs schedOf srvOf dec →
    ∀ (x : Z → Fin n → Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)), D.commit.Binds H →
    (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (Claims.Sampling.E D decode) m P ≤
        Claims.Sampling.κ D δ m + Real.sqrt (Claims.Sampling.γ D m * PrB H D.V P m) +
          (Bext H D.V (Claims.Sampling.E D decode) Ω P m).Pr H +
          (Btap H D.V (Claims.Sampling.E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (Claims.Sampling.E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
      (TapsOpen H D lay k op tops Zs x (fun z => (run tap (schedOf z, srvOf z)).2) →
        {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

/-- **Without `TrustedHost`, `PeriodLeakRegistered` is false**: at `scopelessD`, where `KS` holds (no statement is in
its scope, `scopelessD_ks`) and so sampling at the incorrect set (`notLay` has no wires), the rows format `fxRows`
(well formed), no window, `Θ = {(1, 0)}` and the schedule `emptySched` (which fits), the device `leakyRun`, which
shows the client a cell its record doesn't hold on one of two servings (`leakyRun_untrusted`), tells the two apart,
so `2` secrets are counted against `#Θ = 1`. -/
theorem not_periodLeakRegisteredUntrusted : ¬ PeriodLeakRegisteredUntrusted := fun h => by
  have hc := (h shaD (scopelessD Unit) 1 1 1 (fun _ => notLay) (ksWired_none (fun _ => rfl) (scopelessD_ks shaD))
    (fun _ => 1) 0 (fun _ => [()]) le_rfl scopelessD_listWithin (fun _ => ∅) (fun _ => []) 1 (n := 0) Fin.elim0
    Fin.elim0 Fin.elim0 Fin.elim0 Fin.elim0 ⟨fun i => i.elim0, fun i => i.elim0⟩ noArr fxRows fxRows_wf () leakyRun
    {(1, 0)} Set.univ (fun _ => emptySched) id (fun _ _ => ⟨rfl, emptySched_fits⟩) leakyDec
    (fun z _ => leakyDec_run z) (fun _ => Fin.elim0) (fun _ => Fin.elim0) (fun _ => []) (scopelessD_binds shaD)).2
    fun _ _ i => i.elim0
  have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
    ((Set.encard_le_encard fun z _ => Set.mem_setOf.2 ⟨Set.mem_univ z, fun i => i.elim0⟩).trans hc)
  simp at h2

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
