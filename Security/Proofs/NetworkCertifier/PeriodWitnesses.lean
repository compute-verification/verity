import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.WitnessToys
import Proofs.NetworkCertifier.KSWired
import Proofs.NetworkCertifier.KsToy
import Definitions.NetworkCertifier.Period

/-!
# The monitoring period's honest and nontrivial lemmas

`PooledLeakAccepted` has an `honest` and a `nontrivial` lemma, and `PeriodLeakAccepted` a `nontrivial` one, listed in
the lock's `witnesses`, and none takes a binder over its instance; the lock's `unwitnessed` exempts
`PeriodLeakAccepted`'s `honest` half. `Properties.NetworkPeriodLeak.PeriodLeakRegistered`'s are in
`Proofs.Properties.NetworkPeriodLeak`; its general form `PeriodLeakRegisteredWf` has two unlisted ones here.

* `PeriodLeakAccepted_nontrivial`: the per-window bound fails at the accepting verifier `tapD` at SHA-512, at
  `δ = 1/8` (`tapD_bound_fails`).
* `not_periodLeakRegisteredUntrusted`, unlisted, its exclusion of `trusted-host`: the statement with `TrustedHost`
  dropped and every other premise kept (`PeriodLeakRegisteredUntrusted`) is false, at `scopelessD` (no statement in
  its scope, so `KS` holds), the well-formed rows format `fxRows`, no window and `Θ = {(1, 0)}`, where the device
  `leakyRun` tells two servings apart against `#Θ = 1`.
* `not_periodLeakRegisteredAnyFormat`, unlisted, why it reads `RowsFormat.Wf`, and so why the period check refuses a
  schedule whose frames don't read back (`NetworkCheck.Period.rowsWf`): the statement with `hF` dropped and
  every other premise kept, `TrustedHost` included (`PeriodLeakRegisteredAnyFormat`), is false, at `ksNotD` (where
  `KS` holds, `KsToy`), `WitnessToys`' rows format `blindRows`, which encodes every frame as the empty list, one
  window and `Θ = {(1, 0)}`, where the network certifier's code, recorded in `blindRows`, registers one record on
  every serving while the client's cell is the serving's, against a count of `1`.
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

/-- **`PeriodLeakRegisteredWf` without `TrustedHost`**: its statement with the device's run any function of the
schedule and the server, every other hypothesis kept. -/
def PeriodLeakRegisteredUntrusted : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Claims.Sampling.Deploy Coin Col T) (L K S : ℕ)
    (lay : Claims.Sampling.AuditStatement → BoundaryGates L S), KSBy H D (wiredOk fun s => (lay s).wires) →
    ∀ (k : Claims.Sampling.AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T), 0 ≤ δ →
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ (reg : Claims.Sampling.AuditStatement → Finset ℕ)
      (ops : Claims.Sampling.AuditStatement → List (Opening Bool)) (Dj : ℕ)
      {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement) (Pw : Fin n → Claims.Sampling.Partition)
      (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ),
    WindowsPlaced D δ sw Pw β →
    ∀ {Conn Cell Srv Dev Sec : Type} (_arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K), F.Wf →
    ∀ (cert : Dev)
      (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
      (Θ : Set (ℕ × ℕ)) (Zs : Set Sec) (schedOf : Sec → WardenSchedule Conn) (srvOf : Sec → Srv),
    SchedulesCovered {s | (s.r, s.D) ∈ Θ ∧ F.Fits S n s} Zs schedOf →
    ∀ (dec : (Conn → List (ℕ × CellView Cell)) → Sec), SenderDecodes run cert Zs schedOf srvOf dec →
    ∀ (x : Sec → Fin n → Claims.Msg) (wit : Sec → Fin n → Claims.Sampling.Wit)
      (tops : Claims.Sampling.AuditStatement → List (Opening Bool)), D.commit.Binds H →
    (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (Claims.Sampling.E D decode) m P ≤
        Claims.Sampling.κ D δ m + Real.sqrt (Claims.Sampling.γ D m * PrB H D.V P m) +
          (Bext H D.V (Claims.Sampling.E D decode) Ω P m).Pr H +
          (Btap H D.V (Claims.Sampling.E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (Claims.Sampling.E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
      (CertifiersOpen H D lay k op tops Zs x (fun z => (run cert (schedOf z, srvOf z)).2) →
        {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run cert (schedOf z, srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

/-- **Without `TrustedHost`, `PeriodLeakRegisteredWf` is false**: at `scopelessD`, where `KS` holds (no statement is in
its scope, `scopelessD_ks`) and so sampling at the incorrect set (`notLay` has no wires), the rows format `fxRows`
(well formed), no window, `Θ = {(1, 0)}` and the schedule `emptySched` (which fits), the device `leakyRun`, which
shows the client a frame its record doesn't hold on one of two servings (`leakyRun_untrusted`), tells the two apart,
so `2` secrets are counted against `#Θ = 1`. -/
theorem not_periodLeakRegisteredUntrusted : ¬ PeriodLeakRegisteredUntrusted := fun h => by
  have hc := (h shaD (scopelessD Unit) 1 1 1 (fun _ => notLay) (ksWired_none (fun _ => rfl) (scopelessD_ks shaD))
    (fun _ => 1) 0 (fun _ => [()]) le_rfl scopelessD_listWithin (fun _ => ∅) (fun _ => []) 1 (n := 0) Fin.elim0
    Fin.elim0 Fin.elim0 Fin.elim0 Fin.elim0 ⟨fun i => i.elim0, fun i => i.elim0⟩ noArr fxRows fxRows_wf () leakyRun
    {(1, 0)} Set.univ (fun _ => emptySched) id (fun _ _ => ⟨rfl, emptySched_fits⟩) leakyDec
    (fun z _ => leakyDec_run z) (fun _ => Fin.elim0) (fun _ => Fin.elim0) (fun _ => []) (scopelessD_binds shaD)).2
    fun _ _ i => i.elim0
  have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
    ((Set.encard_le_encard (s := ({true, false} : Set Bool)) fun z _ =>
      Set.mem_ofPred.2 ⟨Set.mem_univ z, fun i => i.elim0⟩).trans hc)
  simp at h2

/-- **`PeriodLeakRegisteredWf` at any rows format**: its statement without `F.Wf`, every other hypothesis kept,
`TrustedHost` included. -/
def PeriodLeakRegisteredAnyFormat : Prop :=
  ∀ {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Claims.Msg → Claims.Msg) (D : Claims.Sampling.Deploy Coin Col T) (L K S : ℕ)
    (lay : Claims.Sampling.AuditStatement → BoundaryGates L S), KSBy H D (wiredOk fun s => (lay s).wires) →
    ∀ (k : Claims.Sampling.AuditStatement → ℕ) (δ : ℝ) (decode : List (ℕ × Col) → List T), 0 ≤ δ →
    Claims.Flock.ListWithin D.flock.len D.flock.enc D.flock.θ decode →
    ∀ (reg : Claims.Sampling.AuditStatement → Finset ℕ)
      (ops : Claims.Sampling.AuditStatement → List (Opening Bool)) (Dj : ℕ)
      {n : ℕ} (sw : Fin n → Claims.Sampling.AuditStatement) (Pw : Fin n → Claims.Sampling.Partition)
      (ρ : Fin n → ℕ → Bool) (op : Fin n → ℕ → List (Obs L K S) → List Bool) (β : Fin n → ℕ),
    WindowsPlaced D δ sw Pw β →
    ∀ {Conn Cell Srv Dev Sec : Type} (arr : Srv → Conn → Arrivals Cell) (F : RowsFormat Conn Cell L K) (cert : Dev)
      (run : Dev → WardenSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S))),
    Verity.Assumptions.TrustedHost cert run (fun p =>
      (laneWire p.1 (wardenCode p.1 (arr p.2)), F.record S n p.1 (arr p.2) (wardenCode p.1 (arr p.2)))) →
    ∀ (Θ : Set (ℕ × ℕ)) (Zs : Set Sec) (schedOf : Sec → WardenSchedule Conn) (srvOf : Sec → Srv),
    SchedulesCovered {s | (s.r, s.D) ∈ Θ ∧ F.Fits S n s} Zs schedOf →
    ∀ (dec : (Conn → List (ℕ × CellView Cell)) → Sec), SenderDecodes run cert Zs schedOf srvOf dec →
    ∀ (x : Sec → Fin n → Claims.Msg) (wit : Sec → Fin n → Claims.Sampling.Wit)
      (tops : Claims.Sampling.AuditStatement → List (Opening Bool)), D.commit.Binds H →
    (∀ m, ScopeAcc D lay k m → RegOpensAt H D reg ops m →
      ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      PrBad H D.V
          (RLeakTap H D.commit D.parse D.partition L K S lay k δ
            (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops)
          (Claims.Sampling.E D decode) m P ≤
        Claims.Sampling.κ D δ m + Real.sqrt (Claims.Sampling.γ D m * PrB H D.V P m) +
          (Bext H D.V (Claims.Sampling.E D decode) Ω P m).Pr H +
          (Btap H D.V (Claims.Sampling.E D decode) D.commit D.parse tops Ω P m).Pr H +
          (Breg H D.V (Claims.Sampling.E D decode) D.commit D.parse ops Ω P m).Pr H) ∧
      (CertifiersOpen H D lay k op tops Zs x (fun z => (run cert (schedOf z, srvOf z)).2) →
        {z | z ∈ Zs ∧ ∀ i, WindowGoodReg H D L K S lay k δ reg ops Dj (sw i) (ρ i) tops (x z i) (wit z i)
            ((run cert (schedOf z, srvOf z)).2 i)}.encard ≤
          Θ.encard * ((∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \
            reg (sw i)).card * (faultySets (sw i).design (Pw i) δ).card * 2 ^ β i * Dj ^ k (sw i)) : ℕ) : ℕ∞))

/-- **Without `RowsFormat.Wf`, `PeriodLeakRegisteredWf` is false**: at `ksNotD`, where `KS` holds (`ksNotD_KS`) and so
`KSBy` at the wired check (`freeLay` has no wires), the registration `goodReg`, which opens (`ksNotD_regOpens`), one
window at `δ = 0` and `Θ = {(1, 0)}`, the network certifier's code recorded in `blindRows` (which doesn't read back,
`blindRows_not_wf`) on the schedule `oneSched` (which fits) registers `obs1` on every serving `z`, the client's cell
being `z` (`cellDec_run`). The circuit accepts `obs1` at `goodReg` (`frameTops_acc`), so every secret meets
`WindowGoodReg`, `0` and `1` among them: `2` against a count of `#Θ · 2^0 · 1 · 2^0 · 1^1 = 1`. -/
theorem not_periodLeakRegisteredAnyFormat : ¬ PeriodLeakRegisteredAnyFormat := fun h => by
  have hgood : ∀ o : List (Obs 1 1 1), o = [obs1] →
      WindowGoodReg shaD (ksNotD Unit) 1 1 1 (fun _ => freeLay) (fun _ => 1) 0 (fun _ => {0}) (fun _ => goodReg) 1
        notStmt (regVal goodReg) (fun _ => frameTops) [] ⟨[]⟩ o := by
    rintro o rfl
    exact ⟨⟨notStmt, notPart, rfl, rfl, fun op o hto hcov hlen hj => frameTops_acc shaD op o hto hcov hlen hj⟩,
      ksNotD_scopeAcc [], notStmt, rfl, ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, fun _ _ => rfl, obs1_covered, rfl,
      fun y hy => by obtain rfl := List.mem_singleton.1 hy; exact Nat.one_pos⟩
  have hopens : ∀ o : List (Obs 1 1 1), o = [obs1] →
      TapOpens shaD notCm freeLay 1 (fun _ _ => []) notStmt frameTops o := by
    rintro o rfl
    exact frameTops_opens shaD
  have hc := (h shaD (ksNotD Unit) 1 1 1 (fun _ => freeLay) (ksWired_none (fun _ => rfl) (ksNotD_KS shaD))
    (fun _ => 1) 0 (fun _ => [()]) le_rfl (Claims.Witness.Flock.listWithin_unit _ _ _) (fun _ => {0})
    (fun _ => goodReg) 1 (n := 1) (fun _ => notStmt) (fun _ => notPart) (fun _ => regVal goodReg)
    (fun _ _ _ => []) (fun _ => 0) ⟨fun _ => rfl, fun _ F hF => by obtain rfl := notFaulty_empty hF; simp [faultyBits]⟩
    cellArr blindRows () blindRun (fun _ => rfl) {(1, 0)} Set.univ (fun _ => oneSched) id
    (fun _ _ => ⟨rfl, oneSched_fits⟩) cellDec (fun z _ => cellDec_run z) (fun _ _ => []) (fun _ _ => ⟨[]⟩)
    (fun _ => frameTops) (notCm_binds shaD)).2 fun z _ i s hs => by
      obtain rfl : notStmt = s := Option.some.inj hs
      exact hopens _ (blindRun_obs z i)
  have h2 := (Set.encard_pair (show (0 : ℕ) ≠ 1 by decide)).symm.le.trans
    ((Set.encard_le_encard (s := ({0, 1} : Set ℕ)) fun z _ =>
      Set.mem_ofPred.2 ⟨Set.mem_univ z, fun i => hgood _ (blindRun_obs z i)⟩).trans hc)
  have hF1 : faultySets notStmt.design notPart 0 = {∅} :=
    Finset.eq_singleton_iff_unique_mem.2 ⟨empty_mem_notFaulty, fun _ hF => notFaulty_empty hF⟩
  have hfree : (freeInputs notStmt.prog freeLay 1 \ {0}).card = 0 := by
    decide
  simp only [hF1, hfree, Finset.card_singleton] at h2
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
