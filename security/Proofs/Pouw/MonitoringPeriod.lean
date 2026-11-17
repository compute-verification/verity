module

public import Properties.Pouw.MonitoringPeriod
public import Proofs.Compose.ComposeMonitoringPeriod
public import Proofs.Compose.Compose.SplitRateClosed
public import Proofs.Pouw.RelativeSecurity
public import Proofs.ResourceAccounting.Lemmas.Compose

/-! A period's work over the windows PoUW's deployed check accepts (`Properties.Pouw.MonitoringPeriod`); `Pins`
is taken as a hypothesis, and `RegisteredBeforeDraw` is not read. -/

@[expose] public section

namespace Properties.Pouw.MonitoringPeriod.SecurityProofs

open PouwCheck Properties.Pouw.MonitoringPeriod
open ResourceAccounting.TS ResourceAccounting.Compose ResourceAccounting.Assumptions.TS ResourceAccounting.Assumptions.Compose Pouw.RelativeSecurity

theorem credited_iff {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (hKd : 0 < Kd) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n) :
    (checkedRun hK p ws opened rel right).Credited x w ↔ ws.AcceptedUpTo p x w :=
  ⟨fun h w' hw => h w' hw ⟨0, hKd⟩, fun h w' hw _ => h w' hw⟩

theorem prefixLen_eq {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (hKd : 0 < Kd) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) :
    (checkedRun hK p ws opened rel right).prefixLen x = ws.prefixLen p x := by
  classical
  unfold Outcomes.prefixLen Executions.prefixLen
  congr 1
  exact Finset.filter_congr fun w _ => credited_iff hK hKd p ws opened rel right x w

theorem creditedLive_eq {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {Kd Np n : ℕ} (hK : Kd ≤ Np)
    (hKd : 0 < Kd) (p : Params) (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : ExecutionPhasePlay C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (x : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n) :
    creditedLive Sc Cd Wp (checkedRun hK p ws opened rel right) x = checkedLive Sc Cd Wp p ws x := by
  classical
  unfold creditedLive checkedLive
  exact Finset.sum_congr rfl fun w _ => if_congr (credited_iff hK hKd p ws opened rel right x w) rfl rfl

theorem jointSound_checked {ι C Ξ V : Type} [Fintype C] [Fintype Ξ] (Cd : Declared ι) {Kd Np n : ℕ}
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (p : Params) (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    {δks : ℝ} (hD : SampleIsCoin hK p ws) (hS : ServiceSound p ws opened rel δks) :
    (checkedLayer Cd hr hK p ws opened rel right).JointSound δks := by
  intro w
  refine (Pouw.RelativeSecurity.prob_mono fun x hx => ?_).trans (hS w)
  obtain ⟨i, hacc, hrel⟩ := hx
  obtain ⟨pub, prof, hc⟩ := hacc
  exact ⟨pub, prof, hc, drawOf hK (coinAt x w).2 i, ((hD x w pub prof hc).2.2 _).2 ⟨i, rfl⟩, hrel⟩

/-- **`CheckedWorkMonitoringPeriod`**: `CircuitWorkMonitoringPeriod` at `checkedRun`, its accepted prefix and live words the check's. -/
theorem checkedWorkMonitoringPeriod : CheckedWorkMonitoringPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn hN hr hK hKd prm ws opened rel right Wp Pl folded
    ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved _hReg hDraw hServ hPins
    hMu hRB hGs hUF
  have h := ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod.CircuitWorkMonitoringPeriod Sc Cd hn hN hr hK Wp Pl
    (checkedRun hK prm ws opened rel right) folded ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG
    hDD hPre hMoved (jointSound_checked Cd hr hK prm ws opened rel right hDraw hServ) hPins hMu hRB hGs hUF
  simp only [prefixLen_eq hK hKd prm ws opened rel right,
    creditedLive_eq Sc Cd hK hKd prm ws opened rel right Wp] at h
  exact h

/-- **`CheckedCreditMonitoringPeriod`**: `CircuitCreditMonitoringPeriod` at `checkedRun`, its accepted prefix and live words the
check's. -/
theorem checkedCreditMonitoringPeriod : CheckedCreditMonitoringPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn hN hr hK hKd prm ws opened rel right Wp Pl folded
    ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved _hReg hDraw hServ hPins
    hMu hRB hGs hUF
  have h := ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod.CircuitCreditMonitoringPeriod Sc Cd hn hN hr hK Wp Pl
    (checkedRun hK prm ws opened rel right) folded ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG
    hDD hPre hMoved (jointSound_checked Cd hr hK prm ws opened rel right hDraw hServ) hPins hMu hRB hGs hUF
  simp only [prefixLen_eq hK hKd prm ws opened rel right,
    creditedLive_eq Sc Cd hK hKd prm ws opened rel right Wp] at h
  exact h

theorem splitRate_h1 : SplitRate 24 (8 / 9) (7 / 8) (1 / 2) 1 1 1 (67 / 8) := by
  have h := ResourceAccounting.SecurityProofs.Compose.Rates.splitRateClosed 24 (8 / 9) (7 / 8) (1 / 2) 1 1 1 (by norm_num)
    (by norm_num) zero_le_one (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) one_pos one_pos
  convert h using 1
  rw [max_eq_left] <;> norm_num

theorem pins_h1 {C B V : Type} {m n Np : ℕ} (hm : 0 < m) {salt : Paths (Paths C m × B) n → Fin n → ByteArray}
    {enc : C → ByteArray} {staged flat : ByteArray → Fin Np → (ℕ → V) → Prop} (hs : SaltIsLiveCoin hm salt enc)
    (hp : StagedPins staged flat) : ∀ x w j v, stagedAt salt staged x w j v → flatAtCoin hm enc flat x w j v := by
  intro x w j v h
  have h' := hp (salt x w) j v h
  rw [hs x w] at h'
  exact h'

/-- **`H1CheckedCreditMonitoringPeriod`**: `CheckedCreditMonitoringPeriod` at H1's numbers (`splitRate_h1`), its `Pins` from `pins_h1`. -/
theorem h1CheckedCreditMonitoringPeriod : H1CheckedCreditMonitoringPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr _hM hK prm ws opened salt enc staged flat Wp Pl folded
    δs δbind δks hS hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hStaged hMu hRB hGs hUF
  exact checkedCreditMonitoringPeriod Sc Cd hn hN hr hK (by norm_num) prm ws opened (stagedAt salt staged)
    (flatAtCoin hr enc flat) Wp Pl folded (8 / 9) (7 / 8) (1 / 2) 1 1 1 (67 / 8) δs δbind δks hS (by norm_num) hρN
    hwmax hlk one_pos one_pos (by norm_num) splitRate_h1 hG hDD hPre hMoved hReg hDraw hServ (pins_h1 hr hSalt hStaged)
    hMu hRB hGs hUF

end Properties.Pouw.MonitoringPeriod.SecurityProofs
