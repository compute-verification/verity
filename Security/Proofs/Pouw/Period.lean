import Properties.Pouw.Period
import Proofs.Accounting.ComposePeriod
import Proofs.Accounting.Compose.SplitRateClosed
import Proofs.Pouw.RelativeSecurity

/-!
# Proofs: a period's work over the windows PoUW's deployed check accepts (`Properties.Pouw.Period`)

`CheckedWorkPeriod` and `CheckedCreditPeriod` are `CircuitWorkPeriod` and `CircuitCreditPeriod` at the period's proofs
with the check's verdict (`checkedRun`). There a window is credited exactly when it and every earlier window are
accepted (`credited_iff`, at `K_d > 0`: the check's verdict is every drawn tile's), so the run's accepted prefix and its
live words are the check's (`prefixLen_eq`, `creditedLive_eq`). `JointSound` holds at `δ_ks`: an unsound drawn tile of
an accepted window is a unit the check publishes as drawn (`DrawIsCoin`) whose opening fails its relation, which
`ServiceSound` bounds (`jointSound_checked`). `Pins` is the hypothesis itself, and `RegisteredBeforeDraw`, protocol
structure, is not read.

`H1CheckedCreditPeriod` is `CheckedCreditPeriod` at H1's numbers, with `SplitRateClosed`'s rate there
(`splitRate_h1`: `Λ = max(67/8, 67/32) = 67/8`) and `Pins` from the window's salt being its live coin and the staged
statement pinning the flat tile (`pins_h1`).

Its premises are met: `Proofs.Pouw.Period.Accept` evaluates the deployed `PouwCheck.protocol.check` to an accept at
the H1 loop's parameters for every ascending draw of 24 of 144 units (`check_h1`), and `Proofs.Pouw.Period.Honest`
builds a period of those windows, the salt each window's live coin, that meets every premise (`h1_meets`).
-/

namespace Properties.Pouw.Period.SecurityProofs

open PouwCheck Properties.Pouw.Period
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **A window is credited under the check exactly when it and every earlier one are accepted**, at `K_d > 0`. -/
theorem credited_iff {C Ξ V : Type} {m Kd Np n : ℕ} (hK : Kd ≤ Np) (hKd : 0 < Kd) (p : Params)
    (ws : Executions (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (x : Paths (Paths C m × (Equiv.Perm (Fin Np) × Ξ)) n) (w : Fin n) :
    (checkedRun hK p ws opened rel right).Credited x w ↔ ws.AcceptedUpTo p x w :=
  ⟨fun h w' hw => h w' hw ⟨0, hKd⟩, fun h w' hw _ => h w' hw⟩

/-- **The run's accepted prefix is the check's.** -/
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

/-- **The run's credited live words are the check's accepted prefix's.** -/
theorem creditedLive_eq {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {Kd Np n : ℕ} (hK : Kd ≤ Np)
    (hKd : 0 < Kd) (p : Params) (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    (Wp : WindowPlay C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (x : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n) :
    creditedLive Sc Cd Wp (checkedRun hK p ws opened rel right) x = checkedLive Sc Cd Wp p ws x := by
  classical
  unfold creditedLive checkedLive
  exact Finset.sum_congr rfl fun w _ => if_congr (credited_iff hK hKd p ws opened rel right x w) rfl rfl

/-- **Joint soundness at the check's run from the service's**: in each window, an accepted window with a drawn tile
whose opening fails its relation is one the check accepts with a published unit (`DrawIsCoin`: the drawn tile's unit is
among `Public.drawn`) whose opening fails its relation, which `ServiceSound` bounds at `δ_ks`. -/
theorem jointSound_checked {ι C Ξ V : Type} [Fintype C] [Fintype Ξ] (Cd : Declared ι) {Kd Np n : ℕ}
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np) (p : Params) (ws : Executions (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n)
    (opened : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n → Fin n → Fin Np → (ℕ → V) → Prop)
    {δks : ℝ} (hD : DrawIsCoin hK p ws) (hS : ServiceSound p ws opened rel δks) :
    (checkedLayer Cd hr hK p ws opened rel right).JointSound δks := by
  intro w
  refine (Pouw.RelativeSecurity.prob_mono fun x hx => ?_).trans (hS w)
  obtain ⟨i, hacc, hrel⟩ := hx
  obtain ⟨pub, prof, hc⟩ := hacc
  exact ⟨pub, prof, hc, drawOf hK (coinAt x w).2 i, ((hD x w pub prof hc).2.2 _).2 ⟨i, rfl⟩, hrel⟩

/-- **`CheckedWorkPeriod`**: `CircuitWorkPeriod` at `checkedRun`, its accepted prefix and live words the check's. -/
theorem checkedWorkPeriod : CheckedWorkPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn hN hr hK hKd prm ws opened rel right Wp Pl folded
    ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved _hReg hDraw hServ hPins
    hMu hRB hGs hUF
  have h := Accounting.SecurityProofs.ComposePeriod.CircuitWorkPeriod Sc Cd hn hN hr hK Wp Pl
    (checkedRun hK prm ws opened rel right) folded ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG
    hDD hPre hMoved (jointSound_checked Cd hr hK prm ws opened rel right hDraw hServ) hPins hMu hRB hGs hUF
  simp only [prefixLen_eq hK hKd prm ws opened rel right,
    creditedLive_eq Sc Cd hK hKd prm ws opened rel right Wp] at h
  exact h

/-- **`CheckedCreditPeriod`**: `CircuitCreditPeriod` at `checkedRun`, its accepted prefix and live words the
check's. -/
theorem checkedCreditPeriod : CheckedCreditPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn hN hr hK hKd prm ws opened rel right Wp Pl folded
    ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved _hReg hDraw hServ hPins
    hMu hRB hGs hUF
  have h := Accounting.SecurityProofs.ComposePeriod.CircuitCreditPeriod Sc Cd hn hN hr hK Wp Pl
    (checkedRun hK prm ws opened rel right) folded ρ p p0 a b wmax Λ δs δbind δks hS hρ hρN hwmax hlk ha hb hΛ hSR hG
    hDD hPre hMoved (jointSound_checked Cd hr hK prm ws opened rel right hDraw hServ) hPins hMu hRB hGs hUF
  simp only [prefixLen_eq hK hKd prm ws opened rel right,
    creditedLive_eq Sc Cd hK hKd prm ws opened rel right Wp] at h
  exact h

/-- `SplitRateClosed` at H1's numbers, `K_d = 24`, `ρ = 8/9`, `w_max = 1`, `p = 7/8`, `p0 = ½`, `a = b = 1`:
`Λ = max(67/8, 67/32) = 67/8`. -/
theorem splitRate_h1 : SplitRate 24 (8 / 9) (7 / 8) (1 / 2) 1 1 1 (67 / 8) := by
  have h := Accounting.SecurityProofs.Compose.Rates.splitRateClosed 24 (8 / 9) (7 / 8) (1 / 2) 1 1 1 (by norm_num)
    (by norm_num) zero_le_one (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) one_pos one_pos
  convert h using 1
  rw [max_eq_left] <;> norm_num

/-- **H1's unit relation pins its tile's rightness**: the staged statement at the window's salt pins the flat tile
there (`StagedPins`), and the salt is the window's round-0 coin (`SaltIsLiveCoin`). -/
theorem pins_h1 {C B V : Type} {m n Np : ℕ} (hm : 0 < m) {salt : Paths (Paths C m × B) n → Fin n → ByteArray}
    {enc : C → ByteArray} {staged flat : ByteArray → Fin Np → (ℕ → V) → Prop} (hs : SaltIsLiveCoin hm salt enc)
    (hp : StagedPins staged flat) : ∀ x w j v, stagedAt salt staged x w j v → flatAtCoin hm enc flat x w j v := by
  intro x w j v h
  have h' := hp (salt x w) j v h
  rw [hs x w] at h'
  exact h'

/-- **`H1CheckedCreditPeriod`**: `CheckedCreditPeriod` at H1's numbers (`splitRate_h1`), its `Pins` from `pins_h1`. -/
theorem h1CheckedCreditPeriod : H1CheckedCreditPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n hn hN hr _hM hK prm ws opened salt enc staged flat Wp Pl folded
    δs δbind δks hS hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hStaged hMu hRB hGs hUF
  exact checkedCreditPeriod Sc Cd hn hN hr hK (by norm_num) prm ws opened (stagedAt salt staged)
    (flatAtCoin hr enc flat) Wp Pl folded (8 / 9) (7 / 8) (1 / 2) 1 1 1 (67 / 8) δs δbind δks hS (by norm_num) hρN
    hwmax hlk one_pos one_pos (by norm_num) splitRate_h1 hG hDD hPre hMoved hReg hDraw hServ (pins_h1 hr hSalt hStaged)
    hMu hRB hGs hUF

end Properties.Pouw.Period.SecurityProofs
