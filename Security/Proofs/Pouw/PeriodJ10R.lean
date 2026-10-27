import Properties.Pouw.PeriodJ10R
import Proofs.Accounting.ComposePeriod
import Proofs.Accounting.Compose.SplitRateClosed
import Proofs.Pouw.RelativeSecurity

/-!
# Proof of `H1J10RCheckedCreditPeriod`

`Accounting.SecurityProofs.ComposePeriod.CircuitCreditPeriod` at `tileRun`, as `Proofs.Pouw.Period` proves
`CheckedCreditPeriod`: a window is credited exactly when it and every earlier one are accepted by `Credit.window`
(`credited_iff`, at `k s_T > 0`), so the run's accepted prefix and live words are `Credit.window`'s (`prefixLen_eq`,
`creditedLive_eq`), and `JointSound` at `tileLayer` follows from `TileDrawIsCoin` and `TileServiceSound` by
monotonicity (`jointSound_tile`). The rate is `SplitRateClosed`'s at H1's lucks (`rate_j10r`: its first term,
`8(1 + Kρ)/(ρK)`), and `Pins` comes from the window's salt being its live coin (`pins_j10r`): the relation is
`j10r_reference`'s at the unit's place, one description.
-/

namespace Properties.Pouw.PeriodJ10R.SecurityProofs

open Properties.Pouw.PeriodJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **A window is credited under `Credit.window` exactly when it and every earlier one are accepted**, at
`k s_T > 0`. -/
theorem credited_iff {C Ξ V : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (hKd : 0 < k sT)
    (ws : CreditWindows (Coin C Ξ m σ sT) n) (opened : Paths (Coin C Ξ m σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ m σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (x : Paths (Coin C Ξ m σ sT) n) (w : Fin n) :
    (tileRun σ k hk sT unitOf ws opened rel right).Credited x w ↔ ws.AcceptedUpTo x w :=
  ⟨fun h w' hw => h w' hw ⟨0, hKd⟩, fun h w' hw _ => h w' hw⟩

/-- **The run's accepted prefix is `Credit.window`'s.** -/
theorem prefixLen_eq {C Ξ V : Type} {m nB mB n : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (hKd : 0 < k sT)
    (ws : CreditWindows (Coin C Ξ m σ sT) n) (opened : Paths (Coin C Ξ m σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ m σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (x : Paths (Coin C Ξ m σ sT) n) :
    (tileRun σ k hk sT unitOf ws opened rel right).prefixLen x = ws.prefixLen x := by
  classical
  unfold PeriodRun.prefixLen CreditWindows.prefixLen
  congr 1
  exact Finset.filter_congr fun w _ => credited_iff σ k hk sT unitOf hKd ws opened rel right x w

/-- **The run's credited live words are `Credit.window`'s accepted prefix's.** -/
theorem creditedLive_eq {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {nB mB n : ℕ} (σ : Fin nB → Fin mB)
    (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (hKd : 0 < k sT)
    (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (x : Paths (Coin C Ξ Cd.rounds σ sT) n) :
    creditedLive Sc Cd Wp (tileRun σ k hk sT unitOf ws opened rel right) x = creditLive Sc Cd Wp ws x := by
  classical
  unfold creditedLive creditLive
  exact Finset.sum_congr rfl fun w _ =>
    if_congr (credited_iff σ k hk sT unitOf hKd ws opened rel right x w) rfl rfl

/-- **Joint soundness at `Credit.window`'s run from the service's**: in each window, an accepted window with a drawn
tile whose opening fails its relation is one `Credit.window` credits with that tile's window unit among the units its
draw demanded (`TileDrawIsCoin`), whose opening fails its relation, which `TileServiceSound` bounds at `δ_ks`. -/
theorem jointSound_tile {ι C Ξ V : Type} [Fintype C] [Fintype Ξ] (Cd : Declared ι) {nB mB n : ℕ}
    (hr : 0 < Cd.rounds) (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
    (unitOf : Fin nB → ℕ) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop) {δks : ℝ}
    (hD : TileDrawIsCoin σ k hk sT unitOf ws) (hS : TileServiceSound σ sT unitOf ws opened rel δks) :
    (tileLayer Cd hr σ k hk sT unitOf ws opened rel right).JointSound δks := by
  intro w
  refine (Pouw.RelativeSecurity.prob_mono fun x hx => ?_).trans (hS w)
  obtain ⟨i, hacc, hrel⟩ := hx
  obtain ⟨d, hd, hmem⟩ := hD x w hacc
  exact ⟨hacc, d, hd, drawOf (hk sT) (coinAt x w).2 i, (hmem _).2 ⟨i, rfl⟩, hrel⟩

/-- `j10rRate` is positive at `K ≥ 1` and `ρ > 0`. -/
theorem j10rRate_pos {K : ℕ} {ρ : ℝ} (hK : 1 ≤ K) (hρ : 0 < ρ) : 0 < j10rRate K ρ := by
  have hK0 : (0 : ℝ) < K := by exact_mod_cast hK
  unfold j10rRate
  have := mul_pos hK0 hρ
  exact div_pos (by nlinarith) (mul_pos hρ hK0)

/-- `SplitRateClosed` at H1's lucks, `w_max = 1`, `a = b = 1`, any `K ≥ 1` and `0 < ρ ≤ 1`: its first term,
`j10rRate K ρ = 8(1 + Kρ)/(ρK)`, is the larger. -/
theorem rate_j10r {K : ℕ} {ρ : ℝ} (hK : 1 ≤ K) (hρ : 0 < ρ) (hρ1 : ρ ≤ 1) :
    SplitRate K ρ (7 / 8) (1 / 2) 1 1 1 (j10rRate K ρ) := by
  have h := Accounting.SecurityProofs.Compose.Rates.splitRateClosed K ρ (7 / 8) (1 / 2) 1 1 1 hK hρ zero_le_one
    (by simpa using hρ1) (by norm_num) (by norm_num) (by norm_num) (by norm_num) one_pos one_pos
  have hK0 : (0 : ℝ) < K := by exact_mod_cast hK
  have hKρ : (0 : ℝ) < K * ρ := mul_pos hK0 hρ
  have hq : 0 ≤ (1 + K * ρ) / (ρ * K) := div_nonneg (by linarith) (mul_pos hρ hK0).le
  have e1 : 1 * (1 + (K : ℝ) * ρ * 1) / (ρ * K * (1 - 7 / 8)) = 8 * ((1 + K * ρ) / (ρ * K)) := by
    field_simp
    ring
  have e2 : 1 * (1 + (K : ℝ) * ρ * 1) / (ρ * K * (1 - 1 / 2)) = 2 * ((1 + K * ρ) / (ρ * K)) := by
    field_simp
    ring
  convert h using 1
  rw [e1, e2, max_eq_left (by linarith)]
  unfold j10rRate
  ring

/-- **J10-R's unit relation pins its tile's rightness**: the relation, `j10r_reference`'s at the unit's place under the
window's salt, is its rightness, the same at the window's round-0 coin, because the salt is that coin
(`SaltIsLiveCoin`). -/
theorem pins_j10r {C B V : Type} {m n Np : ℕ} (hm : 0 < m) {salt : Paths (Paths C m × B) n → Fin n → ByteArray}
    {enc : C → ByteArray} {place : Paths (Paths C m × B) n → Fin n → Fin Np → Place}
    {tile : Noise → List (List ℕ) → Fin Np → (ℕ → V) → Prop}
    (hs : Properties.Pouw.Period.SaltIsLiveCoin hm salt enc) :
    ∀ x w j v, pinnedAt salt place tile x w j v → j10rAtCoin hm enc place tile x w j v := by
  intro x w j v h
  unfold pinnedAt at h
  unfold j10rAtCoin
  rw [← hs x w]
  exact h

/-- **`H1J10RCheckedCreditPeriod`**: `CircuitCreditPeriod` at `tileRun`, its accepted prefix and live words
`Credit.window`'s, at `rate_j10r`'s rate and `pins_j10r`'s pins. -/
theorem h1J10RCheckedCreditPeriod : H1J10RCheckedCreditPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ nB mB σ k hk sT unitOf Sc Cd n hn hN hr _hM hKd ws opened enc place tile
    Wp Pl folded ρ δs δbind δks hS hρ hρ1 hρN hwmax hlk hG hDD hPre hMoved _hReg hDraw hServ hSalt hLucks hUF
  obtain ⟨hMu, hRB, hGs⟩ := hLucks
  have hK0 : 0 < k sT := hKd
  have h := Accounting.SecurityProofs.ComposePeriod.CircuitCreditPeriod Sc Cd hn hN hr (hk sT) Wp Pl
    (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile) (j10rAtCoin hr enc place tile))
    folded ρ (7 / 8) (1 / 2) 1 1 1 (j10rRate (k sT) ρ) δs δbind δks hS hρ hρN hwmax hlk one_pos one_pos
    (j10rRate_pos hKd hρ) (rate_j10r hKd hρ hρ1) hG hDD hPre hMoved
    (jointSound_tile Cd hr σ k hk sT unitOf ws opened _ _ hDraw hServ) (pins_j10r hr hSalt) hMu hRB hGs hUF
  simp only [prefixLen_eq σ k hk sT unitOf hK0 ws opened, creditedLive_eq Sc Cd σ k hk sT unitOf hK0 ws opened _ _ Wp]
    at h
  exact h

private theorem find_getD_nDigitsFrom (ks : ℕ) :
    ∀ f m, (((List.range f).map (· + m)).find?
        (fun m => 128 * ks ≤ 128 ^ m && 127 * ks / 128 ^ (m - 1) ≤ 127)).getD (m + f) =
      J10R.nDigitsFrom ks f m := by
  intro f
  induction f with
  | zero => intro m; simp [J10R.nDigitsFrom]
  | succ f ih =>
    intro m
    have hl : (List.range (f + 1)).map (· + m) = m :: (List.range f).map (· + (m + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      · simp
      · apply List.map_congr_left; intro i _; simp only [Function.comp]; omega
    rw [hl, List.find?_cons, J10R.nDigitsFrom]
    by_cases h : 128 * ks ≤ 128 ^ m ∧ 127 * ks / 128 ^ (m - 1) ≤ 127
    · have hb : (128 * ks ≤ 128 ^ m && 127 * ks / 128 ^ (m - 1) ≤ 127) = true := by simpa using h
      simp only [hb, if_pos h, Option.getD_some]
    · have hb : (128 * ks ≤ 128 ^ m && 127 * ks / 128 ^ (m - 1) ≤ 127) = false := by simpa using h
      simp only [hb, if_neg h]
      rw [← ih (m + 1), show m + (f + 1) = m + 1 + f by omega]

/-- The verifier's digit count is H1's (`J10R.nDigits`): both search `m = 1, …, KS + 1` for the same least `m`, and
`m = KS + 1` always qualifies, so their different fallbacks are never read. -/
theorem nDigits_eq (k : ℕ) : PouwCheck.Credit.nDigits k = J10R.nDigits k := by
  unfold PouwCheck.Credit.nDigits J10R.nDigits
  dsimp only
  rw [← find_getD_nDigitsFrom (k / 32) (k / 32 + 1) 1]
  cases hf : ((List.range (k / 32 + 1)).map (· + 1)).find?
      (fun m => 128 * (k / 32) ≤ 128 ^ m && 127 * (k / 32) / 128 ^ (m - 1) ≤ 127) with
  | some v => rfl
  | none =>
    exfalso
    have hk : k / 32 ≤ 128 ^ (k / 32) := (Nat.lt_pow_self (by norm_num)).le
    have hmem : k / 32 + 1 ∈ (List.range (k / 32 + 1)).map (· + 1) :=
      List.mem_map.2 ⟨k / 32, List.mem_range.2 (Nat.lt_succ_self _), rfl⟩
    refine List.find?_eq_none.1 hf _ hmem ?_
    simp only [Nat.add_sub_cancel, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨?_, Nat.div_le_of_le_mul ?_⟩
    · rw [pow_succ, Nat.mul_comm (128 ^ (k / 32))]
      exact Nat.mul_le_mul le_rfl hk
    · rw [Nat.mul_comm 127]
      exact Nat.mul_le_mul hk le_rfl

/-- **The verifier's blocks are J10-R's** (witness W1): `Credit.j10rBlocks`, which the statement's `Place.blocks`
reads, is `J10R.unitBlocks` at every slot, contraction, tile and place. -/
theorem j10rBlocks_eq (M k R C I J : ℕ) :
    PouwCheck.Credit.j10rBlocks M k (R, C) I J = J10R.unitBlocks M k R C I J := by
  simp only [PouwCheck.Credit.j10rBlocks, J10R.unitBlocks, J10R.Layout.row, J10R.Layout.col, J10R.rb, J10R.cb,
    J10R.rowBits, J10R.colBits, nDigits_eq]

end Properties.Pouw.PeriodJ10R.SecurityProofs
