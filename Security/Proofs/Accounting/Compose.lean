import Models.Accounting.Guarantees.Compose

/-!
# PoUW's credit on an arbitrary declared circuit: proofs (open)

Every proof here is open (note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2 gives the order), and this
file doesn't land until none is a `sorry`:

1. `ServiceCovers`: the service's deficiency at `a = b = 1` is the tiles' weight off `paid`, a tile's weight times
   `N_w` is its words (both `0` at `N_w = 0`), and a tile that isn't good isn't paid (`PaidGood`).
2. `SplitRateClosed`: the folded branch at `W_0 = 0` (`ν = expm1(W_F·a/Λ)/((1 − p)·W_F)`, the exponent concave in
   `W_F` so its endpoints suffice), the whole branch `e^{Ky}(1 − y)^K ≤ 1` at `y = ρ·W_0`, and `L_0 > 0` by convexity
   in `L_0` with `ν₀ = Kρe^K`.
3. `SplitRateMerged`: one class at cost `b` and luck `q`, the folded branch of step 2, then monotonicity in `a ≤ b`,
   `p, p0 ≤ q`.
4. `ServiceBudget`: given the stages' coins, a uniform `K`-subset of `N` misses the wrong tiles with probability at
   most `(1 − ρ·(wrong share))^K`, which factors over the stages; `SplitRate` per stage at its share `≤ w_max`, its
   luck in expectation from `LuckAt` (`E[L_F] ≤ p·E[W_F]`, `E[L_0] ≤ p0·E[W_0]`), then backward induction over the
   stages, as `Accounting.SecurityProofs.TS.StageBudget`.
5. `CircuitWork`: `Accounting.SecurityProofs.TS.DrawnRightOff` at `S = 1` gives one event of probability at most
   `δ_bind + δ_ks` off which every accepted drawn tile is right at its committed slots, so `Of` gives the share game's
   pass (RT-I8's `compose_sampled` shape); `UnitFixed` with step 1 gives the fixed words `(1 − (d_F + d₀))·N_w`, the
   tiles' words summing to `N_w`, and `UnitFolded` the folded ones `(1 − d₀)·N_w`; `ServiceBudget` on `serviceGame` at
   `a`, `b` (the same coins and pass) and `budget_tail` at `t = ln(1/δ_s)` bound `a·d_F + b·d₀ ≥ Λ·ln(1/δ_s)` by
   `δ_s`; `WorkAt` holds at the run's shares `d_F`, `d₀`. `StageCuts` is what makes `δ_bind` the fork form's at every
   stage coin (`DrawnBindingOfFork`), and the proof doesn't read it.
6. `CreditOfWork`: `SheetDIdealFloor`'s reduction of (I1)–(I5) to A1–A7 on the one run, then `floor_of_price`'s
   identity with FOLD-OPT at the folded words (`FoldOptFolded`), which prices the deficit at `a·d_F + b·d₀ < r`, and
   `T_cred > 0` (`Model.Wf`) to clear the division.
7. `CircuitCredit`: step 5 at `a = aOf`, `b = bOf` (`≥ 0` by `stars_nonneg` and `Model.Wf`), and step 6 on each run
   off its event.
8. `CircuitCreditFS`: `Pouw.Guarantees.FiatShamir.FiatShamirFloor` at `m` (proofs', open: to be
   `Pouw.SecurityProofs.FiatShamir.FiatShamirFloor`), then `μ^m ≤ μ^{h·w}` (`μ ≤ 1`, `h·w ≤ m`) and
   `ServiceFloorFS`'s monotonicity in its bound.
-/

namespace Accounting.SecurityProofs.Compose

theorem ServiceCovers : Accounting.Guarantees.Compose.ServiceCovers := by
  sorry

theorem SplitRateClosed : Accounting.Guarantees.Compose.SplitRateClosed := by
  sorry

theorem SplitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  sorry

theorem ServiceBudget : Accounting.Guarantees.Compose.ServiceBudget := by
  sorry

theorem CircuitWork : Accounting.Guarantees.Compose.CircuitWork := by
  sorry

theorem CreditOfWork : Accounting.Guarantees.Compose.CreditOfWork := by
  sorry

theorem CircuitCredit : Accounting.Guarantees.Compose.CircuitCredit := by
  sorry

theorem CircuitCreditFS : Accounting.Guarantees.Compose.CircuitCreditFS := by
  sorry

end Accounting.SecurityProofs.Compose
