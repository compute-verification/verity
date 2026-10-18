import Models.Accounting.Guarantees.Compose

/-!
# PoUW's credit on an arbitrary declared circuit: proofs (open)

Every proof here is open (note:compute-accounting/20261018T0618Z-draft-theorem-compose-r2 gives the order), and this
file doesn't land until none is a `sorry`:

1. `DrawnJointOfSoundAt`: the union over the `s` draws of `SoundAt`'s per-draw bound.
2. `SplitRateClosed`: the folded branch at `W_0 = 0` (`ν = expm1(W_F·a/Λ)/((1 − p)·W_F)`, the exponent concave in
   `W_F` so its endpoints suffice), the whole branch `e^{Ky}(1 − y)^K ≤ 1` at `y = ρ·W_0`, and `L_0 > 0` by convexity
   in `L_0` with `ν₀ = Kρe^K`.
3. `SplitRateMerged`: one class at cost `b` and luck `q`, the folded branch of step 2, then monotonicity in `a ≤ b`,
   `p, p0 ≤ q`.
4. `ServiceBudget`: given the stages' coins, a uniform `K`-subset of `N` misses the wrong tiles with probability at
   most `(1 − ρ·(wrong share))^K`, which factors over the stages; `SplitRate` per stage at its share `≤ w_max`, its
   luck in expectation from `LuckAt` (`E[L_F] ≤ p·E[W_F]`, `E[L_0] ≤ p0·E[W_0]`), then backward induction over the
   stages, as `Accounting.SecurityProofs.TS.StageBudget`.
5. `CircuitCredit`: off the (I8) event `∃ i, Moved ∨ Unsound` (at most `δ_bind + δ_ks`), `ProofsSoundOfDrawn`'s
   reduction gives the share game's pass, `RightFixedDrawn` the fixed words and `RightFoldedDrawn` the folded ones;
   `ServiceBudget` on `serviceGame` at `aOf`, `bOf` (the same coins and pass) and `budget_tail` at `t = ln(1/δ_s)`
   bound `a·F + b·W₀ ≥ Λ·ln(1/δ_s)` by `δ_s`; `floor_of_price`'s identity with FOLD-OPT at the folded words
   (`FoldOptFolded`) prices the deficit at `a·F + b·W₀`, as `SheetDIdealFloor` reduces (I1)–(I5) to A1–A7; and
   `T_cred > 0` clears the division.
6. `CircuitCreditFS`: `Pouw.Guarantees.FiatShamir.FiatShamirFloor` at `m` (proofs', open: to be
   `Pouw.SecurityProofs.FiatShamir.FiatShamirFloor`), then `μ^m ≤ μ^{h·w}` (`μ ≤ 1`, `h·w ≤ m`) and
   `ServiceFloorFS`'s monotonicity in its bound.
-/

namespace Accounting.SecurityProofs.Compose

theorem DrawnJointOfSoundAt : Accounting.Guarantees.Compose.DrawnJointOfSoundAt := by
  sorry

theorem SplitRateClosed : Accounting.Guarantees.Compose.SplitRateClosed := by
  sorry

theorem SplitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  sorry

theorem ServiceBudget : Accounting.Guarantees.Compose.ServiceBudget := by
  sorry

theorem CircuitCredit : Accounting.Guarantees.Compose.CircuitCredit := by
  sorry

theorem CircuitCreditFS : Accounting.Guarantees.Compose.CircuitCreditFS := by
  sorry

end Accounting.SecurityProofs.Compose
