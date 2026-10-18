import Models.Accounting.Guarantees.Compose

/-!
# PoUW's credit on an arbitrary declared circuit: proofs (open)

Every proof here is open (note:compute-accounting/20261018T1230Z-draft-theorem-compose-r3 gives the order), and this
file doesn't land until none is a `sorry`:

1. `SplitRateClosed`: the folded branch at `W_0 = 0` (`ν = expm1(W_F·a/Λ)/((1 − p)·W_F)`, the exponent concave in
   `W_F` so its endpoints suffice) and the whole branch at `W_F = 0` the same at `b`, `p0` (`ν₀`), then the mixed
   point by convexity of the left side in `(W_F, W_0)` at the larger `Λ`, and `L_F, L_0 > 0` by convexity in them.
2. `SplitRateMerged`: one class at cost `b` and luck `q`, the folded branch of step 1, then monotonicity in `a ≤ b`,
   `p, p0 ≤ q`.
3. `ServiceBudget`: given the stages' coins, a uniform `K`-subset of `N` misses the tiles whose checked values aren't
   honest with probability at most `(1 − ρ·(their share))^K`, which factors over the stages; `SplitRate` per stage at
   its share `≤ w_max`, its luck in expectation from `LuckAt` (`E[L_F] ≤ p·E[W_F]`, `E[L_0] ≤ p0·E[W_0]`), then
   backward induction over the stages, as `Accounting.SecurityProofs.TS.StageBudget`.
4. `ServiceCovers`: `Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone` at the tiles' words (`≥ 0`). The
   game's deficiency is the not-paid tiles' `tileCone` (the three modes, at `1` and `1`), so it is enough that a
   tile's layer cone is at most `N_w` times its `tileCone`: a layer edge `DrawnLayer.Late u v` at slot `k` has its
   position a gate of `u`'s unit (`StageSlots`), and either one of `v`'s own (`u = v`) or read by a gate `j` of `v`'s,
   whose round is `v`'s (`Declared.Wf`, `StageCoins`) and before the position's cut (`StageCuts`), so `Declared.Late`
   and a `Declared.UnitLate` edge; along `ReflTransGen` the layer's cone lies in `Declared.Reaches`, and a tile's words
   are `N_w` times its weight (`0` when `N_w = 0`: a tile's words are among the circuit's).
5. `CircuitWork`: the luck first. At each stage, each credited tile's `tileGame … .folded` is `Pinned` (its check is
   `same`), so `Accounting.SecurityProofs.TS.Headline.operandLuck_of_pointMass` (Lemma B at fibre 1) on `OperandHMu` and
   `LuckOfReceipt` with `ReceiptLuckAt` give `Pr[folded ∧ same] ≤ (μ + (1 − μ)·r_B)·Pr[folded]` (`r_B ≤ 1` by
   `Scheme.LuckProved`), and `GuessLuckAt` on `tileGame … .whole` gives `Pr[whole ∧ same] ≤ guess·Pr[whole]`; a tile of
   no credited kind is held to `1`; so `SplitStages.LuckAt (tileLuck …) (tileLuck0 …)`. Then
   `Accounting.SecurityProofs.TS.DrawnRightOff` at `S = 1` gives one event of probability at most `δ_bind + δ_ks` off
   which every accepted drawn tile is right at its committed slots, so `Of` gives the share game's pass (RT-I8's
   `compose_sampled` shape); `ServiceBudget` on `serviceGame` at `a`, `b` and `shareCone` (the same coins and pass)
   and `budget_tail` at `t = ln(1/δ_s)` bound `a·S_F + b·S_0 ≥ Λ·ln(1/δ_s)` by `δ_s`, `S_F`, `S_0` the folded and
   whole shares at `shareCone`. Off both, `UnitFixed` (its `WriterReads` given) at `tileMode` and step 4's `Covers`
   give `WorkAt` at `d_F`, `d₀` the folded and whole shares at `tileCone`: the good tiles' words are at least
   `(1 − (d_F + d₀))·N_w` (`Σ_t tileWords = N_w`, by `Declared.Wf` and `units ≤ N_p`), `d − S ≤ D₀` mode by mode
   (`tileCone` and `shareCone` differ only on tiles of no words), so `a·d_F + b·d₀ < Λ·ln(1/δ_s) + max(a, b)·D₀`, and
   the fold clause's words not skipped whole are at least `(1 − d₀)·N_w` (a tile's cone holds itself,
   `ReflTransGen.refl`).
6. `CreditOfWork`: `SheetDIdealFloor`'s reduction of (I1)–(I5) to A1–A7 on the one run, then `floor_of_price`'s
   identity with FOLD-OPT at the folded words (`FoldOptFolded`), which prices the deficit at `a·d_F + b·d₀ < r`, and
   `T_cred > 0` (`Model.Wf`) to clear the division.
7. `CircuitCredit`: step 5 at `a = aOf`, `b = bOf` (`≥ 0` by `stars_nonneg` and `Model.Wf`), and step 6 on each run
   off its event at `r = rOf`.
8. `CircuitCreditFS`: `Pouw.Guarantees.FiatShamir.FiatShamirFloor` at `m` (proofs', open: to be
   `Pouw.SecurityProofs.FiatShamir.FiatShamirFloor`), then `μ^m ≤ μ^{h·w}` (`μ ≤ 1`, `h·w ≤ m`) and
   `ServiceFloorFS`'s monotonicity in its bound.
-/

namespace Accounting.SecurityProofs.Compose

theorem SplitRateClosed : Accounting.Guarantees.Compose.SplitRateClosed := by
  sorry

theorem SplitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  sorry

theorem ServiceBudget : Accounting.Guarantees.Compose.ServiceBudget := by
  sorry

theorem ServiceCovers : Accounting.Guarantees.Compose.ServiceCovers := by
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
