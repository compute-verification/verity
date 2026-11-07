module

public import Definitions.Pouw.Window
public import Proofs.Core.Game.Prob

/-!
# PoUW's reading of the window: its bound plus the tie to what's served, and its collision term at counted queries

`WindowDelta` bounds the window accepting while its committed transcript's incorrect proof blocks are in the family `𝓑`.
PoUW credits the served computation, so it reads one more term, the window accepting while its committed transcript
isn't the served one (`ServedTie`, `δ_in`). `window_served_le` adds the two: the window accepts while its committed
transcript is not anchored, served and with its incorrect proof blocks outside `𝓑` with probability at most `WindowDelta`'s bound
plus `δ_in`.

`WindowDelta`'s bound holds at each oracle with the collision term `coll` at that oracle's play. `avg_counted_le` reads it
on average over the oracle: at plays whose sessions make at most `q` counted queries, the collision term becomes the
counted bound `f q` (`CollCounted`).
-/

@[expose] public section

namespace Pouw.Window

open scoped ENNReal

/-- **PoUW's reading of the window**: if the window accepts with its committed transcript's incorrect proof blocks in `𝓑` with
probability at most `b` (`WindowDelta`'s bound) and the tie to what's served holds (`ServedTie`, against the served pass
`P`), then it accepts while its committed transcript is not anchored, served and with its incorrect proof blocks outside `𝓑` with
probability at most `b + δin(R, τ)`. -/
theorem window_served_le {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ} {Lw : Reg → Law (WUnit n)}
    {sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool} {U : UnitStmts Tr n} {P : ServedPass Tr n U.Pos}
    {committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr}
    {δin : (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞} {𝓑 : Finset (WUnit n) → Prop}
    (hTie : ServedTie Lw sess committed U P δin) (σ : Strategy (auditReg Lw (windowSession sess)))
    {b : ℝ≥0∞}
    (hW : prob (fun o => o.1 = true ∧ 𝓑 (U.wrong (committed (reg σ) (cont σ)))) (auditReg Lw (windowSession sess)) σ ≤
      b) :
    prob (fun o => o.1 = true ∧ ¬ (Anchored U P (committed (reg σ) (cont σ)) ∧
        Served U P (committed (reg σ) (cont σ)) ∧ ¬ 𝓑 (U.wrong (committed (reg σ) (cont σ)))))
      (auditReg Lw (windowSession sess)) σ ≤ b + δin (reg σ) (cont σ) := by
  refine le_trans (Verity.Game.prob_mono (fun o h => ?_) _ σ)
    (le_trans (Verity.Game.prob_or_le _ _ _ σ) (add_le_add hW (hTie σ)))
  by_cases hB : 𝓑 (U.wrong (committed (reg σ) (cont σ)))
  · exact Or.inl ⟨h.1, hB⟩
  · exact Or.inr ⟨h.1, fun hAS => h.2 ⟨hAS.1, hAS.2, hB⟩⟩

/-- **A per-oracle bound with the collision term, on average over the oracle**: if at each oracle `H` a probability
`p H` is at most `b H` plus the collision term at the play `σ H` (`WindowDelta`'s bound, with `b` its other terms), and
the plays' sessions make at most `q` counted queries, then `avg p ≤ avg b + f q` under `CollCounted`. -/
theorem avg_counted_le {O Reg : Type} [Fintype O] {C : ℕ} {n : Fin C → ℕ} {Lw : O → Reg → Law (WUnit n)}
    {sess : O → (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool}
    {g : (H : O) → (c : Fin C) → (S : Finset (Fin (n c))) → (R : Reg) → Strategy (sess H c S R) → ℝ≥0∞}
    {qs : (H : O) → Strategy (auditReg (Lw H) (windowSession (sess H))) → ℕ} {f : ℕ → ℝ≥0∞}
    (hC : CollCounted Lw sess g qs f) {q : ℕ} (σ : (H : O) → Strategy (auditReg (Lw H) (windowSession (sess H))))
    (hq : ∀ H, qs H (σ H) ≤ q) {p b : O → ℝ≥0∞}
    (hW : ∀ H, p H ≤ b H + coll (Lw H) (sess H) (g H) (reg (σ H)) (cont (σ H))) :
    avg p ≤ avg b + f q := by
  calc avg p ≤ avg fun H => b H + coll (Lw H) (sess H) (g H) (reg (σ H)) (cont (σ H)) := by
        unfold avg; gcongr with H; exact hW H
    _ = avg b + avg fun H => coll (Lw H) (sess H) (g H) (reg (σ H)) (cont (σ H)) := by
        simp only [avg, mul_add, Finset.sum_add_distrib]
    _ ≤ avg b + f q := by gcongr; exact hC q σ hq

end Pouw.Window
