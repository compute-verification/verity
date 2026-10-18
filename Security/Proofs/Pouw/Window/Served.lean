import Definitions.Pouw.Window

/-!
# PoUW's reading of the window: its bound plus the tie to what's served

`WindowEps` bounds the window accepting while its committed transcript's wrong units are in the family `𝓑`.
PoUW credits the served computation, so it reads one more term, the window accepting while its committed transcript
isn't the served one (`ServedTie`, `δ_in`). `window_served_le` adds the two: the window accepts while its committed
transcript is not anchored, served and with its wrong units outside `𝓑` with probability at most `WindowEps`'s bound
plus `δ_in`.
-/

namespace Pouw.Window

open scoped ENNReal

variable {α : Type}

theorem prob_mono {p q : α → Prop} (h : ∀ a, p a → q a) : ∀ (g : Game α) (s : Strategy g), prob p g s ≤ prob q g s
  | .ret a, _ => by
    classical
    simp only [prob]
    split_ifs with hp hq
    · exact le_rfl
    · exact absurd (h a hp) hq
    · exact zero_le
    · exact le_rfl
  | .send _ k, s => prob_mono h (k s.1) s.2
  | @Game.coin _ C iF _ k, s => by
    change ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (k c) (s c) ≤
      ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob q (k c) (s c)
    exact Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono h (k c) (s c)) _

theorem prob_or_le (p q : α → Prop) :
    ∀ (g : Game α) (s : Strategy g), prob (fun a => p a ∨ q a) g s ≤ prob p g s + prob q g s
  | .ret a, _ => by
    classical
    simp only [prob]
    split_ifs <;> simp_all
  | .send _ k, s => prob_or_le p q (k s.1) s.2
  | @Game.coin _ C iF _ k, s => by
    change ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob (fun a => p a ∨ q a) (k c) (s c) ≤
      ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (k c) (s c) +
        ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob q (k c) (s c)
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun c _ => by
      rw [← mul_add]
      exact mul_le_mul_right (prob_or_le p q (k c) (s c)) _

/-- **PoUW's reading of the window**: if the window accepts with its committed transcript's wrong units in `𝓑` with
probability at most `b` (`WindowEps`'s bound) and the tie to what's served holds (`ServedTie`), then it accepts while
its committed transcript is not anchored, served and with its wrong units outside `𝓑` with probability at most
`b + δin(R, τ)`. -/
theorem window_served_le {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ} {Lw : Reg → Law (WUnit n)}
    {sess : (c : Fin C) → Finset (Fin (n c)) → Reg → Game Bool} {U : UnitStmts Tr n}
    {committed : (R : Reg) → Cont Lw (windowSession sess) R → Tr} {anchored served : Tr → Prop}
    {δin : (R : Reg) → Cont Lw (windowSession sess) R → ℝ≥0∞} {𝓑 : Finset (WUnit n) → Prop}
    (hTie : ServedTie Lw sess committed anchored served δin) (σ : Strategy (auditReg Lw (windowSession sess)))
    {b : ℝ≥0∞}
    (hW : prob (fun o => o.1 = true ∧ 𝓑 (U.wrong (committed (reg σ) (cont σ)))) (auditReg Lw (windowSession sess)) σ ≤
      b) :
    prob (fun o => o.1 = true ∧ ¬ (anchored (committed (reg σ) (cont σ)) ∧ served (committed (reg σ) (cont σ)) ∧
        ¬ 𝓑 (U.wrong (committed (reg σ) (cont σ))))) (auditReg Lw (windowSession sess)) σ ≤
      b + δin (reg σ) (cont σ) := by
  refine le_trans (prob_mono (fun o h => ?_) _ σ) (le_trans (prob_or_le _ _ _ σ) (add_le_add hW (hTie σ)))
  by_cases hB : 𝓑 (U.wrong (committed (reg σ) (cont σ)))
  · exact Or.inl ⟨h.1, hB⟩
  · exact Or.inr ⟨h.1, fun hAS => h.2 ⟨hAS.1, hAS.2, hB⟩⟩

end Pouw.Window
