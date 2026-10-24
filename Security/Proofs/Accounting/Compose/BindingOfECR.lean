import Definitions.Accounting.Guarantees.Compose
import Proofs.Accounting.TS.Drawn

/-!
# A prover's `δ_bind` through `ecr/sha-512`: proof

`Accounting.Guarantees.Compose.BindingOfECR`. With `M = Pr[MovedIn w]` and `Φ` the fork form at `L`'s cuts,
`DrawnBindingOfFork` gives `M² ≤ |L|²·Φ`, and the premise `Φ ≤ Pr[F finds] + κ`. AM–GM (`(M − 2δ')² ≥ 0`) gives
`M ≤ δ' + M²/(4δ')` (`le_add_sq_div`). If `F` meets `ecr/sha-512`, `Pr[F finds] ≤ E[cost F]/2^256` is the claim itself,
and the real bound passes to `ENNReal` (`ofReal_add_le`, `ofReal_mul`). If it doesn't, the bound is `⊤` unless `L` is
empty, and then `M² ≤ 0`.
-/

namespace Accounting.SecurityProofs.Compose.Binding

/-- AM–GM: `M ≤ δ' + M²/(4δ')`. -/
theorem le_add_sq_div {M δ' : ℝ} (hδ : 0 < δ') : M ≤ δ' + M ^ 2 / (4 * δ') := by
  rw [← sub_nonneg]
  have : δ' + M ^ 2 / (4 * δ') - M = (M - 2 * δ') ^ 2 / (4 * δ') := by field_simp; ring
  rw [this]
  exact div_nonneg (sq_nonneg _) (by linarith)

/-- The bound from `M² ≤ γ·Φ` and `Φ ≤ P + κ`, with `P ≤ e` when `c` holds and `⊤` in `e`'s place when it doesn't. -/
theorem ofReal_le_of_sq_le {M Φ γ P e κ δ' : ℝ} (hδ : 0 < δ') (hγ : 0 ≤ γ) (hsq : M ^ 2 ≤ γ * Φ) (hΦ : Φ ≤ P + κ)
    (c : Prop) [Decidable c] (hc : c → P ≤ e) :
    ENNReal.ofReal M ≤ ENNReal.ofReal δ' +
      ENNReal.ofReal (γ / (4 * δ')) * ((if c then ENNReal.ofReal e else (⊤ : ENNReal)) + ENNReal.ofReal κ) := by
  by_cases h : c
  · rw [ite_eq_left h]
    have hM2 : M ^ 2 ≤ γ * (e + κ) :=
      hsq.trans (mul_le_mul_of_nonneg_left (hΦ.trans (by linarith [hc h])) hγ)
    have hreal : M ≤ δ' + γ / (4 * δ') * (e + κ) :=
      calc M ≤ δ' + M ^ 2 / (4 * δ') := le_add_sq_div hδ
        _ ≤ δ' + γ * (e + κ) / (4 * δ') := by
          have := div_le_div_of_nonneg_right hM2 (by linarith : (0 : ℝ) ≤ 4 * δ')
          linarith
        _ = δ' + γ / (4 * δ') * (e + κ) := by ring
    calc ENNReal.ofReal M ≤ ENNReal.ofReal (δ' + γ / (4 * δ') * (e + κ)) := ENNReal.ofReal_le_ofReal hreal
      _ ≤ ENNReal.ofReal δ' + ENNReal.ofReal (γ / (4 * δ') * (e + κ)) := ENNReal.ofReal_add_le
      _ = ENNReal.ofReal δ' + ENNReal.ofReal (γ / (4 * δ')) * ENNReal.ofReal (e + κ) := by
        rw [ENNReal.ofReal_mul (div_nonneg hγ (by linarith))]
      _ ≤ ENNReal.ofReal δ' + ENNReal.ofReal (γ / (4 * δ')) * (ENNReal.ofReal e + ENNReal.ofReal κ) := by
        gcongr
        exact ENNReal.ofReal_add_le
  · rw [ite_eq_right h, top_add]
    rcases hγ.eq_or_lt with h0 | hpos
    · have hM : M = 0 := by
        rw [← h0, zero_mul] at hsq
        exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp (le_antisymm hsq (sq_nonneg M))
      rw [hM, ENNReal.ofReal_zero]
      exact zero_le
    · rw [ENNReal.mul_top (ENNReal.ofReal_pos.mpr (div_pos hpos (by linarith))).ne', add_top]
      exact le_top

open Classical in
theorem bindingOfECR : Accounting.Guarantees.Compose.BindingOfECR := by
  intro J Q Pos V _ C B _ _ S m s pf w L H F κ δ' hL hδ hfork
  exact ofReal_le_of_sq_le hδ (by positivity) (Accounting.SecurityProofs.TS.DrawnBindingOfFork C B S m s pf w L hL)
    hfork _ (fun h => h)

end Accounting.SecurityProofs.Compose.Binding
