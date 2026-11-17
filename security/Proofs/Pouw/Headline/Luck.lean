module

public import Proofs.Pouw.TS
public import Proofs.Pouw.HMu
public import Proofs.ResourceAccounting.Lemmas.TS.Receipt
public import Proofs.ResourceAccounting.Lemmas.HMu

/-! The headlines' luck from the receipt. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.TS.Headline

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS ResourceAccounting.SecurityProofs.TS Pouw.RelativeSecurity

theorem operandLuck_of_pointMass {Ω : Type} [Fintype Ω] (μ : ℝ) :
    ∀ {S : ℕ} (G : ReceiptGame Ω S), G.OperandPointMass μ → G.OperandLuck μ
  | _, .done, _ => trivial
  | _, .round d _ right _ next, h => by
    obtain ⟨hd01, ⟨O, α, R, hPM, hid, hd, hr⟩, hN⟩ := h
    refine And.intro hd01 (And.intro ?_ fun ω => operandLuck_of_pointMass μ (next ω) (hN ω))
    have hInj : R.Inj 1 := by
      intro o _ b
      exact ⟨{b}, le_of_eq (Finset.card_singleton b), fun ω _ h => by
        rw [hid] at h
        exact Finset.mem_singleton.mpr h⟩
    have hB := ResourceAccounting.SecurityProofs.HMu.LemmaB Ω O α α R μ 1 ⟨hPM, hInj⟩
    have h2 : prob (fun ω => 0 < R.d (R.view ω)) = prob (fun ω => 0 < d ω) := by
      congr 1
      funext ω
      rw [hd ω]
    rw [h2] at hB
    calc prob (fun ω => right ω ∧ 0 < d ω) ≤ prob R.lucky :=
          prob_mono fun ω h => ⟨by rw [hid]; exact hr ω h.1, by rw [← hd ω]; exact h.2⟩
      _ ≤ ((1 : ℕ) : ℝ) * μ * prob (fun ω => 0 < d ω) := hB
      _ = μ * prob (fun ω => 0 < d ω) := by rw [Nat.cast_one, one_mul]

theorem luck_of_receipt_half {Ω : Type} [Fintype Ω] {S : ℕ} (g : RoundGame Ω S) (G : ReceiptGame Ω S)
    (hG : g = G.game) (hPin : G.Pinned) (hOp : OperandHMu (3 / 4) G) (hRL : ReceiptLuckAt (1 / 2) G) :
    LuckAt (7 / 8) g := by
  have h := LuckOfReceipt G (3 / 4) (1 / 2) hPin (operandLuck_of_pointMass (3 / 4) G hOp) hRL (by norm_num)
  have e : (3 / 4 : ℝ) + (1 - 3 / 4) * (1 / 2) = 7 / 8 := by norm_num
  rw [e, ← hG] at h
  exact h

end ResourceAccounting.SecurityProofs.TS.Headline
