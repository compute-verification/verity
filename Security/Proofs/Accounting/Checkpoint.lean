import Models.Accounting.Guarantees.Checkpoint
import Proofs.Accounting.HMu.Luck

/-!
# The whole-unit guess from its checkpoints

`prob_prefix_le` is the chain rule: the first `k` checkpoints together with the mode have probability at most `q^k` times
the mode's, one checkpoint at a time. `WholeGuessOfCheckpoints` reads it at `k = n` on the event `same`, which lies
inside it. A negative `q` leaves nothing to prove: the first checkpoint's bound then makes the mode null.
-/

namespace Accounting.SecurityProofs.Checkpoint

open Accounting.TS Accounting.Assumptions.Checkpoint Accounting.HMuProofs
open Pouw.RelativeSecurity (prob)

theorem prefix_succ_iff {C : Type} (E : ℕ → C → Prop) (W : C → Prop) (k : ℕ) (c : C) :
    ((∀ r < k + 1, E r c) ∧ W c) ↔ (E k c ∧ (∀ r < k, E r c) ∧ W c) := by
  constructor
  · rintro ⟨hE, hW⟩
    exact ⟨hE k (Nat.lt_succ_self k), fun r hr => hE r (Nat.lt_succ_of_lt hr), hW⟩
  · rintro ⟨hk, hE, hW⟩
    refine ⟨fun r hr => ?_, hW⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hr with hr | rfl
    · exact hE r hr
    · exact hk

/-- **The chain rule over checkpoints.** -/
theorem prob_prefix_le {C : Type} [Fintype C] {q : ℝ} (hq : 0 ≤ q) {n : ℕ} {E : ℕ → C → Prop} {W : C → Prop}
    (h : CheckpointGuessAt q n E W) (k : ℕ) (hk : k ≤ n) :
    prob (fun c => (∀ r < k, E r c) ∧ W c) ≤ q ^ k * prob W := by
  induction k with
  | zero =>
      rw [pow_zero, one_mul]
      exact prob_mono fun c hc => hc.2
  | succ k ih =>
      calc prob (fun c => (∀ r < k + 1, E r c) ∧ W c)
          = prob (fun c => E k c ∧ (∀ r < k, E r c) ∧ W c) := prob_congr (prefix_succ_iff E W k)
        _ ≤ q * prob (fun c => (∀ r < k, E r c) ∧ W c) := h k hk
        _ ≤ q * (q ^ k * prob W) := mul_le_mul_of_nonneg_left (ih (Nat.le_of_succ_le hk)) hq
        _ = q ^ (k + 1) * prob W := by ring

theorem WholeGuessOfCheckpoints : Accounting.Guarantees.Checkpoint.WholeGuessOfCheckpoints := by
  intro C M _ _ mode s right same E n q hsame h
  have hd : ∀ c, (0 < (if mode c = s then (1 : ℝ) else 0)) ↔ mode c = s := fun c => by
    by_cases hc : mode c = s <;> simp [hc]
  have hP : prob (fun c => same c ∧ 0 < (if mode c = s then (1 : ℝ) else 0)) ≤
      prob (fun c => (∀ r < n, E r c) ∧ mode c = s) :=
    prob_mono fun c hc => ⟨hsame c hc.1, (hd c).1 hc.2⟩
  have hW : prob (fun c => 0 < (if mode c = s then (1 : ℝ) else 0)) = prob (fun c => mode c = s) := prob_congr hd
  show (∀ c, 0 ≤ (if mode c = s then (1 : ℝ) else 0) ∧ (if mode c = s then (1 : ℝ) else 0) ≤ 1) ∧
      prob (fun c => same c ∧ 0 < (if mode c = s then (1 : ℝ) else 0)) ≤
        q ^ n * prob (fun c => 0 < (if mode c = s then (1 : ℝ) else 0)) ∧
      ∀ _ : C, True
  refine ⟨fun c => ?_, ?_, fun _ => trivial⟩
  · by_cases hc : mode c = s <;> simp [hc]
  rw [hW]
  by_cases hq : 0 ≤ q
  · exact hP.trans (prob_prefix_le hq h n le_rfl)
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [pow_zero, one_mul]
    exact hP.trans (prob_mono fun c hc => hc.2)
  rw [not_le] at hq
  have h0 := h 0 hn
  have h0' : prob (fun c => (∀ r < 0, E r c) ∧ mode c = s) = prob (fun c => mode c = s) :=
    prob_congr fun c => by simp
  have hz : prob (fun c => mode c = s) = 0 :=
    le_antisymm
      (by nlinarith [prob_nonneg (fun c => E 0 c ∧ (∀ r < 0, E r c) ∧ mode c = s), prob_nonneg (fun c => mode c = s)])
      (prob_nonneg _)
  rw [hz, mul_zero]
  exact hP.trans ((prob_mono fun c hc => hc.2).trans hz.le)

end Accounting.SecurityProofs.Checkpoint
