import Definitions.Core.Guarantees.Game
import Proofs.Core.Game.Basic
import Proofs.Core.Game.Interleave
import Proofs.Core.Game.Union
import Proofs.Core.Game.Batch
import Proofs.Core.Game.BatchInduced

/-! # Proofs that sound games compose (`Verity.Guarantees.Game`) -/

namespace Verity.SecurityProofs

open scoped ENNReal
open Game

theorem SoundAnd : Guarantees.SoundAnd := by
  intro α g accept E F δ ε hE hF
  unfold Sound at *
  calc value (fun a => accept a ∧ ¬ (E a ∧ F a)) g
      ≤ value (fun a => (accept a ∧ ¬ E a) ∨ (accept a ∧ ¬ F a)) g :=
        value_mono (fun a ⟨ha, hn⟩ => by tauto) g
    _ ≤ value (fun a => accept a ∧ ¬ E a) g + value (fun a => accept a ∧ ¬ F a) g := value_or_le _ _ g
    _ ≤ δ + ε := add_le_add hE hF

theorem SoundMono : Guarantees.SoundMono := by
  intro α g accept accept' E F δ δ' h hacc hEF hδ
  unfold Sound at *
  refine le_trans (value_mono (fun a ⟨ha, hF⟩ => ⟨hacc a ha, fun hE => hF (hEF a ha hE)⟩) g) (h.trans hδ)

/-- The probability of an event about the first of two interleaved games is at most its probability in that game
alone. -/
private theorem value_interleave_fst {α β : Type} (p : α → Prop) (g₁ : Game α) (g₂ : Game β) :
    value (fun x => p x.1) (g₁.interleave g₂) ≤ value p g₁ :=
  calc value (fun x => p x.1) (g₁.interleave g₂)
      = value (fun x => p x.1 ∧ True) (g₁.interleave g₂) := value_congr (fun _ => by simp) _
    _ ≤ value p g₁ * value (fun _ => True) g₂ := value_interleave_le p (fun _ => True) g₁ g₂
    _ ≤ value p g₁ * 1 := mul_le_mul_right (value_le_one _ g₂) _
    _ = value p g₁ := mul_one _

private theorem value_interleave_snd {α β : Type} (q : β → Prop) (g₁ : Game α) (g₂ : Game β) :
    value (fun x => q x.2) (g₁.interleave g₂) ≤ value q g₂ :=
  calc value (fun x => q x.2) (g₁.interleave g₂)
      = value (fun x => True ∧ q x.2) (g₁.interleave g₂) := value_congr (fun _ => by simp) _
    _ ≤ value (fun _ => True) g₁ * value q g₂ := value_interleave_le (fun _ => True) q g₁ g₂
    _ ≤ 1 * value q g₂ := mul_le_mul_left (value_le_one _ g₁) _
    _ = value q g₂ := one_mul _

theorem SoundInterleave : Guarantees.SoundInterleave := by
  intro α β g₁ g₂ accept₁ E₁ accept₂ E₂ δ₁ δ₂ h₁ h₂
  unfold Sound at *
  calc value (fun x => (accept₁ x.1 ∧ accept₂ x.2) ∧ ¬ (E₁ x.1 ∧ E₂ x.2)) (g₁.interleave g₂)
      ≤ value (fun x => (accept₁ x.1 ∧ ¬ E₁ x.1) ∨ (accept₂ x.2 ∧ ¬ E₂ x.2)) (g₁.interleave g₂) :=
        value_mono (fun x hx => by tauto) _
    _ ≤ value (fun x => accept₁ x.1 ∧ ¬ E₁ x.1) (g₁.interleave g₂)
          + value (fun x => accept₂ x.2 ∧ ¬ E₂ x.2) (g₁.interleave g₂) := value_or_le _ _ _
    _ ≤ δ₁ + δ₂ :=
        add_le_add ((value_interleave_fst (fun a => accept₁ a ∧ ¬ E₁ a) g₁ g₂).trans h₁)
          ((value_interleave_snd (fun b => accept₂ b ∧ ¬ E₂ b) g₁ g₂).trans h₂)

theorem SoundBatch : Guarantees.SoundBatch := by
  intro α pre post g d accept E δ h
  exact (value_batch_getD_le (fun a => accept a ∧ ¬ E a) d g post pre).trans h

theorem ProbBatchInduced : Guarantees.ProbBatchInduced :=
  fun pre post g d Q Cls ε hg σ hσ => prob_batch_getD_le_of_induced Q d g post pre Cls ε hg σ hσ

theorem SoundBind : Guarantees.SoundBind := by
  intro α β g f accept E δ h
  exact value_bind_le_of_forall g f _ δ h

end Verity.SecurityProofs
