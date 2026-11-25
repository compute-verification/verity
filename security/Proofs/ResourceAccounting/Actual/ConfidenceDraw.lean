module

public import Definitions.Pouw.Confidence
public import Proofs.Core.Game.Draw

/-! Soundness against the feasible runs is soundness (`Verity.Sound`) at the run of the verifier's one coin. -/

@[expose] public section

namespace ResourceAccounting

open Verity Finset

/-- `ResourceAccounting.Sound` at an error of at least 0 is, at each admissible strategy, soundness at the run in
which the verifier's coin is drawn. -/
theorem sound_iff_draw {L : Type*} [LinearOrder L] (Ω : Type) [Fintype Ω] [Nonempty Ω] (𝓕 : Set (Actual.Run L))
    (V : Finset L) (acc : Set (ℕ × L × ℕ × ℕ × ℕ) → Prop) (E : (Ω → Actual.Run L) → Ω → Prop) {δ : ℝ}
    (hδ : 0 ≤ δ) :
    Sound Ω 𝓕 V acc E δ ↔ ∀ S : Ω → Actual.Run L, (∀ r, S r ∈ 𝓕) → CausalFrom S →
      (∀ r r', (S r).inputs (↑V)ᶜ = (S r').inputs (↑V)ᶜ) →
      Verity.Sound (Game.draw Ω) (fun r => acc ((S r).view V)) (E S) (ENNReal.ofReal δ) := by
  classical
  refine forall_congr' fun S => forall_congr' fun _ => forall_congr' fun _ => forall_congr' fun _ => ?_
  rw [sound_draw_iff_pr hδ, Claims.Pr, Claims.prUniform,
    div_le_iff₀ (Nat.cast_pos.2 Fintype.card_pos)]
  exact Iff.of_eq (by congr!)

end ResourceAccounting
