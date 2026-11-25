module

public import Proofs.Core.Game.Tape

/-!
# One uniform coin as a run

`Game.draw C` is the run in which the verifier flips one uniform coin of `C` and ends with it. Its probability at
every strategy and its value are the coin's probability `prCoin` (`prob_draw`, `value_draw`), and `Claims.Pr` is that
probability as a real (`pr_eq_toReal_prCoin`). So a bound stated over one uniform coin, in `prCoin` or in `Claims.Pr`,
is soundness at the run `Game.draw C` (`sound_draw_iff`, `sound_draw_of_pr`, `pr_le_of_sound_draw`): a statement in
either reads the tree's one probability and one soundness. `sound_of_imp` and `Sound.mono` are soundness's two
general steps: a claim that holds wherever the verifier accepts is sound with no error, and soundness passes to a
verifier that accepts less, a weaker claim and a larger error.
-/

@[expose] public section

namespace Verity

open scoped ENNReal

variable {C : Type} [Fintype C] [Nonempty C]

namespace Game

/-- The value of one uniform coin is the coin's probability. -/
theorem value_draw (p : C → Prop) : value p (Game.draw C) = prCoin p :=
  value_coin_ret p C id

/-- The probability of an event of one uniform coin, at its one strategy, is the coin's probability. -/
theorem prob_draw (p : C → Prop) (s : Strategy (Game.draw C)) : prob p (Game.draw C) s = prCoin p := by
  classical
  change ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * prob p (.ret c) (s c) = prCoin p
  simp only [prob, prCoin, avg]

end Game

/-- Soundness at one uniform coin is the bound on the coin's probability. -/
theorem sound_draw_iff {acc E : C → Prop} {δ : ℝ≥0∞} :
    Sound (Game.draw C) acc E δ ↔ Game.prCoin (fun c => acc c ∧ ¬ E c) ≤ δ := by
  rw [Sound, Game.value_draw]

/-- A bound on `Claims.Pr` over one uniform coin is soundness at the run of that coin. -/
theorem sound_draw_of_pr {acc E : C → Prop} {ε : ℝ} (h : Claims.Pr (fun c => acc c ∧ ¬ E c) ≤ ε) :
    Sound (Game.draw C) acc E (ENNReal.ofReal ε) := by
  rw [sound_draw_iff, Game.prCoin_eq_ofReal_pr]
  exact ENNReal.ofReal_le_ofReal h

/-- Soundness at the run of one uniform coin, at a real error, is the bound on `Claims.Pr`. -/
theorem pr_le_of_sound_draw {acc E : C → Prop} {ε : ℝ} (hε : 0 ≤ ε)
    (h : Sound (Game.draw C) acc E (ENNReal.ofReal ε)) : Claims.Pr (fun c => acc c ∧ ¬ E c) ≤ ε := by
  rw [sound_draw_iff] at h
  rw [Game.pr_eq_toReal_prCoin]
  exact ENNReal.toReal_le_of_le_ofReal hε h

/-- A claim that holds at every outcome the verifier accepts is sound with no error. -/
theorem sound_of_imp {α : Type} (g : Game α) {acc E : α → Prop} (h : ∀ a, acc a → E a) : Sound g acc E 0 := by
  have h0 : Game.value (fun a => acc a ∧ ¬ E a) g ≤ Game.value (fun _ => False) g :=
    Game.value_mono (fun a ha => ha.2 (h a ha.1)) g
  exact h0.trans (Game.value_false g).le

/-- Soundness passes to a verifier that accepts less, a weaker claim and a larger error. -/
theorem Sound.mono {α : Type} {g : Game α} {acc acc' E E' : α → Prop} {δ δ' : ℝ≥0∞} (h : Sound g acc E δ)
    (hacc : ∀ a, acc' a → acc a) (hE : ∀ a, E a → E' a) (hδ : δ ≤ δ') : Sound g acc' E' δ' :=
  ((Game.value_mono (fun a ha => ⟨hacc a ha.1, fun he => ha.2 (hE a he)⟩) g).trans h).trans hδ

end Verity
