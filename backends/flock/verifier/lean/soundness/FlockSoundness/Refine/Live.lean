import FlockSoundness.Refine.Tapes
import FlockSoundness.Protocol.Refine.Live

/-!
# The live session, and simulating it in the model

The transfer (R11) compares two interactions. The **live** one is the coin server's session as the executable sees it:
the prover sends framed bytes, the server draws coins, and at the end the executable checks what the server recorded.
The **model** one is the compiled game the soundness theorems bound.

* `Sim pL pg L g` is a simulation of a live interaction `L` by a model interaction `g`. It is data: how the model answers
  each live move, bijections between their coins, and a strategy for the model wherever no live run can succeed. With it
  come the facts that make a live acceptance (`pL`) a model acceptance (`pg`). `Sim.strategy` is the model strategy it
  plays against a live one, and `Sim.prob_le` compares the two probabilities.
* `liveTable` is one table's live session on the Blake3 unsalted path, whose server retains every round's bytes.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β : Type}

namespace Sim

variable {pL : β → Prop} {pg : α → Prop}

/-- **The model accepts at least as often.** Against any live strategy, the live verdict holds with probability at most
that of the model's verdict against the simulated strategy. -/
theorem prob_le : ∀ {L : Game β} {g : Game α} (d : Sim pL pg L g) (s : Strategy L),
    prob pL L s ≤ prob pg g (d.strategy s)
  | _, _, .ret h, _ => by
      classical
      simp only [prob]
      split_ifs with h₁ h₂ <;> simp_all
  | _, _, .model m d, s => by simp only [prob, strategy]; exact prob_le d s
  | _, _, .live f, s => by simp only [prob, strategy]; exact prob_le (f s.1) s.2
  | _, _, @coin _ _ _ _ C C' iF iN iF' iN' k k' e f, s => by
      simp only [prob, strategy]
      rw [← e.sum_comp, @Fintype.card_congr C' C iF' iF e]
      exact Finset.sum_le_sum fun c' _ => mul_le_mul_right (prob_le (f c') (s (e c'))) _
  | _, _, .dead h _, s => by rw [h s]; exact zero_le

end Sim

/-- An outcome of `g.map f` is the image of an outcome of `g`. -/
theorem Out.of_map {γ : Type} {f : β → γ} {g : Game β} {c : γ} (h : Out (g.map f) c) : ∃ b, Out g b ∧ c = f b := by
  obtain ⟨b, hb, hc⟩ := Out.of_bind h
  cases hc
  exact ⟨b, hb, rfl⟩

end FlockSoundness.Game

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game

local notation "𝔽" => FlockLevel3.GF128

theorem roundOf_coins (dg : ByteArray → ByteArray) (b : ByteArray) (c : Array F128) :
    (roundOf dg b c.size fun i => fOf c[i.val]!).coins = c := by
  apply Array.ext (by simp [roundOf])
  intro i h₁ h₂
  simp only [roundOf, Array.getElem_ofFn]
  rw [getElem!_pos c i h₂]
  rfl

/-- A live run's streams extend the rounds they had. -/
theorem liveRounds_out (dg : ByteArray → ByteArray) : ∀ (N : ℕ) (acc : List Round × List Round)
    {rs : List Round × List Round}, Out (liveRounds dg N acc) rs → ∃ l₀ l₁, rs = (acc.1 ++ l₀, acc.2 ++ l₁)
  | 0, acc, _, .ret _ => ⟨[], [], by simp⟩
  | N + 1, acc, _, .send none (.ret _) => ⟨[], [], by simp⟩
  | N + 1, acc, _, .send (some s) (.coin c h) => by
      obtain ⟨l₀, l₁, rfl⟩ := liveRounds_out dg N _ h
      unfold addRound
      split_ifs
      · exact ⟨roundOf dg s.bytes s.n c :: l₀, l₁, by simp⟩
      · exact ⟨l₀, roundOf dg s.bytes s.n c :: l₁, by simp⟩

end FlockSoundness.Refine
