import FlockSoundness.Game.Union

/-!
# Probabilities for a fixed prover strategy

`value` bounds the optimal prover. The table theorems (`table_sound_exec`) bound `prob` for every
strategy instead, and the two differ when some branch has no strategy at all (a `send` of an empty
type), so the audit theorems are stated for every strategy too. These are the strategy-level tools
they use:

* `prob_mono`, `prob_le_one`, `prob_or_le`: monotonicity and the union bound;
* `prob_and_const`: an event conjoined with a fact fixed by the strategy;
* `Strategy.ofMap`, `prob_map`: relabelling the output (`map`) keeps the game's strategies.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β : Type}

theorem prob_mono {p q : α → Prop} (h : ∀ a, p a → q a) :
    ∀ (g : Game α) (s : Strategy g), prob p g s ≤ prob q g s
  | .ret a, _ => by
      classical
      simp only [prob]
      by_cases hp : p a
      · simp [hp, h a hp]
      · simp [hp]
  | .send _ k, s => prob_mono h (k s.1) s.2
  | @Game.coin _ C _ _ k, s => Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono h (k c) (s c)) _

theorem prob_le_one (p : α → Prop) (g : Game α) (s : Strategy g) : prob p g s ≤ 1 :=
  (prob_le_value p g s).trans (value_le_one p g)

theorem prob_false : ∀ (g : Game α) (s : Strategy g), prob (fun _ => False) g s = 0
  | .ret _, _ => by simp [prob]
  | .send _ k, s => prob_false (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by simp only [prob, prob_false (k _) (s _), mul_zero, Finset.sum_const_zero]

theorem prob_coin (p : α → Prop) (C : Type) [Fintype C] [Nonempty C] (k : C → Game α)
    (s : Strategy (Game.coin C k)) :
    prob p (.coin C k) s = avg fun c => prob p (k c) (s c) := rfl

theorem prob_send (p : α → Prop) (M : Type) (k : M → Game α) (s : Strategy (Game.send M k)) :
    prob p (.send M k) s = prob p (k s.1) s.2 := rfl

/-- The union bound, for a fixed strategy. -/
theorem prob_or_le (p q : α → Prop) :
    ∀ (g : Game α) (s : Strategy g), prob (fun a => p a ∨ q a) g s ≤ prob p g s + prob q g s
  | .ret a, _ => by
      classical
      simp only [prob]
      by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]
  | .send _ k, s => prob_or_le p q (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      simp only [prob_coin]
      rw [← avg_add]
      exact avg_mono fun c => prob_or_le p q (k c) (s c)

/-- An event conjoined with a fact that does not depend on the run. -/
theorem prob_and_const (p : α → Prop) (Q : Prop) [Decidable Q] (g : Game α) (s : Strategy g) :
    prob (fun a => p a ∧ Q) g s = if Q then prob p g s else 0 := by
  by_cases hQ : Q
  · simp only [hQ, and_true, ite_true]
  · simp only [hQ, and_false, ite_false, prob_false]

theorem prCoin_false {C : Type} [Fintype C] : prCoin (fun _ : C => False) = 0 := by
  simp [prCoin, avg]

theorem avg_ite_eq_prCoin {C : Type} [Fintype C] (p : C → Prop) [DecidablePred p] :
    avg (fun c => if p c then (1 : ℝ≥0∞) else 0) = prCoin p := by
  unfold prCoin
  exact congrArg avg (funext fun c => by by_cases h : p c <;> simp [h])

/-- A strategy for the relabelled game is a strategy for the game. -/
def Strategy.ofMap (f : α → β) : (g : Game α) → Strategy (g.map f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofMap f (k c) (s c)

/-- Relabelling the output relabels the event. -/
theorem prob_map (p : β → Prop) (f : α → β) :
    ∀ (g : Game α) (s : Strategy (g.map f)),
      prob p (g.map f) s = prob (fun a => p (f a)) g (Strategy.ofMap f g s)
  | .ret _, _ => rfl
  | .send _ k, s => prob_map p f (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      show avg (fun c => prob p ((k c).map f) (s c)) = avg fun c => prob _ (k c) _
      exact congrArg avg (funext fun c => prob_map p f (k c) (s c))

end FlockSoundness.Game
