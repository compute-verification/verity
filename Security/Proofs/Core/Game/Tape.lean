import Proofs.Core.Game.Prob

/-!
# A strategy's play as a function of one uniform tape

A strategy fixes every message, so the play's outcome is a function of the coins alone. `Tape g s` holds, at each coin,
the coin and a tape for every branch below it; under the uniform law on tapes the coin is uniform and the branch's tape
is independent of it, so `Tape.run` has the play's law (`prob_eq_prCoin_tape`). A collision finder that replays the
strategy on such a tape (`Claims.Finder` over `Tape g s`) then succeeds with the probability of the play's event.
-/

namespace Verity.Game

open scoped ENNReal

variable {α : Type}

/-- **A coin tape for `g` under `s`**: at each coin, the coin and a tape for every branch. -/
def Tape : (g : Game α) → Strategy g → Type
  | .ret _, _ => Unit
  | .send _ k, s => Tape (k s.1) s.2
  | @Game.coin _ C _ _ k, s => C × ((c : C) → Tape (k c) (s c))

/-- The tapes are finite. -/
noncomputable def Tape.fintype : (g : Game α) → (s : Strategy g) → Fintype (Tape g s)
  | .ret _, _ => (inferInstance : Fintype Unit)
  | .send _ k, s => Tape.fintype (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
    letI := fun c => Tape.fintype (k c) (s c)
    haveI := Classical.decEq C
    (inferInstance : Fintype (C × ((c : C) → Tape (k c) (s c))))

/-- The tapes are inhabited. -/
theorem Tape.nonempty : (g : Game α) → (s : Strategy g) → Nonempty (Tape g s)
  | .ret _, _ => ⟨()⟩
  | .send _ k, s => Tape.nonempty (k s.1) s.2
  | @Game.coin _ C _ iN k, s => by
    obtain ⟨c⟩ := iN
    show Nonempty (C × ((c : C) → Tape (k c) (s c)))
    exact ⟨(c, fun c => Classical.choice (Tape.nonempty (k c) (s c)))⟩

noncomputable instance (g : Game α) (s : Strategy g) : Fintype (Tape g s) := Tape.fintype g s

instance (g : Game α) (s : Strategy g) : Nonempty (Tape g s) := Tape.nonempty g s

/-- **The play of `s` on the tape `t`**: each coin read off the tape, then the branch's own tape. -/
def Tape.run : (g : Game α) → (s : Strategy g) → Tape g s → α
  | .ret a, _, _ => a
  | .send _ k, s, t => Tape.run (k s.1) s.2 t
  | @Game.coin _ _ _ _ k, s, t => Tape.run (k t.1) (s t.1) (t.2 t.1)

theorem avg_prod {C P : Type} [Fintype C] [Fintype P] (f : C × P → ℝ≥0∞) :
    avg f = avg fun c => avg fun p => f (c, p) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun p _ => by rw [mul_assoc]

/-- **One coordinate of independent uniform coordinates is uniform.** -/
theorem prCoin_pi_eval {ι : Type} [Fintype ι] [DecidableEq ι] {Ω : ι → Type} [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (i : ι) (P : Ω i → Prop) :
    prCoin (fun ω : ∀ j, Ω j => P (ω i)) = prCoin P := by
  classical
  have key : (fun ω : ∀ j, Ω j => ∀ j ∈ ({i} : Finset ι), (∃ h : j = i, P (h ▸ ω j))) = fun ω => P (ω i) := by
    funext ω
    refine propext ⟨fun H => ?_, fun hp j hj => ?_⟩
    · obtain ⟨_, hp⟩ := H i (Finset.mem_singleton_self i)
      exact hp
    · rw [Finset.mem_singleton] at hj
      subst hj
      exact ⟨rfl, hp⟩
  have h := prCoin_pi_forall (Ω := Ω) {i} (fun j x => ∃ h : j = i, P (h ▸ x))
  beta_reduce at h
  rw [key, Finset.prod_singleton] at h
  rw [h]
  exact congrArg prCoin (funext fun x => propext ⟨fun ⟨_, hp⟩ => hp, fun hp => ⟨rfl, hp⟩⟩)

/-- **The tape has the play's law**: the probability of `p` in the play of `s` is the probability that `s` played on a
uniform tape ends in `p`. -/
theorem prob_eq_prCoin_tape (p : α → Prop) :
    ∀ (g : Game α) (s : Strategy g), prob p g s = prCoin fun t : Tape g s => p (Tape.run g s t)
  | .ret a, s => by
      classical
      show prob p (.ret a) s = prCoin fun _ : Unit => p a
      rw [← avg_ite_eq_prCoin, avg_const]
      by_cases h : p a <;> simp [prob, h]
  | .send _ k, s => prob_eq_prCoin_tape p (k s.1) s.2
  | @Game.coin _ C iF iN k, s => by
      classical
      let := fun c => Tape.fintype (k c) (s c)
      have := fun c => Tape.nonempty (k c) (s c)
      show prob p (.coin C k) s =
        prCoin fun t : C × ((c : C) → Tape (k c) (s c)) => p (Tape.run (k t.1) (s t.1) (t.2 t.1))
      rw [prob_coin, ← avg_ite_eq_prCoin, avg_prod]
      refine congrArg avg (funext fun c => ?_)
      try dsimp only
      rw [prob_eq_prCoin_tape p (k c) (s c), avg_ite_eq_prCoin,
        prCoin_pi_eval (Ω := fun c => Tape (k c) (s c)) c (fun t => p (Tape.run (k c) (s c) t))]

end Verity.Game
