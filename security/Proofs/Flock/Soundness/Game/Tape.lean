module

public import Definitions.Flock.Game.Tape
public import Definitions.Flock.Game
public import Proofs.Flock.Soundness.Game

/-! Sampling a strategy's coins ahead on a tape gives every event the probability it has when each coin is flipped as
the play reaches it (`prob_bind_tape`, `prob_seqG`). -/

@[expose] public section

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β γ : Type}

theorem Strategy.ofMap_map (f : α → β) :
    ∀ (g : Game α) (s : Strategy g), Strategy.ofMap f g (Strategy.map f g s) = s
  | .ret _, _ => rfl
  | .send _ k, s => by
      obtain ⟨m, s'⟩ := s
      exact congrArg (Sigma.mk m) (Strategy.ofMap_map f (k m) s')
  | @Game.coin _ _ _ _ k, s => funext fun c => Strategy.ofMap_map f (k c) (s c)

theorem avg_congr_fintype {C : Type} (i₁ i₂ : Fintype C) (f : C → ℝ≥0∞) : @avg C i₁ f = @avg C i₂ f := by
  rw [Subsingleton.elim i₁ i₂]

theorem avg_prod_iter {A B : Type} [Fintype A] [Fintype B] (F : A × B → ℝ≥0∞) :
    avg F = avg fun a => avg fun b => F (a, b) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [mul_assoc]

theorem avg_fst {A B : Type} [Fintype A] [Fintype B] [Nonempty B] (f : A → ℝ≥0∞) :
    avg (fun x : A × B => f x.1) = avg f := by
  rw [avg_prod_iter]
  exact congrArg avg (funext fun a => avg_const (f a))

theorem avg_pi_eval {ι : Type} [Fintype ι] [DecidableEq ι] {X : ι → Type} [∀ i, Fintype (X i)]
    [∀ i, Nonempty (X i)] (i : ι) (h : X i → ℝ≥0∞) : avg (fun T : (j : ι) → X j => h (T i)) = avg h := by
  let f : (j : ι) → X j → ℝ≥0∞ := fun j x => if hj : j = i then h (cast (congrArg X hj) x) else 1
  have h1 : ∀ T : (j : ι) → X j, ∏ j, f j (T j) = h (T i) := fun T => by
    rw [Finset.prod_eq_single i (fun j _ hj => dite_eq_right hj) (fun hi => absurd (Finset.mem_univ i) hi)]
    exact dite_eq_left rfl
  have h2 : ∏ j, avg (f j) = avg h := by
    rw [Finset.prod_eq_single i
      (fun j _ hj => (congrArg avg (funext fun x => dite_eq_right hj)).trans (avg_const 1))
      (fun hi => absurd (Finset.mem_univ i) hi)]
    exact congrArg avg (funext fun x => dite_eq_left rfl)
  calc avg (fun T : (j : ι) → X j => h (T i)) = avg (fun T : (j : ι) → X j => ∏ j, f j (T j)) :=
        congrArg avg (funext fun T => (h1 T).symm)
    _ = ∏ j, avg (f j) := avg_pi_prod f
    _ = avg h := h2

theorem avg_prod_pi {C : Type} [Fintype C] [DecidableEq C] {X : C → Type} [∀ c, Fintype (X c)]
    [∀ c, Nonempty (X c)] (h : (c : C) → X c → ℝ≥0∞) :
    avg (fun x : C × ((c : C) → X c) => h x.1 (x.2 x.1)) = avg fun c => avg (h c) := by
  rw [avg_prod_iter]
  exact congrArg avg (funext fun c => avg_pi_eval c (h c))

section Tape

variable (f : α → Game β) (T : (a : α) → Strategy (f a) → Type)

theorem prob_bind_tape (p : β → Prop) (hT : ∀ a s, Fintype (T a s)) (hN : ∀ a s, Nonempty (T a s))
    (V : (a : α) → (s : Strategy (f a)) → T a s → ℝ≥0∞)
    (hV : ∀ a s, prob p (f a) s = @avg _ (hT a s) (V a s)) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      prob p (g.bind f) s = @avg _ (Tape.fintype f T hT g s)
        (fun t => V (Tape.land f T g s t).1 (Tape.land f T g s t).2.1 (Tape.land f T g s t).2.2)
  | .ret a, s => hV a s
  | .send _ k, s => prob_bind_tape p hT hN V hV (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      classical
      let _ := fun c => Tape.fintype f T hT (k c) (s c)
      have := fun c => Tape.nonempty f T hN (k c) (s c)
      calc prob p ((Game.coin C k).bind f) s
          = avg (fun c => @avg _ (Tape.fintype f T hT (k c) (s c)) (fun t =>
              V (Tape.land f T (k c) (s c) t).1 (Tape.land f T (k c) (s c) t).2.1
                (Tape.land f T (k c) (s c) t).2.2)) :=
            congrArg avg (funext fun c => prob_bind_tape p hT hN V hV (k c) (s c))
        _ = avg (fun x : C × ((c : C) → Tape f T (k c) (s c)) =>
              V (Tape.land f T (k x.1) (s x.1) (x.2 x.1)).1 (Tape.land f T (k x.1) (s x.1) (x.2 x.1)).2.1
                (Tape.land f T (k x.1) (s x.1) (x.2 x.1)).2.2) :=
            (avg_prod_pi (X := fun c => Tape f T (k c) (s c)) fun c t =>
              V (Tape.land f T (k c) (s c) t).1 (Tape.land f T (k c) (s c) t).2.1
                (Tape.land f T (k c) (s c) t).2.2).symm
        _ = _ := avg_congr_fintype _ _ _

end Tape

theorem prob_const_ret (Q : Prop) (g : Game α) (s : Strategy g) (a : γ) :
    prob (fun _ => Q) g s = prob (fun _ => Q) (Game.ret a) PUnit.unit := by
  by_cases hQ : Q
  · simp only [hQ]
    exact (prob_true g s).trans (prob_true (Game.ret a) PUnit.unit).symm
  · simp only [hQ]
    exact (prob_false g s).trans (prob_false (Game.ret a) PUnit.unit).symm

theorem prob_seqG (g : β → Game α) (F : Option α → Prop) :
    ∀ (l : List β) (k : ℕ) (h : k < l.length) (s : Strategy (seqG g l)),
      prob (fun out => F out[k]?) (seqG g l) s =
        @avg _ (seqTape.fintype g l k h s) fun t =>
          prob (fun a => F (some a)) (g (l.get ⟨k, h⟩)) (seqStrat g l k h s t)
  | [], _, h, _ => absurd h (Nat.not_lt_zero _)
  | b :: l, 0, h, s => by
      have : Nonempty (seqTape g (b :: l) 0 h s) := ⟨PUnit.unit⟩
      refine (prob_bind_of_const (fun a => (seqG g l).map (a :: ·)) _ (fun a => F (some a)) (fun a t => ?_)
        (g b) s).trans (@avg_const _ (seqTape.fintype g (b :: l) 0 h s) _ _).symm
      rw [prob_map]
      simp only [List.getElem?_cons_zero]
      exact prob_const_ret (F (some a)) _ _ a
  | b :: l, k + 1, h, s => by
      refine (prob_bind_tape (fun a => (seqG g l).map (a :: ·))
        (fun a s' => seqTape g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s'))
        (fun out => F out[k + 1]?) (fun _ _ => seqTape.fintype g l k _ _) (fun _ _ => seqTape.nonempty g l k _ _)
        (fun a s' t => prob (fun x => F (some x)) (g (l.get ⟨k, Nat.lt_of_succ_lt_succ h⟩))
          (seqStrat g l k (Nat.lt_of_succ_lt_succ h) (Strategy.ofMap (a :: ·) (seqG g l) s') t))
        (fun a s' => ?_) (g b) s).trans rfl
      rw [prob_map]
      simp only [List.getElem?_cons_succ]
      exact prob_seqG g F l k (Nat.lt_of_succ_lt_succ h) _

end FlockSoundness.Game
