module

public import Definitions.Core.Decisive
public import Definitions.Core.Game.Tape
public import Definitions.Core.Interaction
public import Proofs.Core.Game.Prob

/-! A play's coins as one uniform coin. -/

@[expose] public section

namespace Verity.Game

open scoped ENNReal

variable {α : Type}

instance Tape.instNonempty : (g : Game α) → (s : Strategy g) → Nonempty (Tape g s)
  | .ret _, _ => inferInstanceAs (Nonempty Unit)
  | .send _ k, s => Tape.instNonempty (k s.1) s.2
  | @Game.coin _ C _ _ k, s =>
    haveI := fun c => Tape.instNonempty (k c) (s c)
    inferInstanceAs (Nonempty (C × ((c : C) → Tape (k c) (s c))))

namespace Tape

theorem avg_prod {C D : Type} [Fintype C] [Fintype D] (f : C × D → ℝ≥0∞) :
    avg f = avg fun c => avg fun d => f (c, d) := by
  unfold avg
  rw [Fintype.sum_prod_type, Fintype.card_prod, Nat.cast_mul,
    ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun d _ => ?_
  ring

theorem avg_equiv {C D : Type} [Fintype C] [Fintype D] (e : C ≃ D) (f : D → ℝ≥0∞) :
    avg (fun c => f (e c)) = avg f := by
  unfold avg
  rw [Fintype.card_congr e]
  exact Fintype.sum_equiv e _ _ (fun _ => rfl)

theorem prCoin_apply {ι : Type} [Fintype ι] [DecidableEq ι] {T : ι → Type} [∀ i, Fintype (T i)]
    [∀ i, Nonempty (T i)] (i : ι) (P : T i → Prop) :
    prCoin (fun f : (j : ι) → T j => P (f i)) = prCoin P := by
  classical
  unfold prCoin
  calc avg (fun f : (j : ι) → T j => if P (f i) then (1 : ℝ≥0∞) else 0)
      = avg (fun y : T i × ((j : {j // j ≠ i}) → T j) => if P y.1 then (1 : ℝ≥0∞) else 0) :=
        avg_equiv (Equiv.piSplitAt i T) (fun y => if P y.1 then (1 : ℝ≥0∞) else 0)
    _ = avg (fun x : T i => if P x then (1 : ℝ≥0∞) else 0) := by
        rw [avg_prod]
        refine congrArg avg (funext fun x => ?_)
        exact avg_const (C := (j : {j // j ≠ i}) → T j) (if P x then (1 : ℝ≥0∞) else 0)

theorem prCoin_pair {C : Type} [Fintype C] [DecidableEq C] {T : C → Type} [iT : ∀ c, Fintype (T c)]
    [nT : ∀ c, Nonempty (T c)] (P : (c : C) → T c → Prop) :
    prCoin (fun t : C × ((c : C) → T c) => P t.1 (t.2 t.1)) = avg fun c => prCoin (P c) := by
  unfold prCoin
  rw [avg_prod]
  refine congrArg avg (funext fun c => ?_)
  exact prCoin_apply (T := T) c (P c)

end Tape

open Classical in
theorem prob_eq_prCoin (p : α → Prop) : ∀ (g : Game α) (s : Strategy g),
    prob p g s = prCoin fun t : Tape g s => p (Tape.out g s t)
  | .ret a, s => by
      show (if p a then 1 else 0) = ∑ _t : Unit, ((Fintype.card Unit : ℝ≥0∞))⁻¹ * (if p a then 1 else 0)
      simp
  | .send _ k, s => prob_eq_prCoin p (k s.1) s.2
  | @Game.coin _ C _ _ k, s => by
      rw [prob_coin]
      have ih : ∀ c, prob p (k c) (s c) = prCoin fun x : Tape (k c) (s c) => p (Tape.out (k c) (s c) x) :=
        fun c => prob_eq_prCoin p (k c) (s c)
      simp only [ih]
      exact (Tape.prCoin_pair (T := fun c => Tape (k c) (s c)) (iT := fun c => Tape.instFintype (k c) (s c))
        (nT := fun c => Tape.instNonempty (k c) (s c)) fun c x => p (Tape.out (k c) (s c) x)).symm

theorem pr_eq_toReal_prCoin {C : Type} [Fintype C] (p : C → Prop) : Claims.Pr p = (prCoin p).toReal := by
  classical
  rw [prCoin_eq_card, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
  unfold Claims.Pr Claims.prUniform
  congr 2

theorem prCoin_eq_ofReal_pr {C : Type} [Fintype C] [Nonempty C] (p : C → Prop) :
    prCoin p = ENNReal.ofReal (Claims.Pr p) := by
  rw [pr_eq_toReal_prCoin, ENNReal.ofReal_toReal]
  exact ne_top_of_le_ne_top ENNReal.one_ne_top
    ((prCoin_mono (q := fun _ => True) fun _ _ => trivial).trans_eq (by simp [prCoin, avg_const]))

theorem prob_le_ofReal_pr (p : α → Prop) (g : Game α) (s : Strategy g) (q : Tape g s → Prop)
    (h : ∀ t, p (Tape.out g s t) → q t) : prob p g s ≤ ENNReal.ofReal (Claims.Pr q) := by
  rw [prob_eq_prCoin, ← prCoin_eq_ofReal_pr]
  exact prCoin_mono h

theorem prCoin_or_le {C : Type} [Fintype C] (p q : C → Prop) :
    prCoin (fun c => p c ∨ q c) ≤ prCoin p + prCoin q := by
  classical
  unfold prCoin
  rw [← avg_add]
  refine avg_mono fun c => ?_
  by_cases hp : p c <;> by_cases hq : q c <;> simp [hp, hq]

theorem Tape.reaches_out : ∀ (g : Game α) (s : Strategy g) (t : Tape g s), Reaches g (Tape.out g s t)
  | .ret _, _, _ => rfl
  | .send _ k, s, t => ⟨s.1, Tape.reaches_out (k s.1) s.2 t⟩
  | @Game.coin _ C _ _ k, s, t =>
    let t' : C × ((c : C) → Tape (k c) (s c)) := t
    ⟨t'.1, Tape.reaches_out (k t'.1) (s t'.1) (t'.2 t'.1)⟩

theorem reaches_bind {β : Type} {f : α → Game β} {b : β} :
    ∀ {g : Game α}, Reaches (g.bind f) b → ∃ a, Reaches g a ∧ Reaches (f a) b
  | .ret a, h => ⟨a, rfl, h⟩
  | .send _ k, ⟨m, h⟩ => let ⟨a, ha, hb⟩ := reaches_bind (g := k m) h; ⟨a, ⟨m, ha⟩, hb⟩
  | @Game.coin _ _ _ _ k, ⟨c, h⟩ => let ⟨a, ha, hb⟩ := reaches_bind (g := k c) h; ⟨a, ⟨c, ha⟩, hb⟩

end Verity.Game
