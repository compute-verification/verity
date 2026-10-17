import Proofs.Flock.Soundness.Game.Interleave

/-!
# Piece (a), the oracle count: two interactions, each continued on its own

Candidate 3's table runs its two reps concurrently and then each rep's sparse phase on that rep's outputs, rep 0's
first. `value_interleave_bind_le` is `value_interleave_le` with those continuations: the probability that both
continued interactions end in their events is at most the product of the two continued values, since each
continuation reads only its own interaction's output and draws its own coins.

It goes through `valueF`, the value of a game whose outcome `a` pays `u a` (`value p` is `valueF` at `p`'s
indicator): `value p (g.bind f)` is `valueF` at the continuations' values (`value_bind_valueF`), and the repetition
theorem holds for products of payoffs (`valueF_interleave_le`).
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Game
open scoped ENNReal

variable {α β γ δ : Type}

/-- **The value of a game whose outcome `a` pays `u a`**, for an optimal prover. -/
noncomputable def valueF (u : α → ℝ≥0∞) : Game α → ℝ≥0∞
  | .ret a => u a
  | .send _ k => ⨆ m, valueF u (k m)
  | @Game.coin _ C iF _ k => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * valueF u (k c)

theorem valueF_bind (u : β → ℝ≥0∞) (f : α → Game β) :
    ∀ g : Game α, valueF u (g.bind f) = valueF (fun a => valueF u (f a)) g
  | .ret _ => rfl
  | .send _ k => iSup_congr fun m => valueF_bind u f (k m)
  | @Game.coin _ C _ _ k => Finset.sum_congr rfl fun c _ => by rw [← valueF_bind u f (k c)]

open Classical in
theorem value_eq_valueF (p : α → Prop) : ∀ g : Game α, value p g = valueF (fun a => if p a then 1 else 0) g
  | .ret _ => rfl
  | .send _ k => iSup_congr fun m => value_eq_valueF p (k m)
  | @Game.coin _ C _ _ k => Finset.sum_congr rfl fun c _ => by rw [value_eq_valueF p (k c)]

theorem value_bind_valueF (p : β → Prop) (f : α → Game β) (g : Game α) :
    value p (g.bind f) = valueF (fun a => value p (f a)) g := by
  classical
  rw [value_eq_valueF, valueF_bind]
  congr 1
  funext a
  rw [value_eq_valueF]

theorem valueF_mono {u v : α → ℝ≥0∞} (h : ∀ a, u a ≤ v a) : ∀ g : Game α, valueF u g ≤ valueF v g
  | .ret a => h a
  | .send _ k => iSup_mono fun m => valueF_mono h (k m)
  | @Game.coin _ _ _ _ k => Finset.sum_le_sum fun c _ => mul_le_mul_right (valueF_mono h (k c)) _

theorem valueF_mul_const (v : α → ℝ≥0∞) (x : ℝ≥0∞) :
    ∀ g : Game α, valueF (fun a => v a * x) g = valueF v g * x
  | .ret _ => rfl
  | .send _ k => by
      simp only [valueF]
      rw [ENNReal.iSup_mul]
      exact iSup_congr fun m => valueF_mul_const v x (k m)
  | @Game.coin _ C _ _ k => by
      simp only [valueF]
      rw [Finset.sum_mul]
      exact Finset.sum_congr rfl fun c _ => by rw [valueF_mul_const v x (k c), mul_assoc]

theorem valueF_const_mul (v : α → ℝ≥0∞) (x : ℝ≥0∞) (g : Game α) :
    valueF (fun a => x * v a) g = x * valueF v g := by
  simp only [mul_comm x]
  exact valueF_mul_const v x g

theorem valueF_interleaveWith_le (u : α → ℝ≥0∞) (v : β → ℝ≥0∞)
    (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β)) (v₁ : ℝ≥0∞)
    (hadv : ∀ g₂, valueF (fun x => u x.1 * v x.2) (adv₁ g₂) ≤ v₁ * valueF v g₂)
    (hfin : ∀ b, valueF (fun x => u x.1 * v x.2) (fin₁ b) ≤ v₁ * v b) :
    ∀ g₂ : Game β, valueF (fun x => u x.1 * v x.2) (interleaveWith adv₁ fin₁ g₂) ≤ v₁ * valueF v g₂
  | .ret b => hfin b
  | .send M k => by
      simp only [interleaveWith, valueF]
      refine iSup_le fun first => ?_
      cases first
      · simp only [Bool.false_eq_true, ite_false, valueF]
        rw [ENNReal.mul_iSup]
        exact iSup_mono fun m => valueF_interleaveWith_le u v adv₁ fin₁ v₁ hadv hfin (k m)
      · simp only [ite_true]
        exact hadv _
  | @Game.coin _ C _ _ k => by
      simp only [interleaveWith, valueF]
      refine iSup_le fun first => ?_
      cases first
      · simp only [Bool.false_eq_true, ite_false, valueF]
        calc ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ *
              valueF (fun x => u x.1 * v x.2) (interleaveWith adv₁ fin₁ (k c))
            ≤ ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (v₁ * valueF v (k c)) :=
              Finset.sum_le_sum fun c _ => mul_le_mul_right
                (valueF_interleaveWith_le u v adv₁ fin₁ v₁ hadv hfin (k c)) _
          _ = v₁ * ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * valueF v (k c) := by
              rw [Finset.mul_sum]
              exact Finset.sum_congr rfl fun c _ => by ring
      · simp only [ite_true]
        exact hadv _

/-- **Repetition for payoffs**: whatever schedule the prover chooses, the product of the two interactions' payoffs is
worth at most the product of their values. -/
theorem valueF_interleave_le (u : α → ℝ≥0∞) (v : β → ℝ≥0∞) :
    ∀ (g₁ : Game α) (g₂ : Game β),
      valueF (fun x => u x.1 * v x.2) (interleave g₁ g₂) ≤ valueF u g₁ * valueF v g₂
  | .ret a, g₂ => by
      simp only [interleave, map, valueF_bind]
      exact (valueF_const_mul v (u a) g₂).le
  | .send M k, g₂ => by
      refine valueF_interleaveWith_le u v _ _ _ (fun g₂ => ?_) (fun b => ?_) g₂
      · simp only [valueF]
        rw [ENNReal.iSup_mul]
        exact iSup_mono fun m => valueF_interleave_le u v (k m) g₂
      · rw [map, valueF_bind]
        exact (valueF_mul_const u (v b) (.send M k)).le
  | @Game.coin _ C _ _ k, g₂ => by
      refine valueF_interleaveWith_le u v _ _ _ (fun g₂ => ?_) (fun b => ?_) g₂
      · simp only [valueF]
        calc ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * valueF (fun x => u x.1 * v x.2) (interleave (k c) g₂)
            ≤ ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (valueF u (k c) * valueF v g₂) :=
              Finset.sum_le_sum fun c _ => mul_le_mul_right (valueF_interleave_le u v (k c) g₂) _
          _ = (∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * valueF u (k c)) * valueF v g₂ := by
              rw [Finset.sum_mul]
              exact Finset.sum_congr rfl fun c _ => by ring
      · rw [map, valueF_bind]
        exact (valueF_mul_const u (v b) (@Game.coin _ C _ _ k)).le

/-- **Two interactions, each continued on its own output**: run `g₁` and `g₂` concurrently, then `k₁` on `g₁`'s
output and `k₂` on `g₂`'s. Both continuations end in their events with probability at most the product of the
continued interactions' values. -/
theorem value_interleave_bind_le (P : γ → Prop) (Q : δ → Prop) (g₁ : Game α) (g₂ : Game β)
    (k₁ : α → Game γ) (k₂ : β → Game δ) :
    value (fun x => P x.1 ∧ Q x.2) ((interleave g₁ g₂).bind fun ab =>
        (k₁ ab.1).bind fun c => (k₂ ab.2).bind fun d => .ret (c, d)) ≤
      value P (g₁.bind k₁) * value Q (g₂.bind k₂) := by
  classical
  rw [value_bind_valueF, value_bind_valueF P, value_bind_valueF Q]
  refine le_trans (valueF_mono (v := fun ab => value P (k₁ ab.1) * value Q (k₂ ab.2)) (fun ab => ?_) _)
    (valueF_interleave_le (fun a => value P (k₁ a)) (fun b => value Q (k₂ b)) g₁ g₂)
  rw [value_bind_valueF]
  have hc : ∀ c, value (fun x : γ × δ => P x.1 ∧ Q x.2) ((k₂ ab.2).bind fun d => .ret (c, d)) =
      (if P c then 1 else 0) * value Q (k₂ ab.2) := fun c => by
    rw [value_bind_ret]
    exact value_const_and (P c) Q _
  simp only [hc]
  rw [valueF_mul_const, ← value_eq_valueF]

/-! ## Leaves

Helpers applied by unification rather than by `rw`, for games whose types agree only up to unfolding (a statement's
`m` as the class's). -/

theorem eq_le_eq {a b c d : ℝ≥0∞} (h₁ : a = b) (h₂ : c = d) (h : b ≤ d) : a ≤ c := h₁ ▸ h₂ ▸ h

theorem value_ret_of_not {p : α → Prop} {a : α} (h : ¬ p a) : value p (.ret a) = 0 := by
  classical
  simp [h]

theorem value_ret_le {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a → q b) :
    value p (.ret a) ≤ value q (.ret b) := by
  classical
  simp only [value_ret]
  split_ifs with hp hq
  exacts [le_rfl, absurd (h hp) hq, zero_le, le_rfl]

theorem ite_some_of {c : Prop} [Decidable c] (hc : c) {Q : Option α → Prop} {a : α} (h : Q (some a)) :
    Q (if c then some a else none) := by
  simpa only [hc, ↓reduceIte] using h

theorem of_ite_some {c : Prop} [Decidable c] {a b : α} (h : (if c then some a else none) = some b) :
    c ∧ a = b := by
  split_ifs at h with hc
  exact ⟨hc, Option.some.inj h⟩

end FlockC3.Fork
