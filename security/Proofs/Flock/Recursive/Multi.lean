module

public import Proofs.Flock.Recursive.Compose
public import Proofs.Flock.Recursive.Defs
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Game

/-! Several outer sessions in a row (`seqC`), each recorded with the strategy the prover names for it after the sessions before. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game
open scoped ENNReal

section Exp

variable {α β : Type}

/-- **The expectation of `F` at the outcome of `g` played by `s`.** -/
noncomputable def exG (F : α → ℝ≥0∞) : (g : Game α) → Strategy g → ℝ≥0∞
  | .ret a, _ => F a
  | .send _ k, s => exG F (k s.1) s.2
  | @Game.coin _ C _ _ k, s => avg fun c : C => exG F (k c) (s c)

open Classical in
theorem prob_eq_exG (p : α → Prop) : ∀ (g : Game α) (s : Strategy g),
    prob p g s = exG (fun a => if p a then 1 else 0) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_eq_exG p (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => prob_eq_exG p (k c) (s c))

theorem exG_mono {F G : α → ℝ≥0∞} (h : ∀ a, F a ≤ G a) : ∀ (g : Game α) (s : Strategy g), exG F g s ≤ exG G g s
  | .ret a, _ => h a
  | .send _ k, s => exG_mono h (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => avg_mono fun c => exG_mono h (k c) (s c)

theorem exG_const (x : ℝ≥0∞) : ∀ (g : Game α) (s : Strategy g), exG (fun _ => x) g s = x
  | .ret _, _ => rfl
  | .send _ k, s => exG_const x (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => (congrArg avg (funext fun c => exG_const x (k c) (s c))).trans (avg_const x)

theorem exG_add (F G : α → ℝ≥0∞) : ∀ (g : Game α) (s : Strategy g),
    exG (fun a => F a + G a) g s = exG F g s + exG G g s
  | .ret _, _ => rfl
  | .send _ k, s => exG_add F G (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => (congrArg avg (funext fun c => exG_add F G (k c) (s c))).trans (avg_add _ _)

theorem exG_sum {ι : Type} (S : Finset ι) (F : ι → α → ℝ≥0∞) : ∀ (g : Game α) (s : Strategy g),
    exG (fun a => ∑ i ∈ S, F i a) g s = ∑ i ∈ S, exG (F i) g s
  | .ret _, _ => rfl
  | .send _ k, s => exG_sum S F (k s.1) s.2
  | @Game.coin _ _ _ _ k, s =>
      (congrArg avg (funext fun c => exG_sum S F (k c) (s c))).trans (avg_finset_sum S _)

theorem exG_const_mul (x : ℝ≥0∞) (F : α → ℝ≥0∞) : ∀ (g : Game α) (s : Strategy g),
    exG (fun a => x * F a) g s = x * exG F g s
  | .ret _, _ => rfl
  | .send _ k, s => exG_const_mul x F (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => by
      show avg (fun c => exG (fun a => x * F a) (k c) (s c)) = x * avg fun c => exG F (k c) (s c)
      rw [show (fun c => exG (fun a => x * F a) (k c) (s c)) = fun c => exG F (k c) (s c) * x from
        funext fun c => (exG_const_mul x F (k c) (s c)).trans (mul_comm _ _), avg_mul_const, mul_comm]

theorem exG_map (F : β → ℝ≥0∞) (f : α → β) : ∀ (g : Game α) (s : Strategy (g.map f)),
    exG F (g.map f) s = exG (fun a => F (f a)) g (Strategy.ofMap f g s)
  | .ret _, _ => rfl
  | .send _ k, s => exG_map F f (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => exG_map F f (k c) (s c))

theorem exG_bind_le {F G : β → ℝ≥0∞} (f : α → Game β) (h : ∀ a t, exG F (f a) t ≤ exG G (f a) t) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), exG F (g.bind f) s ≤ exG G (g.bind f) s
  | .ret a, s => h a s
  | .send _ k, s => exG_bind_le f h (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => avg_mono fun c => exG_bind_le f h (k c) (s c)

theorem exG_bind_of_const {F : β → ℝ≥0∞} (f : α → Game β) (H : α → ℝ≥0∞) (h : ∀ a t, exG F (f a) t = H a) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), exG F (g.bind f) s = exG H g (Strategy.ofBind f g s)
  | .ret a, s => h a s
  | .send _ k, s => exG_bind_of_const f H h (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => exG_bind_of_const f H h (k c) (s c))

theorem prob_bind_strat (p : β → Prop) : ∀ (g : Game α) (f : α → Game β) (s : Strategy g)
    (t : (a : α) → Strategy (f a)),
    prob p (g.bind f) (Strategy.bind g f s t) = exG (fun a => prob p (f a) (t a)) g s
  | .ret _, _, _, _ => rfl
  | .send _ k, f, s, t => prob_bind_strat p (k s.1) f s.2 t
  | @Game.coin _ _ _ _ k, f, s, t => congrArg avg (funext fun c => prob_bind_strat p (k c) f (s c) t)

theorem prob_map_strat (p : β → Prop) (f : α → β) : ∀ (g : Game α) (s : Strategy g),
    prob p (g.map f) (Strategy.map f g s) = prob (fun a => p (f a)) g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_map_strat p f (k s.1) s.2
  | @Game.coin _ _ _ _ k, s => congrArg avg (funext fun c => prob_map_strat p f (k c) (s c))

theorem map_ofMap (f : α → β) : ∀ (g : Game α) (s : Strategy (g.map f)), Strategy.map f g (Strategy.ofMap f g s) = s
  | .ret _, _ => rfl
  | .send _ k, s => by
      obtain ⟨m, s'⟩ := s
      exact congrArg (Sigma.mk m) (map_ofMap f (k m) s')
  | @Game.coin _ _ _ _ k, s => funext fun c => map_ofMap f (k c) (s c)

end Exp

section Seq

variable {α : Type}

/-- **`g` with the prover's messages fixed by `τ`**: only the coins remain. -/
def play : (g : Game α) → Strategy g → Game α
  | .ret a, _ => .ret a
  | .send _ k, τ => play (k τ.1) τ.2
  | @Game.coin _ C iF iN k, τ => @Game.coin _ C iF iN fun c => play (k c) (τ c)

theorem exG_play (F : α → ℝ≥0∞) : ∀ (g : Game α) (τ : Strategy g) (s : Strategy (play g τ)),
    exG F (play g τ) s = exG F g τ
  | .ret _, _, _ => rfl
  | .send _ k, τ, s => exG_play F (k τ.1) τ.2 s
  | @Game.coin _ _ _ _ k, τ, s => congrArg avg (funext fun c => exG_play F (k c) (τ c) (s c))

/-- **`g` played by a strategy the prover names first**: the prover sends a name `τ`, `g` is played by `of τ`, and the
output is the name with the outcome. -/
def committed {T : Type} (g : Game α) (of : T → Strategy g) : Game (T × α) :=
  .send T fun τ => (play g (of τ)).map fun a => (τ, a)

theorem exG_committed {T : Type} (g : Game α) (of : T → Strategy g) (F : T × α → ℝ≥0∞)
    (s : Strategy (committed g of)) : exG F (committed g of) s = exG (fun a => F (s.1, a)) g (of s.1) :=
  (exG_map F _ (play g (of s.1)) s.2).trans (exG_play _ g (of s.1) _)

/-- **`S` games in a row, each played by a strategy the prover names at its start**, after everything before it; the
output is each game's name and outcome. -/
def seqC : (S : ℕ) → {T α : Fin S → Type} → (gs : (j : Fin S) → Game (α j)) → ((j : Fin S) → T j → Strategy (gs j)) →
    Game ((j : Fin S) → T j × α j)
  | 0, _, _, _, _ => .ret fun j => j.elim0
  | S + 1, T, α, gs, of =>
      (committed (gs 0) (of 0)).bind fun r =>
        (seqC S (T := fun j => T j.succ) (fun j => gs j.succ) fun j => of j.succ).map fun rs =>
          (Fin.cons r rs : (j : Fin (S + 1)) → T j × α j)

theorem exG_seqC_le : ∀ (S : ℕ) {T α : Fin S → Type} (gs : (j : Fin S) → Game (α j))
    (of : (j : Fin S) → T j → Strategy (gs j)) (j : Fin S) (F' : T j × α j → ℝ≥0∞) (F : T j → ℝ≥0∞)
    (_ : ∀ τ, exG (fun a => F' (τ, a)) (gs j) (of j τ) ≤ F τ) (s : Strategy (seqC S gs of)),
    exG (fun rs => F' (rs j)) (seqC S gs of) s ≤ exG (fun rs => F (rs j).1) (seqC S gs of) s
  | 0, _, _, _, _, j, _, _, _, _ => j.elim0
  | S + 1, T, α, gs, of, j, F', F, hF, s => by
      cases j using Fin.cases with
      | zero =>
          refine (exG_bind_of_const _ (fun r => F' r) (fun r t => ?_) _ s).trans_le
            (le_of_le_of_eq ?_ (exG_bind_of_const _ (fun r => F r.1) (fun r t => ?_) _ s).symm)
          · rw [exG_map]
            simp only [Fin.cons_zero]
            exact exG_const _ _ _
          · rw [exG_committed, exG_committed]
            dsimp only
            rw [exG_const]
            exact hF _
          · rw [exG_map]
            simp only [Fin.cons_zero]
            exact exG_const _ _ _
      | succ j =>
          refine exG_bind_le _ (fun r t => ?_) _ s
          rw [exG_map, exG_map]
          simp only [Fin.cons_succ]
          exact exG_seqC_le S (T := fun j => T j.succ) (fun j => gs j.succ) (fun j => of j.succ) j F' F hF _

open Classical in
theorem prob_seqC_exists_le (S : ℕ) {T α : Fin S → Type} (gs : (j : Fin S) → Game (α j))
    (of : (j : Fin S) → T j → Strategy (gs j)) (E : (j : Fin S) → T j × α j → Prop) (B : (j : Fin S) → T j → ℝ≥0∞)
    (hB : ∀ j τ, prob (fun a => E j (τ, a)) (gs j) (of j τ) ≤ B j τ) (s : Strategy (seqC S gs of)) :
    prob (fun rs => ∃ j, E j (rs j)) (seqC S gs of) s ≤ exG (fun rs => ∑ j, B j (rs j).1) (seqC S gs of) s := by
  calc prob (fun rs => ∃ j, E j (rs j)) (seqC S gs of) s
      ≤ ∑ j, prob (fun rs => E j (rs j)) (seqC S gs of) s :=
        (prob_mono (q := fun rs : (j : Fin S) → T j × α j => ∃ j ∈ Finset.univ, E j (rs j))
          (fun _ ⟨j, h⟩ => ⟨j, Finset.mem_univ _, h⟩) _ _).trans
          (prob_exists_finset_le Finset.univ (fun j (rs : (j : Fin S) → T j × α j) => E j (rs j)) _ _)
    _ ≤ ∑ j, exG (fun rs => B j (rs j).1) (seqC S gs of) s := Finset.sum_le_sum fun j _ => by
        rw [prob_eq_exG]
        exact exG_seqC_le S gs of j (fun r => if E j r then 1 else 0) (B j)
          (fun τ => (prob_eq_exG _ _ _).symm.trans_le (hB j τ)) s
    _ = _ := (exG_sum _ _ _ _).symm

theorem outcome_play : ∀ (g : Game α) (τ : Strategy g) (s : Strategy (play g τ)) {a : α},
    Outcome (play g τ) s a → Outcome g τ a
  | .ret _, _, _, _, h => h
  | .send _ k, τ, s, _, h => outcome_play (k τ.1) τ.2 s h
  | @Game.coin _ _ _ _ k, τ, s, _, h => by
      obtain ⟨c, hc⟩ := h
      exact ⟨c, outcome_play (k c) (τ c) (s c) hc⟩

theorem outcome_committed {T : Type} (g : Game α) (of : T → Strategy g) (s : Strategy (committed g of))
    {r : T × α} (h : Outcome (committed g of) s r) : Outcome g (of r.1) r.2 := by
  obtain ⟨a, ha, hr⟩ := Outcome.of_map (fun a => (s.1, a)) (play g (of s.1)) s.2 h
  cases hr
  exact outcome_play g (of s.1) _ ha

theorem outcome_seqC : ∀ (S : ℕ) {T α : Fin S → Type} (gs : (j : Fin S) → Game (α j))
    (of : (j : Fin S) → T j → Strategy (gs j)) (s : Strategy (seqC S gs of)) {b : (j : Fin S) → T j × α j},
    Outcome (seqC S gs of) s b → ∀ j, Outcome (gs j) (of j (b j).1) (b j).2
  | 0, _, _, _, _, _, _, _, j => j.elim0
  | S + 1, T, α, gs, of, s, b, h, j => by
      obtain ⟨r, hr, u, hu⟩ := Outcome.of_bind (committed (gs 0) (of 0)) _ s h
      obtain ⟨rs, hrs, hb⟩ := Outcome.of_map _ _ u hu
      subst hb
      cases j using Fin.cases with
      | zero => simpa only [Fin.cons_zero] using outcome_committed (gs 0) (of 0) _ hr
      | succ j => simpa only [Fin.cons_succ] using
          outcome_seqC S (fun j => gs j.succ) (fun j => of j.succ) _ hrs j

end Seq

section Leaves

variable {M C β : Type} [Fintype C] [Nonempty C]

theorem AllLeaves.of_forall {r : ℕ} {f : List (M × C) → Game β} {P : (t : List (M × C)) → Strategy (f t) → Prop}
    (h : ∀ t s', P t s') (s : Strategy (rounds M C r f)) : AllLeaves r f P s := by
  induction r generalizing f with
  | zero => exact h [] s
  | succ r ih => exact fun ρ => ih (fun t s' => h _ s') (s.2 ρ)

end Leaves

end FlockSoundness.Discharge.Recursion
