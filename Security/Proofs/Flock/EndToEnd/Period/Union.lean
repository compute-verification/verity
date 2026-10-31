import Proofs.Flock.EndToEnd.Period.Run
import Proofs.Flock.Soundness.Game.Prob
import Proofs.Flock.Soundness.Game.Interleave
import Proofs.Flock.Soundness.Game.Seq
import Mathlib.Algebra.BigOperators.Field

/-!
# A period's run: the union bound and the horizon escape

* `period_union`: if every sampling block's event has probability at most `ε i` at every strategy satisfying `good i`,
  then in a period whose prover satisfies `good` at every sampling block on every run (`PeriodGood`), some sampling
  block's event happens with probability at most `Σ_i ε i`.
* `period_horizon`: if at every such strategy the sampling block's records all carry one weight `w` and the sampling
  block's "miss" event has probability at most `x ^ w`, then every sampling block misses with total weight at least `M`
  with probability at most `x ^ M`, for `x ≤ 1`. It is `FlockSoundness.Audit.Law.horizon_escape`'s induction on the
  period's game tree: sampling block `i`'s weight is fixed by the strategy that plays it, before its coins.

* `period_unionE`: `period_union` against bounds `ε i + f i τ` that read the strategy: the sum gains `f`'s expectation
  at the strategy that plays each sampling block (`periodE`). `expect_compTo` is that expectation as a game's: a game
  played after the period's sampling blocks up to one (`compTo`).

Both are inductions on the sampling blocks; within a sampling block, `value_runBind_le_add` and `value_runBind_le_mul`
take the sampling block's own probability out of the run, with the rest of the period bounded at every end of the
sampling block.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α δ γ : Type}

theorem atEnds_mono {f : α → Game δ} {Q Q' : (a : α) → Strategy (f a) → Prop} (hQ : ∀ a s', Q a s' → Q' a s') :
    ∀ (g : Game α) (s : Strategy (g.bind f)), AtEnds f Q g s → AtEnds f Q' g s
  | .ret a, s, h => hQ a s h
  | .send _ kk, s, h => atEnds_mono hQ (kk s.1) s.2 h
  | @Game.coin _ _ _ _ kk, s, h => fun c => atEnds_mono hQ (kk c) (s c) (h c)

private theorem avg_inv_card {C : Type} [Fintype C] [Nonempty C] :
    ∑ _x : C, (Fintype.card C : ℝ≥0∞)⁻¹ = 1 := by
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  exact ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)

theorem atEnds_of_forall {f : α → Game δ} {Q : (a : α) → Strategy (f a) → Prop} (hQ : ∀ a s', Q a s') :
    ∀ (g : Game α) (s : Strategy (g.bind f)), AtEnds f Q g s
  | .ret a, s => hQ a s
  | .send _ kk, s => atEnds_of_forall hQ (kk s.1) s.2
  | @Game.coin _ _ _ _ kk, s => fun c => atEnds_of_forall hQ (kk c) (s c)

/-- **The expectation at the ends of `g`** played by `s`: of `c a s'` at the end `a`, with `s'` the strategy `s` leaves
for `f a`, over `g`'s coins. -/
noncomputable def endsE (f : α → Game δ) (c : (a : α) → Strategy (f a) → ℝ) :
    (g : Game α) → Strategy (g.bind f) → ℝ
  | .ret a, s => c a s
  | .send _ kk, s => endsE f c (kk s.1) s.2
  | @Game.coin _ C iF _ kk, s => (∑ x : C, endsE f c (kk x) (s x)) / (@Fintype.card C iF)

/-- **A run's expectation is the expectation, at the ends of its first game, of the rest's.** -/
theorem expect_runBind (F : γ → ℝ) (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ)
    (ks : (a : α) → (s' : Strategy (f a)) → Strategy (k a s')) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      expect F (runBind f k g s) (runBindStrat f k ks g s) = endsE f (fun a s' => expect F (k a s') (ks a s')) g s
  | .ret _, _ => rfl
  | .send _ kk, s => expect_runBind F f k ks (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => by
      show (∑ x : C, expect F (runBind f k (kk x) (s x)) (runBindStrat f k ks (kk x) (s x))) / _ = _
      simp only [expect_runBind F f k ks (kk _) (s _)]
      rfl

theorem endsE_congr {f : α → Game δ} {c c' : (a : α) → Strategy (f a) → ℝ} (h : ∀ a s', c a s' = c' a s')
    (g : Game α) (s : Strategy (g.bind f)) : endsE f c g s = endsE f c' g s := by
  have : c = c' := funext fun a => funext fun s' => h a s'
  subst this
  rfl

theorem endsE_nonneg {f : α → Game δ} {c : (a : α) → Strategy (f a) → ℝ} (hc : ∀ a s', 0 ≤ c a s') :
    ∀ (g : Game α) (s : Strategy (g.bind f)), 0 ≤ endsE f c g s
  | .ret a, s => hc a s
  | .send _ kk, s => endsE_nonneg hc (kk s.1) s.2
  | @Game.coin _ _ _ _ kk, s =>
      div_nonneg (Finset.sum_nonneg fun x _ => endsE_nonneg hc (kk x) (s x)) (Nat.cast_nonneg _)

theorem endsE_mono {f : α → Game δ} {c c' : (a : α) → Strategy (f a) → ℝ} :
    ∀ (g : Game α) (s : Strategy (g.bind f)), AtEnds f (fun a s' => c a s' ≤ c' a s') g s →
      endsE f c g s ≤ endsE f c' g s
  | .ret _, _, h => h
  | .send _ kk, s, h => endsE_mono (kk s.1) s.2 h
  | @Game.coin _ _ _ _ kk, s, h =>
      div_le_div_of_nonneg_right (Finset.sum_le_sum fun x _ => endsE_mono (kk x) (s x) (h x)) (Nat.cast_nonneg _)

theorem endsE_const (f : α → Game δ) (r : ℝ) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), endsE f (fun _ _ => r) g s = r
  | .ret _, _ => rfl
  | .send _ kk, s => endsE_const f r (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => by
      show (∑ x : C, endsE f (fun _ _ => r) (kk x) (s x)) / _ = r
      simp only [endsE_const f r (kk _) (s _), Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      exact mul_div_cancel_left₀ r (by exact_mod_cast Fintype.card_ne_zero)

theorem endsE_add (f : α → Game δ) (c c' : (a : α) → Strategy (f a) → ℝ) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      endsE f (fun a s' => c a s' + c' a s') g s = endsE f c g s + endsE f c' g s
  | .ret _, _ => rfl
  | .send _ kk, s => endsE_add f c c' (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => by
      show (∑ x : C, endsE f (fun a s' => c a s' + c' a s') (kk x) (s x)) / _ =
        (∑ x : C, endsE f c (kk x) (s x)) / _ + (∑ x : C, endsE f c' (kk x) (s x)) / _
      simp only [endsE_add f c c' (kk _) (s _), Finset.sum_add_distrib, add_div]

theorem endsE_mul (f : α → Game δ) (r : ℝ) (c : (a : α) → Strategy (f a) → ℝ) :
    ∀ (g : Game α) (s : Strategy (g.bind f)), endsE f (fun a s' => r * c a s') g s = r * endsE f c g s
  | .ret _, _ => rfl
  | .send _ kk, s => endsE_mul f r c (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => by
      show (∑ x : C, endsE f (fun a s' => r * c a s') (kk x) (s x)) / _ = r * ((∑ x : C, endsE f c (kk x) (s x)) / _)
      simp only [endsE_mul f r c (kk _) (s _), ← Finset.mul_sum, mul_div_assoc]

theorem endsE_sum {ι : Type} (t : Finset ι) (f : α → Game δ) (c : ι → (a : α) → Strategy (f a) → ℝ) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      endsE f (fun a s' => ∑ x ∈ t, c x a s') g s = ∑ x ∈ t, endsE f (c x) g s
  | .ret _, _ => rfl
  | .send _ kk, s => endsE_sum t f c (kk s.1) s.2
  | @Game.coin _ C _ _ kk, s => by
      show (∑ y : C, endsE f (fun a s' => ∑ x ∈ t, c x a s') (kk y) (s y)) / _ =
        ∑ x ∈ t, (∑ y : C, endsE f (c x) (kk y) (s y)) / _
      simp only [endsE_sum t f c (kk _) (s _)]
      rw [Finset.sum_comm, Finset.sum_div]

private theorem ofReal_avg {C : Type} [Fintype C] [Nonempty C] (E : C → ℝ) (hE : ∀ x, 0 ≤ E x) :
    ENNReal.ofReal ((∑ x, E x) / Fintype.card C) = ∑ x, (Fintype.card C : ℝ≥0∞)⁻¹ * ENNReal.ofReal (E x) := by
  rw [div_eq_mul_inv, ENNReal.ofReal_mul (Finset.sum_nonneg fun x _ => hE x),
    ENNReal.ofReal_sum_of_nonneg (fun x _ => hE x),
    ENNReal.ofReal_inv_of_pos (by exact_mod_cast Fintype.card_pos), ENNReal.ofReal_natCast, Finset.sum_mul]
  exact Finset.sum_congr rfl fun x _ => mul_comm _ _

/-- **One sampling block, then the rest: the union step.** If the rest of the run has `p` with probability at most
`ε + c a s'` from every end `a` of `g` outside `q`, the run has `p` with probability at most `g`'s probability of `q`,
plus `ε`, plus the expectation of `c` at the ends of `g`. -/
theorem value_runBind_le_add (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ) (p : γ → Prop)
    (q : α → Prop) (ε : ℝ≥0∞) (c : (a : α) → Strategy (f a) → ℝ) (hc : ∀ a s', 0 ≤ c a s') :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      AtEnds f (fun a s' => ¬ q a → value p (k a s') ≤ ε + ENNReal.ofReal (c a s')) g s →
      value p (runBind f k g s) ≤ prob q g (Strategy.ofBind f g s) + (ε + ENNReal.ofReal (endsE f c g s))
  | .ret a, s, h => by
      classical
      show value p (k a s) ≤ (if q a then 1 else 0) + (ε + ENNReal.ofReal (c a s))
      by_cases hq : q a
      · rw [if_pos hq]
        exact (value_le_one p _).trans le_self_add
      · rw [if_neg hq, zero_add]
        exact h hq
  | .send _ kk, s, h => value_runBind_le_add f k p q ε c hc (kk s.1) s.2 h
  | @Game.coin _ C _ _ kk, s, h => by
      have hx : ∀ x, value p (runBind f k (kk x) (s x)) ≤
          prob q (kk x) (Strategy.ofBind f (kk x) (s x)) + (ε + ENNReal.ofReal (endsE f c (kk x) (s x))) :=
        fun x => value_runBind_le_add f k p q ε c hc (kk x) (s x) (h x)
      calc value p (runBind f k (Game.coin C kk) s)
          = ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * value p (runBind f k (kk x) (s x)) := value_coin _ _ _
        _ ≤ ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ *
              (prob q (kk x) (Strategy.ofBind f (kk x) (s x)) + (ε + ENNReal.ofReal (endsE f c (kk x) (s x)))) :=
            Finset.sum_le_sum fun x _ => by gcongr; exact hx x
        _ = (∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * prob q (kk x) (Strategy.ofBind f (kk x) (s x))) +
              ((∑ _x : C, (Fintype.card C : ℝ≥0∞)⁻¹) * ε +
                ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * ENNReal.ofReal (endsE f c (kk x) (s x))) := by
            rw [Finset.sum_mul, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
            exact Finset.sum_congr rfl fun x _ => by ring
        _ = prob q (Game.coin C kk) (Strategy.ofBind f (Game.coin C kk) s) +
              (ε + ENNReal.ofReal (endsE f c (Game.coin C kk) s)) := by
            rw [avg_inv_card, one_mul, ← ofReal_avg _ fun x => endsE_nonneg hc (kk x) (s x)]
            rfl

open Classical in
/-- **One sampling block, then the rest: the product step.** If the rest of the run has `p` with probability at most `c`
from every end of `g` in `q`, and never from an end outside `q`, the run has `p` with probability at most `g`'s
probability of `q`, times `c`. -/
theorem value_runBind_le_mul (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ) (p : γ → Prop)
    (q : α → Prop) (c : ℝ≥0∞) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      AtEnds f (fun a s' => value p (k a s') ≤ if q a then c else 0) g s →
      value p (runBind f k g s) ≤ prob q g (Strategy.ofBind f g s) * c
  | .ret a, s, h => by
      classical
      show value p (k a s) ≤ (if q a then 1 else 0) * c
      have h' : value p (k a s) ≤ if q a then c else 0 := h
      by_cases hq : q a
      · rw [if_pos hq] at h'
        rw [if_pos hq, one_mul]
        exact h'
      · rw [if_neg hq] at h'
        rw [if_neg hq, zero_mul]
        exact h'
  | .send _ kk, s, h => value_runBind_le_mul f k p q c (kk s.1) s.2 h
  | @Game.coin _ C _ _ kk, s, h => by
      have hc : ∀ x, value p (runBind f k (kk x) (s x)) ≤ prob q (kk x) (Strategy.ofBind f (kk x) (s x)) * c :=
        fun x => value_runBind_le_mul f k p q c (kk x) (s x) (h x)
      calc value p (runBind f k (Game.coin C kk) s)
          = ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * value p (runBind f k (kk x) (s x)) := value_coin _ _ _
        _ ≤ ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (prob q (kk x) (Strategy.ofBind f (kk x) (s x)) * c) :=
            Finset.sum_le_sum fun x _ => by gcongr; exact hc x
        _ = (∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * prob q (kk x) (Strategy.ofBind f (kk x) (s x))) * c := by
            rw [Finset.sum_mul]
            exact Finset.sum_congr rfl fun x _ => (mul_assoc _ _ _).symm
        _ = prob q (Game.coin C kk) (Strategy.ofBind f (Game.coin C kk) s) * c := rfl

variable {β : ℕ → Type}

theorem periodGood_mono (G : (i : ℕ) → Game (β i)) {good good' : (i : ℕ) → Strategy (G i) → Prop} :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)), (∀ j τ, i ≤ j → j < i + T → good j τ → good' j τ) →
      PeriodGood G good i T s → PeriodGood G good' i T s
  | 0, _, _, _, _ => trivial
  | T + 1, i, s, h, hs =>
      ⟨h i _ le_rfl (by omega) hs.1,
        atEnds_mono (fun _ s' hb => periodGood_mono G T (i + 1) s'
          (fun j τ h1 h2 => h j τ (by omega) (by omega)) hb) (G i) s hs.2⟩

theorem atEnds_of_atSomeEnd {ι : Type} {f : α → Game δ} {P : ι → (a : α) → Strategy (f a) → Prop}
    {Q : (a : α) → Strategy (f a) → Prop} {good : ι → Prop} (hQ : ∀ a s', (∀ x, P x a s' → good x) → Q a s') :
    ∀ (g : Game α) (s : Strategy (g.bind f)), (∀ x, AtSomeEnd f (P x) g s → good x) → AtEnds f Q g s
  | .ret a, s, h => hQ a s h
  | .send _ kk, s, h => atEnds_of_atSomeEnd hQ (kk s.1) s.2 h
  | @Game.coin _ _ _ _ kk, s, h => fun c => atEnds_of_atSomeEnd hQ (kk c) (s c) fun x hx => h x ⟨c, hx⟩

/-- **`PeriodGood` from its flat form**: `good` at every strategy the period plays a sampling block with
(`PeriodPlays`). -/
theorem periodGood_of_plays (G : (i : ℕ) → Game (β i)) (good : (i : ℕ) → Strategy (G i) → Prop) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)), (∀ j τ, PeriodPlays G i T s j τ → good j τ) →
      PeriodGood G good i T s
  | 0, _, _, _ => trivial
  | T + 1, i, s, h =>
      ⟨h i _ (Or.inl ⟨rfl, HEq.rfl⟩),
        atEnds_of_atSomeEnd (ι := (j : ℕ) × Strategy (G j))
          (P := fun x _ s' => PeriodPlays G (i + 1) T s' x.1 x.2) (good := fun x => good x.1 x.2)
          (fun _ s' h' => periodGood_of_plays G good T (i + 1) s' fun j τ hp => h' ⟨j, τ⟩ hp) (G i) s
          fun x hx => h x.1 x.2 (Or.inr hx)⟩

/-- **The expectation of `f` at sampling block `i + d`** of the period played by `s`, at the strategy that plays it
there, over the coins of the sampling blocks before it; `0` past the period's end. -/
noncomputable def periodE (G : (i : ℕ) → Game (β i)) (f : (j : ℕ) → Strategy (G j) → ℝ)
    (d i T : ℕ) (s : Strategy (periodG G i T)) : ℝ :=
  match T, s, d with
  | 0, _, _ => 0
  | _ + 1, s, 0 => f i (Strategy.ofBind _ (G i) s)
  | T + 1, s, d + 1 => endsE (fun _ => periodG G (i + 1) T) (fun _ s' => periodE G f d (i + 1) T s') (G i) s
termination_by structural T

section PeriodE

variable (G : (i : ℕ) → Game (β i))

theorem periodE_nonneg {f : (j : ℕ) → Strategy (G j) → ℝ} (hf : ∀ j τ, 0 ≤ f j τ) :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)), 0 ≤ periodE G f d i T s
  | _, _, 0, _ => le_rfl
  | 0, i, _ + 1, _ => hf i _
  | d + 1, i, T + 1, s => endsE_nonneg (fun _ s' => periodE_nonneg hf d (i + 1) T s') (G i) s

/-- `periodE` is monotone in `f` at the strategies satisfying `good`, on a period whose prover satisfies it. -/
theorem periodE_mono {good : (i : ℕ) → Strategy (G i) → Prop} {f f' : (j : ℕ) → Strategy (G j) → ℝ} :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)), PeriodGood G good i T s →
      (∀ j τ, j = i + d → good j τ → f j τ ≤ f' j τ) → periodE G f d i T s ≤ periodE G f' d i T s
  | _, _, 0, _, _, _ => le_rfl
  | 0, i, _ + 1, _, hs, h => h i _ rfl hs.1
  | d + 1, i, T + 1, s, hs, h => endsE_mono (G i) s (atEnds_mono (fun _ s' hg =>
      periodE_mono d (i + 1) T s' hg fun j τ hj => h j τ (by omega)) (G i) s hs.2)

theorem periodE_const_le (r : ℕ → ℝ) (hr : ∀ j, 0 ≤ r j) :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)), periodE G (fun j _ => r j) d i T s ≤ r (i + d)
  | _, _, 0, _ => hr _
  | 0, _, _ + 1, _ => le_rfl
  | d + 1, i, T + 1, s => by
      refine (endsE_mono (G i) s (atEnds_of_forall (fun _ s' => periodE_const_le r hr d (i + 1) T s') (G i) s)).trans
        (le_of_eq ?_)
      exact (endsE_const _ (r (i + 1 + d)) (G i) s).trans (congrArg r (by omega))

theorem periodE_zero : ∀ (d i T : ℕ) (s : Strategy (periodG G i T)), periodE G (fun _ _ => 0) d i T s = 0
  | _, _, 0, _ => rfl
  | 0, _, _ + 1, _ => rfl
  | d + 1, i, T + 1, s =>
      (endsE_congr (fun _ s' => periodE_zero d (i + 1) T s') (G i) s).trans (endsE_const _ 0 (G i) s)

theorem periodE_add (f f' : (j : ℕ) → Strategy (G j) → ℝ) :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)),
      periodE G (fun j τ => f j τ + f' j τ) d i T s = periodE G f d i T s + periodE G f' d i T s
  | _, _, 0, _ => (add_zero 0).symm
  | 0, _, _ + 1, _ => rfl
  | d + 1, i, T + 1, s =>
      (endsE_congr (fun _ s' => periodE_add f f' d (i + 1) T s') (G i) s).trans (endsE_add _ _ _ (G i) s)

theorem periodE_mul (a : ℕ → ℝ) (f : (j : ℕ) → Strategy (G j) → ℝ) :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)),
      periodE G (fun j τ => a j * f j τ) d i T s = a (i + d) * periodE G f d i T s
  | _, _, 0, _ => (mul_zero _).symm
  | 0, _, _ + 1, _ => rfl
  | d + 1, i, T + 1, s => by
      refine (endsE_congr (fun _ s' => periodE_mul a f d (i + 1) T s') (G i) s).trans ?_
      refine (endsE_mul _ (a (i + 1 + d)) (fun _ s' => periodE G f d (i + 1) T s') (G i) s).trans ?_
      exact congrArg (· * periodE G f (d + 1) i (T + 1) s) (congrArg a (by omega))

theorem periodE_sum {ι : Type} (t : Finset ι) (f : ι → (j : ℕ) → Strategy (G j) → ℝ) :
    ∀ (d i T : ℕ) (s : Strategy (periodG G i T)),
      periodE G (fun j τ => ∑ x ∈ t, f x j τ) d i T s = ∑ x ∈ t, periodE G (f x) d i T s
  | _, _, 0, _ => Finset.sum_const_zero.symm
  | 0, _, _ + 1, _ => rfl
  | d + 1, i, T + 1, s =>
      (endsE_congr (fun _ s' => periodE_sum t f d (i + 1) T s') (G i) s).trans (endsE_sum t _ _ (G i) s)

end PeriodE

/-- **The union bound over a period's sampling blocks, against bounds that read the strategy.** If every sampling
block's event has probability at most `ε i + f i τ` at every strategy `τ` satisfying `good i`, then in a period whose
prover satisfies `good` at every sampling block on every run, some sampling block's event happens with probability at
most `Σ_i ε i` plus the sum over the sampling blocks of `f`'s expectation at the strategy that plays each. -/
theorem period_unionE {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ)
    (bad : ρ → Prop) (good : (i : ℕ) → Strategy (G i) → Prop) (ε : ℕ → ℝ≥0∞) (f : (j : ℕ) → Strategy (G j) → ℝ)
    (hf : ∀ j τ, 0 ≤ f j τ)
    (hstage : ∀ i τ, good i τ → prob (fun b => bad (rec i τ b)) (G i) τ ≤ ε i + ENNReal.ofReal (f i τ)) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)), PeriodGood G good i T s →
      value (fun l => ∃ r ∈ l, bad r) (periodRun G rec i T s) ≤
        ∑ t ∈ Finset.range T, ε (i + t) + ENNReal.ofReal (∑ d ∈ Finset.range T, periodE G f d i T s)
  | 0, i, s, _ => by
      classical
      show value (fun l => ∃ r ∈ l, bad r) (Game.ret []) ≤ _
      simp
  | T + 1, i, s, hs => by
      have hc : ∀ (b : β i) (s' : Strategy (periodG G (i + 1) T)), 0 ≤ ∑ d ∈ Finset.range T, periodE G f d (i + 1) T s' :=
        fun _ s' => Finset.sum_nonneg fun d _ => periodE_nonneg G hf d (i + 1) T s'
      have h := value_runBind_le_add (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => ∃ r ∈ l, bad r) (fun b => bad (rec i (Strategy.ofBind _ (G i) s) b))
        (∑ t ∈ Finset.range T, ε (i + 1 + t)) (fun _ s' => ∑ d ∈ Finset.range T, periodE G f d (i + 1) T s') hc
        (G i) s
        (atEnds_mono (fun b s' hg hnb => by
          rw [value_map]
          refine (value_mono (fun l hl => ?_) _).trans (period_unionE G rec bad good ε f hf hstage T (i + 1) s' hg)
          obtain ⟨r, hr, hbr⟩ := hl
          rcases List.mem_cons.1 hr with rfl | hr
          · exact absurd hbr hnb
          · exact ⟨r, hr, hbr⟩) (G i) s hs.2)
      have hE : ∑ d ∈ Finset.range (T + 1), periodE G f d i (T + 1) s = f i (Strategy.ofBind _ (G i) s) +
          endsE (fun _ => periodG G (i + 1) T) (fun _ s' => ∑ d ∈ Finset.range T, periodE G f d (i + 1) T s') (G i) s := by
        rw [Finset.sum_range_succ', add_comm]
        exact congrArg (fun x => f i (Strategy.ofBind _ (G i) s) + x)
          (endsE_sum (Finset.range T) _ (fun d _ s' => periodE G f d (i + 1) T s') (G i) s).symm
      have hε : ∑ t ∈ Finset.range (T + 1), ε (i + t) = ε i + ∑ t ∈ Finset.range T, ε (i + 1 + t) := by
        rw [Finset.sum_range_succ', Nat.add_zero, add_comm]
        exact congrArg (ε i + ·) (Finset.sum_congr rfl fun t _ => congrArg ε (by omega))
      calc value (fun l => ∃ r ∈ l, bad r) (periodRun G rec i (T + 1) s)
          ≤ prob (fun b => bad (rec i (Strategy.ofBind _ (G i) s) b)) (G i) (Strategy.ofBind _ (G i) s) +
              (∑ t ∈ Finset.range T, ε (i + 1 + t) + ENNReal.ofReal (endsE (fun _ => periodG G (i + 1) T)
                (fun _ s' => ∑ d ∈ Finset.range T, periodE G f d (i + 1) T s') (G i) s)) := h
        _ ≤ (ε i + ENNReal.ofReal (f i (Strategy.ofBind _ (G i) s))) +
              (∑ t ∈ Finset.range T, ε (i + 1 + t) + ENNReal.ofReal (endsE (fun _ => periodG G (i + 1) T)
                (fun _ s' => ∑ d ∈ Finset.range T, periodE G f d (i + 1) T s') (G i) s)) := by
            gcongr; exact hstage i _ hs.1
        _ = ∑ t ∈ Finset.range (T + 1), ε (i + t) +
              ENNReal.ofReal (∑ d ∈ Finset.range (T + 1), periodE G f d i (T + 1) s) := by
            rw [hε, hE, ENNReal.ofReal_add (hf _ _) (endsE_nonneg hc (G i) s)]
            ring

/-- **The union bound over a period's sampling blocks.** -/
theorem period_union {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ)
    (bad : ρ → Prop) (good : (i : ℕ) → Strategy (G i) → Prop) (ε : ℕ → ℝ≥0∞)
    (hstage : ∀ i τ, good i τ → prob (fun b => bad (rec i τ b)) (G i) τ ≤ ε i)
    (T i : ℕ) (s : Strategy (periodG G i T)) (hs : PeriodGood G good i T s) :
    value (fun l => ∃ r ∈ l, bad r) (periodRun G rec i T s) ≤ ∑ t ∈ Finset.range T, ε (i + t) := by
  have h := period_unionE G rec bad good ε (fun _ _ => 0) (fun _ _ => le_rfl)
    (fun j τ hτ => by rw [ENNReal.ofReal_zero, add_zero]; exact hstage j τ hτ) T i s hs
  rwa [Finset.sum_congr rfl fun d _ => periodE_zero G d i T s, Finset.sum_const_zero, ENNReal.ofReal_zero,
    add_zero] at h

/-- **The horizon escape over a period's sampling blocks.** -/
theorem period_horizon {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ)
    (miss : ρ → Prop) (wt : ρ → ℕ) (good : (i : ℕ) → Strategy (G i) → Prop) (x : ℝ≥0∞) (hx : x ≤ 1)
    (hstage : ∀ i τ, good i τ →
      ∃ w : ℕ, (∀ b, wt (rec i τ b) = w) ∧ prob (fun b => miss (rec i τ b)) (G i) τ ≤ x ^ w) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)) (M : ℕ), PeriodGood G good i T s →
      value (fun l => (∀ r ∈ l, miss r) ∧ M ≤ (l.map wt).sum) (periodRun G rec i T s) ≤ x ^ M
  | 0, i, s, M, _ => by
      classical
      show value (fun l => (∀ r ∈ l, miss r) ∧ M ≤ (l.map wt).sum) (Game.ret []) ≤ _
      rcases Nat.eq_zero_or_pos M with rfl | hM
      · simp
      · simp [Nat.pos_iff_ne_zero.1 hM]
  | T + 1, i, s, M, hs => by
      obtain ⟨w, hw, hp⟩ := hstage i _ hs.1
      have h := value_runBind_le_mul (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => (∀ r ∈ l, miss r) ∧ M ≤ (l.map wt).sum) (fun b => miss (rec i (Strategy.ofBind _ (G i) s) b))
        (x ^ (M - w)) (G i) s
        (atEnds_mono (fun b s' hg => by
          classical
          rw [value_map]
          by_cases hm : miss (rec i (Strategy.ofBind _ (G i) s) b)
          · rw [if_pos hm]
            refine (value_mono (fun l hl => ?_) _).trans (period_horizon G rec miss wt good x hx hstage T (i + 1) s'
              (M - w) hg)
            obtain ⟨hall, hsum⟩ := hl
            refine ⟨fun r hr => hall r (List.mem_cons_of_mem _ hr), ?_⟩
            simp only [List.map_cons, List.sum_cons, hw b] at hsum
            omega
          · rw [if_neg hm]
            refine (value_mono (q := fun _ => False) (fun l hl => hm (hl.1 _ List.mem_cons_self)) _).trans_eq ?_
            exact value_false _) (G i) s hs.2)
      calc value (fun l => (∀ r ∈ l, miss r) ∧ M ≤ (l.map wt).sum) (periodRun G rec i (T + 1) s)
          ≤ prob (fun b => miss (rec i (Strategy.ofBind _ (G i) s) b)) (G i) (Strategy.ofBind _ (G i) s) *
              x ^ (M - w) := h
        _ ≤ x ^ w * x ^ (M - w) := by gcongr
        _ = x ^ (w + (M - w)) := (pow_add _ _ _).symm
        _ ≤ x ^ M := pow_le_pow_of_le_one zero_le hx (by omega)

/-- **The expectation of a game played after a period's sampling blocks up to `i + d`** is `periodE` of the game's
expectation at the strategy that plays sampling block `i + d`. -/
theorem expect_compTo {γ : Type} (G : (i : ℕ) → Game (β i)) (F : γ → ℝ) (K : (j : ℕ) → Strategy (G j) → Game γ)
    (KS : (j : ℕ) → (τ : Strategy (G j)) → Strategy (K j τ)) :
    ∀ (d i T : ℕ) (h : d < T) (s : Strategy (periodG G i T)),
      expect F (compTo G K d i T h s) (compStrat G K KS d i T h s) =
        periodE G (fun j τ => expect F (K j τ) (KS j τ)) d i T s
  | _, _, 0, h, _ => absurd h (Nat.not_lt_zero _)
  | 0, _, _ + 1, _, _ => rfl
  | d + 1, i, T + 1, _, s =>
      (expect_runBind F _ _ _ (G i) s).trans
        (endsE_congr (fun _ s' => expect_compTo G F K KS d (i + 1) T _ s') (G i) s)

end FlockSoundness.Game
