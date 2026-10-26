import Proofs.Flock.EndToEnd.Period.Run
import Proofs.Flock.Soundness.Game.Prob
import Proofs.Flock.Soundness.Game.Interleave

/-!
# A period's run: the union bound and the horizon escape

* `period_union`: if every sampling block's event has probability at most `ε i` at every strategy satisfying `good i`,
  then in a period whose prover satisfies `good` at every sampling block on every run (`PeriodGood`), some sampling
  block's event happens with probability at most `Σ_i ε i`.
* `period_horizon`: if at every such strategy the sampling block's records all carry one weight `w` and the sampling
  block's "miss" event has probability at most `x ^ w`, then every sampling block misses with total weight at least `M`
  with probability at most `x ^ M`, for `x ≤ 1`. It is `FlockSoundness.Audit.Law.horizon_escape`'s induction on the
  period's game tree: sampling block `i`'s weight is fixed by the strategy that plays it, before its coins.

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

/-- **One sampling block, then the rest: the union step.** If the rest of the run has `p` with probability at most `c`
from every end of `g` outside `q`, the run has `p` with probability at most `g`'s probability of `q`, plus `c`. -/
theorem value_runBind_le_add (f : α → Game δ) (k : (a : α) → Strategy (f a) → Game γ) (p : γ → Prop)
    (q : α → Prop) (c : ℝ≥0∞) :
    ∀ (g : Game α) (s : Strategy (g.bind f)),
      AtEnds f (fun a s' => ¬ q a → value p (k a s') ≤ c) g s →
      value p (runBind f k g s) ≤ prob q g (Strategy.ofBind f g s) + c
  | .ret a, s, h => by
      classical
      show value p (k a s) ≤ (if q a then 1 else 0) + c
      by_cases hq : q a
      · rw [if_pos hq]
        exact (value_le_one p _).trans le_self_add
      · rw [if_neg hq, zero_add]
        exact h hq
  | .send _ kk, s, h => value_runBind_le_add f k p q c (kk s.1) s.2 h
  | @Game.coin _ C _ _ kk, s, h => by
      have hc : ∀ x, value p (runBind f k (kk x) (s x)) ≤ prob q (kk x) (Strategy.ofBind f (kk x) (s x)) + c :=
        fun x => value_runBind_le_add f k p q c (kk x) (s x) (h x)
      calc value p (runBind f k (Game.coin C kk) s)
          = ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * value p (runBind f k (kk x) (s x)) := value_coin _ _ _
        _ ≤ ∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (prob q (kk x) (Strategy.ofBind f (kk x) (s x)) + c) :=
            Finset.sum_le_sum fun x _ => by gcongr; exact hc x
        _ = (∑ x : C, (Fintype.card C : ℝ≥0∞)⁻¹ * prob q (kk x) (Strategy.ofBind f (kk x) (s x))) +
              (∑ _x : C, (Fintype.card C : ℝ≥0∞)⁻¹) * c := by
            rw [Finset.sum_mul, ← Finset.sum_add_distrib]
            exact Finset.sum_congr rfl fun x _ => mul_add _ _ _
        _ = prob q (Game.coin C kk) (Strategy.ofBind f (Game.coin C kk) s) + c := by
            rw [avg_inv_card, one_mul]
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

/-- **The union bound over a period's sampling blocks.** -/
theorem period_union {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ)
    (bad : ρ → Prop) (good : (i : ℕ) → Strategy (G i) → Prop) (ε : ℕ → ℝ≥0∞)
    (hstage : ∀ i τ, good i τ → prob (fun b => bad (rec i τ b)) (G i) τ ≤ ε i) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)), PeriodGood G good i T s →
      value (fun l => ∃ r ∈ l, bad r) (periodRun G rec i T s) ≤ ∑ t ∈ Finset.range T, ε (i + t)
  | 0, i, s, _ => by
      classical
      show value (fun l => ∃ r ∈ l, bad r) (Game.ret []) ≤ _
      simp
  | T + 1, i, s, hs => by
      have h := value_runBind_le_add (fun _ => periodG G (i + 1) T)
        (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
        (fun l => ∃ r ∈ l, bad r) (fun b => bad (rec i (Strategy.ofBind _ (G i) s) b))
        (∑ t ∈ Finset.range T, ε (i + 1 + t)) (G i) s
        (atEnds_mono (fun b s' hg hnb => by
          rw [value_map]
          refine (value_mono (fun l hl => ?_) _).trans (period_union G rec bad good ε hstage T (i + 1) s' hg)
          obtain ⟨r, hr, hbr⟩ := hl
          rcases List.mem_cons.1 hr with rfl | hr
          · exact absurd hbr hnb
          · exact ⟨r, hr, hbr⟩) (G i) s hs.2)
      calc value (fun l => ∃ r ∈ l, bad r) (periodRun G rec i (T + 1) s)
          ≤ prob (fun b => bad (rec i (Strategy.ofBind _ (G i) s) b)) (G i) (Strategy.ofBind _ (G i) s) +
              ∑ t ∈ Finset.range T, ε (i + 1 + t) := h
        _ ≤ ε i + ∑ t ∈ Finset.range T, ε (i + 1 + t) := by gcongr; exact hstage i _ hs.1
        _ = ∑ t ∈ Finset.range (T + 1), ε (i + t) := by
            rw [Finset.sum_range_succ', Nat.add_zero, add_comm]
            congr 1
            exact Finset.sum_congr rfl fun t _ => by rw [Nat.add_assoc, Nat.add_comm 1 t]

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

end FlockSoundness.Game
