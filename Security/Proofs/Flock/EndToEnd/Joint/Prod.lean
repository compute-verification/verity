import Proofs.Flock.EndToEnd.Period.Union

/-!
# A period's run: the product of the escapes

`period_prod`: if at every strategy satisfying `good` a sampling block's records all carry one escape `e`, and the
sampling block's "miss" event has probability at most `e`, then every sampling block misses with the product of their
escapes below `c` with probability at most `c`. It is `period_horizon` with the escapes read off the records: sampling
block `i`'s escape is fixed by the strategy that plays it, before its coins, so the rest of the period has to miss below
`c / e`.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {β : ℕ → Type}

/-- **The joint escape over a period's sampling blocks.** -/
theorem period_prod {ρ : Type} (G : (i : ℕ) → Game (β i)) (rec : (i : ℕ) → Strategy (G i) → β i → ρ)
    (miss : ρ → Prop) (esc : ρ → ℝ≥0∞) (good : (i : ℕ) → Strategy (G i) → Prop)
    (hstage : ∀ i τ, good i τ →
      ∃ e : ℝ≥0∞, e ≠ ⊤ ∧ (∀ b, esc (rec i τ b) = e) ∧ prob (fun b => miss (rec i τ b)) (G i) τ ≤ e) :
    ∀ (T i : ℕ) (s : Strategy (periodG G i T)) (c : ℝ≥0∞), PeriodGood G good i T s →
      value (fun l => (∀ r ∈ l, miss r) ∧ (l.map esc).prod < c) (periodRun G rec i T s) ≤ c
  | 0, i, s, c, _ => by
      classical
      show value (fun l => (∀ r ∈ l, miss r) ∧ (l.map esc).prod < c) (Game.ret []) ≤ _
      rw [value_ret]
      split_ifs with h
      · simpa using h.2.le
      · exact zero_le
  | T + 1, i, s, c, hs => by
      classical
      obtain ⟨e, he, hw, hp⟩ := hstage i _ hs.1
      by_cases he0 : e = 0
      · have h := value_runBind_le_mul (fun _ => periodG G (i + 1) T)
          (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
          (fun l => (∀ r ∈ l, miss r) ∧ (l.map esc).prod < c) (fun b => miss (rec i (Strategy.ofBind _ (G i) s) b))
          1 (G i) s
          (atEnds_mono (fun b s' _ => by
            by_cases hm : miss (rec i (Strategy.ofBind _ (G i) s) b)
            · rw [if_pos hm]
              exact value_le_one _ _
            · rw [if_neg hm, value_map]
              refine (value_mono (q := fun _ => False) (fun l hl => hm (hl.1 _ List.mem_cons_self)) _).trans_eq ?_
              exact value_false _) (G i) s hs.2)
        refine h.trans ?_
        rw [mul_one]
        exact (hp.trans_eq he0).trans (zero_le)
      · have h := value_runBind_le_mul (fun _ => periodG G (i + 1) T)
          (fun b s' => (periodRun G rec (i + 1) T s').map (rec i (Strategy.ofBind _ (G i) s) b :: ·))
          (fun l => (∀ r ∈ l, miss r) ∧ (l.map esc).prod < c) (fun b => miss (rec i (Strategy.ofBind _ (G i) s) b))
          (c / e) (G i) s
          (atEnds_mono (fun b s' hg => by
            by_cases hm : miss (rec i (Strategy.ofBind _ (G i) s) b)
            · rw [if_pos hm, value_map]
              refine (value_mono (fun l hl => ?_) _).trans
                (period_prod G rec miss esc good hstage T (i + 1) s' (c / e) hg)
              obtain ⟨hall, hlt⟩ := hl
              refine ⟨fun r hr => hall r (List.mem_cons_of_mem _ hr), ?_⟩
              simp only [List.map_cons, List.prod_cons, hw b] at hlt
              exact (ENNReal.lt_div_iff_mul_lt (Or.inl he0) (Or.inl he)).2 (by rwa [mul_comm])
            · rw [if_neg hm, value_map]
              refine (value_mono (q := fun _ => False) (fun l hl => hm (hl.1 _ List.mem_cons_self)) _).trans_eq ?_
              exact value_false _) (G i) s hs.2)
        calc value (fun l => (∀ r ∈ l, miss r) ∧ (l.map esc).prod < c) (periodRun G rec i (T + 1) s)
            ≤ prob (fun b => miss (rec i (Strategy.ofBind _ (G i) s) b)) (G i) (Strategy.ofBind _ (G i) s) *
                (c / e) := h
          _ ≤ e * (c / e) := by gcongr
          _ ≤ c := ENNReal.mul_div_le

end FlockSoundness.Game
