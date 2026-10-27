import Definitions.Pouw.Assumptions.PerCoin
import Proofs.Pouw.Prob

/-!
# TT per coin: its finder, and TT's budget form from it

* `finderQueries_length`: the finder `B^A` queries exactly as often as `A` does, so it runs in about `A`'s time.
* `tt_budget_of_perCoin`: `TTPerCoin` gives `TT`'s budget form at every adversary whose online cost is at most `T`
  on every coin, with the error `ε(q_A, N)` plus the probability that `H` collides on the finder's log.
-/

namespace Pouw.Proofs

open Finset

variable {Q R S : Type}

theorem queries_length {α : Type} (H : Q → R) : ∀ t : QTree Q R α, (t.queries H).length = t.calls H
  | .done _ => rfl
  | .query q k => by simp [QTree.queries, QTree.calls, queries_length H (k (H q))]

theorem finderQueries_length {CM : CostModel Q R S} (hQ : CM.Queries) (A : Adversary CM) (U : Workload)
    (H : Q → R) (s : S) :
    (finderQueries hQ A U H s).length = A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s := by
  have h := (Classical.choose_spec (hQ A.online U A.act (A.pre.eval H) s) H).2
  simp only [finderQueries, CostModel.tree, List.length_append, queries_length, h]

/-- **TT's budget form from TT per coin**: for an adversary whose online cost is at most `T` on every coin, the
probability that its correct-checked units' `m·k·n` work exceeds `T / (1 − γ₀)` is at most `ε(q_A, N)` plus the
probability that `H` collides on its finder's log. -/
theorem tt_budget_of_perCoin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {P : Protocol Q R S} {D : Layout → Prop} {γ₀ : ℝ} {ε : ℕ → ℕ → ℝ} (hγ₀ : γ₀ < 1)
    (h : Assumptions.TTPerCoin CM P D γ₀ ε) :
    ∃ hQ : CM.Queries, ∀ U : Workload, U.InDomain D → ∀ T : ℝ, ∀ A : Adversary CM,
      (∀ (H : Q → R) (s : S), CM.cost A.online H U A.act (A.pre.eval H) s ≤ T) →
      pr (fun x : (Q → R) × S =>
          T / (1 - γ₀) < ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ))
        ≤ ε (A.queries U) U.layout.N +
          pr (fun x : (Q → R) × S => Collides x.1 (finderQueries hQ A U x.1 x.2)) := by
  obtain ⟨hQ, h⟩ := h
  refine ⟨hQ, fun U hU T A hA => ?_⟩
  let E : (Q → R) × S → Prop := fun x =>
    CM.cost A.online x.1 U A.act (A.pre.eval x.1) x.2 <
        (1 - γ₀) * ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ) ∧
      ¬ Collides x.1 (finderQueries hQ A U x.1 x.2)
  let C : (Q → R) × S → Prop := fun x => Collides x.1 (finderQueries hQ A U x.1 x.2)
  refine le_trans (pr_mono (P' := fun x => E x ∨ C x) ?_) ((pr_or_le E C).trans (add_le_add_right (h U hU A) _))
  rintro ⟨H, s⟩ hx
  dsimp only at hx
  by_cases hc : C (H, s)
  · exact Or.inr hc
  · refine Or.inl ⟨?_, hc⟩
    dsimp only
    have h1 : 0 < 1 - γ₀ := by linarith
    rw [div_lt_iff₀ h1] at hx
    have := hA H s
    nlinarith [hx, this]

end Pouw.Proofs
