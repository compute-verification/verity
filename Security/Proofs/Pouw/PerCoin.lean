import Definitions.Pouw.Guarantees.PerCoin
import Proofs.Pouw.Sanity
import Proofs.Pouw.Theorem1

/-!
# TT per coin: its finder, and TT's budget form from it

* `finderQueries_length`: the finder `B^A` queries exactly as often as `A` does, so it runs in about `A`'s time.
* `tt_budget_of_perCoin`: `TTPerCoin` gives `TT`'s budget form at every adversary whose online cost is at most `T`
  on every coin, with the error `ε(q_A, N)` plus the probability that `H` collides on the finder's log.
* `ttPerCoin_outputPriced` (honest): it holds with error 0 in the witness model `outputPriced`.
* `ttPerCoin_false_of_const` (nontrivial): it fails below error 1 wherever a free constant transcript is correct.
* `gammaFromTTPerCoin`, `endToEndPerCoin`: `G_γ` and resource exhaustion from it, each at the adversary's own cost
  and query count, with the finder's collision outside the event.
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
  refine le_trans (pr_mono (P' := fun x => E x ∨ C x) ?_) ((pr_or_le E C).trans (add_le_add (h U hU A) le_rfl))
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

/-- **TT per coin's honest lemma**: in the witness model it holds with error 0 for every protocol and domain, since
every program costs at least the `m·k·n` work of the units it commits. -/
theorem ttPerCoin_outputPriced [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (D : Layout → Prop) : Assumptions.TTPerCoin (outputPriced Q R S) P D Params.γ0 (fun _ _ => 0) := by
  refine ⟨outputPriced_queries, fun U _ A => le_of_eq (pr_eq_zero fun x hx => ?_)⟩
  obtain ⟨hlt, -⟩ := hx
  have h1 : ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ) ≤
      (outputPriced Q R S).cost A.online x.1 U A.act (A.pre.eval x.1) x.2 :=
    checked_le_outputPriced_cost P A.online x.1 U A.act (A.pre.eval x.1) x.2
  have hS : 0 ≤ ∑ u ∈ checkedSet P U x.1 x.2 (A.transcript U x.1 x.2), ((U.layout.shape u).Wmm : ℝ) :=
    Finset.sum_nonneg fun _ _ => by positivity
  have hγ : (0 : ℝ) ≤ Params.γ0 := by unfold Params.γ0; norm_num
  nlinarith [mul_nonneg hγ hS]

/-- **TT per coin's nontrivial lemma**: if a constant transcript has every unit's checked values correct for every
`H` and `s`, TT per coin fails below error 1 at zero queries, in every model with free constants, on every workload
with positive work. Its finder's log is empty there, so no collision rescues it. -/
theorem ttPerCoin_false_of_const [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] [Nonempty R] [Nonempty S]
    (CM : CostModel Q R S) (hCM : Sanity.FreeConstants CM) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload)
    (hU : U.InDomain D) (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (τ : Transcript)
    (hτ : ∀ H s, checkedSet P U H s τ = range U.layout.N) {γ₀ : ℝ} (hγ₀ : γ₀ < 1) (ε : ℕ → ℕ → ℝ)
    (hTT : Assumptions.TTPerCoin CM P D γ₀ ε) : 1 ≤ ε 0 U.layout.N := by
  obtain ⟨hQ, hTT⟩ := hTT
  obtain ⟨p, hp⟩ := hCM τ
  let A : Adversary CM := ⟨QTree.done [], p, fun _ => 0⟩
  have hcalls : ∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s = 0 :=
    fun H s => by simp [A, QTree.calls, QTree.eval, (hp H U _ [] s).2.2]
  have hq : A.queries U = 0 :=
    Nat.eq_zero_of_le_zero (Finset.sup_le fun x _ => (hcalls x.1 x.2).le)
  have h := hTT U hU A
  rw [hq] at h
  refine le_of_eq_of_le (pr_eq_one fun x => ⟨?_, ?_⟩).symm h
  · show CM.cost p x.1 U (fun _ => 0) [] x.2 <
      (1 - γ₀) * ∑ u ∈ checkedSet P U x.1 x.2 (CM.run p x.1 U (fun _ => 0) [] x.2), ((U.layout.shape u).Wmm : ℝ)
    rw [(hp x.1 U (fun _ => 0) [] x.2).1, hτ]
    have := (hp x.1 U (fun _ => 0) [] x.2).2.1
    have h1 : 0 < 1 - γ₀ := by linarith
    nlinarith [mul_pos h1 hpos]
  · rintro ⟨y, hy, -⟩
    have h0 : (finderQueries hQ A U x.1 x.2).length = 0 := by
      rw [finderQueries_length]; exact hcalls x.1 x.2
    rw [List.eq_nil_of_length_eq_zero h0] at hy
    simp at hy

/-- **(a) per coin**: `G_γ` from `TTPerCoin`, pointwise by `wins_imp_tt` at `T` the adversary's own cost. -/
theorem gammaFromTTPerCoin : Pouw.Guarantees.GammaFromTTPerCoin := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hQ hγ₀ hΩ hW hTT U hU A
  obtain ⟨_, h⟩ := hTT
  refine le_trans (pr_mono ?_) (h U hU A)
  rintro ⟨H, s⟩ ⟨hwin, hc⟩
  refine ⟨?_, hc⟩
  have h1 := wins_imp_tt P U H s _ hΩ (hW U hU) hwin
  rw [div_lt_iff₀ (by linarith : (0 : ℝ) < 1 - γ₀)] at h1
  exact h1.trans_eq (mul_comm _ _)

/-- **(d) per coin**: `TTPerCoin` and the `Ω_max` bound give resource exhaustion with `δ_s = (1 − ε_s)^t` and
`η = ε(A.queries U, N)`. -/
theorem endToEndPerCoin : Pouw.Guarantees.EndToEndPerCoin := by
  intro Q R S _ _ _ _ CM P D γ₀ Ωmax ε hQ hγ₀ hΩ hW hTT U hU A t εs hεs
  have hG := gammaFromTTPerCoin Q R S CM P D γ₀ Ωmax ε hQ hγ₀ hΩ hW hTT U hU A
  let E1 := fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
    AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
      εs * totalWref P U.layout <
        ∑ u ∈ range U.layout.N \ correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1), (P.Wref U.layout u : ℝ)
  let E2 := fun y : (Q → R) × S =>
    Wins P U (reductionGamma γ₀ Ωmax) (CM.cost A.online y.1 U A.act (A.pre.eval y.1) y.2)
        (A.transcript U y.1 y.2) y.1 y.2 ∧
      ¬ Collides y.1 (finderQueries hQ A U y.1 y.2)
  have hE1 : pr E1 ≤ (1 - εs) ^ t := by
    let G := fun y : ((Q → R) × S) × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
      AuditAccepts (correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2)) y.2 ∧
        εs * ∑ u ∈ range U.layout.N, (P.Wref U.layout u : ℝ) <
          ∑ u ∈ range U.layout.N \ correctSet P U y.1.1 y.1.2 (A.transcript U y.1.1 y.1.2), (P.Wref U.layout u : ℝ)
    calc pr E1 = pr G := pr_equiv (Equiv.prodAssoc _ _ _).symm G
      _ ≤ _ := workWeightedSampling ((Q → R) × S) (P.Wref U.layout) U.layout.N t
          (fun y => correctSet P U y.1 y.2 (A.transcript U y.1 y.2)) εs hεs
  calc _ ≤ pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) => E1 x ∨ E2 (x.1, x.2.1)) :=
        pr_mono fun x hx => by
          by_cases h1 : E1 x
          · exact Or.inl h1
          · refine Or.inr ⟨wins_of_accept P U x.1 x.2.1 _ (reductionGamma_lt_one hγ₀ hΩ) hx.2.1 ?_, hx.2.2⟩
            by_contra h2
            exact h1 ⟨hx.1, lt_of_not_ge h2⟩
    _ ≤ pr E1 + pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) => E2 (x.1, x.2.1)) :=
        pr_or_le _ _
    _ ≤ (1 - εs) ^ t + ε (A.queries U) U.layout.N := add_le_add hE1 ((pr_fst2_le E2).trans hG)

end Pouw.Proofs
