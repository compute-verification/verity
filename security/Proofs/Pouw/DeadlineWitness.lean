module

public import Proofs.Pouw.Deadline
public import Proofs.Pouw.Witness
import Proofs.Pouw.Prob

/-!
# The deadline guarantees say something (witnesses)

`Pouw.Guarantees.WorkBetweenSaltAndDeadline` and `Pouw.Guarantees.SpareCapacityAtDeadline`:

* honest: the milestone's sixteen units are in `D1.m1Domain`, and the honest reference in the output-priced model, on
  the run that delivers at the deadline `1` and spends the milestone's `16·2^44` over `(0, 1]`
  (`Witness.Deadline.honestRun`), meets the honest clock with `A' = A` and the budget of `16·2^44` and `16·m1Queries`
  oracle calls;
* nontrivial: a protocol that compares nothing, on one unit of shape `0×0×0`, makes every committed unit correct, so the
  audit accepts everything. An adversary that commits that unit at no cost, on a run that spends nothing and delivers
  on time, passes one sample with probability 1, above the bound `1/2` at `ε_s = 1/2` and error 0.

The instances are `Definitions.Pouw.Witness.Honest`'s.
-/

@[expose] public section

namespace Pouw.SecurityProofs

open Finset Pouw.Proofs Pouw.Witness.Deadline

namespace DeadlineWitness

theorem honestRun_spentIn (H : D1.M1Q → D1.M1R) (s : Salt) :
    honestRun.spentIn H s honestRun.ts 1 = 16 * 2 ^ 44 := by
  simp [honestRun, TimedRun.spentIn]

theorem honest_cost_le (H : D1.M1Q → D1.M1R) (s : Salt) :
    (outputPriced D1.M1Q D1.M1R Salt).cost (pricedHonest (D1.m1 D1.m1Wref)).online H M1.sixteen
      (pricedHonest (D1.m1 D1.m1Wref)).act ((pricedHonest (D1.m1 D1.m1Wref)).pre.eval H) s ≤ 16 * 2 ^ 44 :=
  (outputPriced_cost_le _ H M1.sixteen _ _ s).trans sum_Wmm_sixteen.le

theorem honest_honestClock :
    Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
      (pricedHonest (D1.m1 D1.m1Wref)) honestRun 1 :=
  Pouw.CostModel.honestClock_iff.2 ⟨pricedHonest (D1.m1 D1.m1Wref), rfl, fun H s =>
    ⟨fun _ => ⟨rfl, fun _ hm => hm⟩, (honest_cost_le H s).trans (honestRun_spentIn H s).ge, le_rfl⟩⟩

theorem honest_boundedBy :
    (pricedHonest (D1.m1 D1.m1Wref)).BoundedBy M1.sixteen honestRun 1 (16 * 2 ^ 44)
      (M1.sixteen.layout.N * D1.m1Queries) :=
  fun H s => ⟨(honestRun_spentIn H s).le,
    (pricedHonest_bounded (D1.m1 D1.m1Wref) (M1.queryBound D1.m1Wref) M1.sixteen H s).2⟩

theorem dot_correct (H : Unit → Unit) (s : Unit) (u : Fin 1) :
    (u : ℕ) ∈ correctSet acceptAll dot H s (commitAll.transcript dot H s) := by
  classical
  unfold correctSet
  rw [Finset.mem_filter]
  exact ⟨Finset.mem_range.2 u.2, ⟨0, 0, 0⟩, rfl,
    ⟨fun _ h => absurd h (Nat.not_lt_zero _), fun x hx => absurd hx (Finset.notMem_empty x)⟩,
    fun _ h => absurd h (Nat.not_lt_zero _)⟩

theorem dot_totalWref : totalWref acceptAll dot.layout = 1 := by
  simp [totalWref, acceptAll, dot]

theorem dot_certified : (0 : ℝ) < (1 - reductionGap 0 1) * (1 - 1 / 2) * totalWref acceptAll dot.layout := by
  rw [dot_totalWref]
  norm_num [reductionGap]

end DeadlineWitness

open DeadlineWitness

/-- **`WorkBetweenSaltAndDeadline`'s premises hold at the milestone**: its sixteen units are in the domain, and the
honest reference in the output-priced model, on the run that delivers at the deadline, is within the budget and meets
the honest clock. -/
theorem WorkBetweenSaltAndDeadline_honest :
    M1.sixteen.InDomain D1.m1Domain ∧
      (pricedHonest (D1.m1 D1.m1Wref)).BoundedBy M1.sixteen honestRun 1 (16 * 2 ^ 44)
        (M1.sixteen.layout.N * D1.m1Queries) ∧
      Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
        (pricedHonest (D1.m1 D1.m1Wref)) honestRun 1 :=
  ⟨M1.sixteen_inDomain, honest_boundedBy, honest_honestClock⟩

/-- **`SpareCapacityAtDeadline`'s premises hold at the milestone**: its sixteen units are in the domain, and the honest
reference in the output-priced model meets the honest clock on the run that delivers at the deadline. -/
theorem SpareCapacityAtDeadline_honest :
    M1.sixteen.InDomain D1.m1Domain ∧
      Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
        (pricedHonest (D1.m1 D1.m1Wref)) honestRun 1 :=
  ⟨M1.sixteen_inDomain, honest_honestClock⟩

/-- **`WorkBetweenSaltAndDeadline`'s bound fails where the audit accepts everything**: one sample passes with
probability 1, above `(1 − 1/2)^1 + 0`, while the run spends nothing. -/
theorem WorkBetweenSaltAndDeadline_nontrivial :
    ¬ pr (fun x : (Unit → Unit) × Unit × (Fin 1 → Ticket (acceptAll.Wref dot.layout) dot.layout.N) =>
        AcceptsBy idleRun 0 x.1 x.2.1 (correctSet acceptAll dot x.1 x.2.1 (commitAll.transcript dot x.1 x.2.1))
            x.2.2 ∧
          (0 : ℝ) < (1 - reductionGap 0 1) * (1 - 1 / 2) * totalWref acceptAll dot.layout ∧
          ¬ commitAll.Breaks dot x.1 x.2.1)
      ≤ (1 - 1 / 2) ^ 1 + 0 := by
  rw [pr_eq_one, not_le]
  · norm_num
  · intro x
    exact ⟨⟨le_rfl, fun i => dot_correct x.1 x.2.1 (x.2.2 i).1⟩, dot_certified,
      not_breaks_of_hashless freeModel_hashless _ _ _ _⟩

/-- **`SpareCapacityAtDeadline`'s bound fails where the audit accepts everything**: with no capacity and no spare work,
one sample passes with probability 1, above `(1 − 1/2)^1 + 0`. -/
theorem SpareCapacityAtDeadline_nontrivial :
    ¬ pr (fun x : (Unit → Unit) × Unit × (Fin 1 → Ticket (acceptAll.Wref dot.layout) dot.layout.N) =>
        AcceptsBy idleRun 0 x.1 x.2.1 (correctSet acceptAll dot x.1 x.2.1 (commitAll.transcript dot x.1 x.2.1))
            x.2.2 ∧
          0 * (0 - idleRun.ts) - (1 - reductionGap 0 1) * (1 - 1 / 2) * totalWref acceptAll dot.layout < 0 ∧
          ¬ commitAll.Breaks dot x.1 x.2.1)
      ≤ (1 - 1 / 2) ^ 1 + 0 := by
  rw [pr_eq_one, not_le]
  · norm_num
  · intro x
    exact ⟨⟨le_rfl, fun i => dot_correct x.1 x.2.1 (x.2.2 i).1⟩, by linarith [dot_certified],
      not_breaks_of_hashless freeModel_hashless _ _ _ _⟩

end Pouw.SecurityProofs
