module

public import Proofs.Pouw.Deadline
public import Proofs.Pouw.Witness
import Proofs.Pouw.Prob

/-!
# The deadline guarantees say something (witnesses)

`Pouw.Guarantees.WorkBetweenSaltAndDeadline` and `Pouw.Guarantees.SpareCapacityAtDeadline`:

* honest: the milestone's sixteen units are in `D1.m1Domain`; the witnesses' Site (`Witness.Clock.site`, drift `0`)
  passes the clock's check, its device meets `clock-within-drift` with real time equal to the reading, and its entries
  at readings `0` and `1` are in time at the deadline `Δ = 1`; and the honest reference in the output-priced model, on
  the run that spends the milestone's `16·2^44` over `(0, 1]` (`Witness.Deadline.honestRun`), meets the honest clock
  with `A' = A` and the budget of `16·2^44` and `16·m1Queries` oracle calls;
* nontrivial: a protocol that compares nothing, on one unit of shape `0×0×0`, makes every committed unit correct, so the
  audit accepts everything. An adversary that commits that unit at no cost, on a run that spends nothing, with the
  device's two entries at one reading, passes one sample with probability 1, above the bound `1/2` at `ε_s = 1/2` and
  error 0.

The instances are `Definitions.Pouw.Witness.Honest`'s.
-/

@[expose] public section

namespace Pouw.SecurityProofs

open Finset Pouw.Proofs Pouw.Witness.Deadline

namespace DeadlineWitness

theorem honestRun_spentIn (H : D1.M1Q → D1.M1R) (s : Salt) :
    honestRun.spentIn H s ((Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1).ts s)
      ((Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1).ts s + realWindow Witness.Clock.site Witness.Clock.key 1) =
      16 * 2 ^ 44 := by
  rw [Witness.Clock.stamps_ts, Witness.Clock.realWindow_site]
  simp [honestRun, TimedRun.spentIn]

theorem honest_cost_le (H : D1.M1Q → D1.M1R) (s : Salt) :
    (outputPriced D1.M1Q D1.M1R Salt).cost (pricedHonest (D1.m1 D1.m1Wref)).online H M1.sixteen
      (pricedHonest (D1.m1 D1.m1Wref)).act ((pricedHonest (D1.m1 D1.m1Wref)).pre.eval H) s ≤ 16 * 2 ^ 44 :=
  (outputPriced_cost_le _ H M1.sixteen _ _ s).trans sum_Wmm_sixteen.le

theorem honest_honestClock :
    Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
      (pricedHonest (D1.m1 D1.m1Wref)) (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1)
      honestRun (realWindow Witness.Clock.site Witness.Clock.key 1) :=
  Pouw.CostModel.honestClock_iff.2 ⟨pricedHonest (D1.m1 D1.m1Wref), rfl, fun H s =>
    ⟨fun _ _ => ⟨rfl, fun _ hm => hm⟩, (honest_cost_le H s).trans (honestRun_spentIn H s).ge, le_rfl⟩⟩

theorem honest_boundedBy :
    (pricedHonest (D1.m1 D1.m1Wref)).BoundedBy M1.sixteen (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1) honestRun
      (realWindow Witness.Clock.site Witness.Clock.key 1) (16 * 2 ^ 44) (M1.sixteen.layout.N * D1.m1Queries) :=
  fun H s => ⟨(honestRun_spentIn H s).le,
    (pricedHonest_bounded (D1.m1 D1.m1Wref) (M1.queryBound D1.m1Wref) M1.sixteen H s).2⟩

/-- The witnesses' clock: the Site passes the check, its device meets `clock-within-drift`, and its entries at
readings `0` and `1` are in time at `Δ = 1` for every trace. -/
theorem honest_clock :
    Verity.Assumptions.ClockWithin Witness.Clock.site Witness.Clock.key
        (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1).real ∧
      (∃ q, clockDrift Witness.Clock.site Witness.Clock.key = .ok q) ∧
      ∀ (H : D1.M1Q → D1.M1R) (s : Salt) (e : Trace),
        InTime (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1) 1 H s e :=
  ⟨Witness.Clock.clockWithin_real _ _, Witness.Clock.site_ok,
    fun H s e => Witness.Clock.inTime 0 1 1 (by decide) (by decide) H s e⟩

theorem dot_correct (H : Unit → Unit) (s : Unit) (u : Fin 1) :
    (u : ℕ) ∈ correctSet acceptAll dot H s (commitAll.trace dot H s) := by
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

/-- **`WorkBetweenSaltAndDeadline`'s premises hold at the milestone**: its sixteen units are in the domain; the
witnesses' Site passes the clock's check, its device meets `clock-within-drift`, and the device's entries at readings
`0` and `1` are in time at the deadline `Δ = 1`; and the honest reference in the output-priced model is within the
budget and meets the honest clock in the real window. -/
theorem WorkBetweenSaltAndDeadline_honest :
    M1.sixteen.InDomain D1.m1Domain ∧
      (Verity.Assumptions.ClockWithin Witness.Clock.site Witness.Clock.key
          (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1).real ∧
        (∃ q, clockDrift Witness.Clock.site Witness.Clock.key = .ok q) ∧
        ∀ (H : D1.M1Q → D1.M1R) (s : Salt) (e : Trace),
          InTime (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1) 1 H s e) ∧
      (pricedHonest (D1.m1 D1.m1Wref)).BoundedBy M1.sixteen (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1) honestRun
        (realWindow Witness.Clock.site Witness.Clock.key 1) (16 * 2 ^ 44) (M1.sixteen.layout.N * D1.m1Queries) ∧
      Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
        (pricedHonest (D1.m1 D1.m1Wref)) (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1)
        honestRun (realWindow Witness.Clock.site Witness.Clock.key 1) :=
  ⟨M1.sixteen_inDomain, honest_clock, honest_boundedBy, honest_honestClock⟩

/-- **`SpareCapacityAtDeadline`'s premises hold at the milestone**: its sixteen units are in the domain, the witnesses'
clock holds as in `WorkBetweenSaltAndDeadline_honest`, and the honest reference in the output-priced model meets the
honest clock in the real window. -/
theorem SpareCapacityAtDeadline_honest :
    M1.sixteen.InDomain D1.m1Domain ∧
      (Verity.Assumptions.ClockWithin Witness.Clock.site Witness.Clock.key
          (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1).real ∧
        (∃ q, clockDrift Witness.Clock.site Witness.Clock.key = .ok q) ∧
        ∀ (H : D1.M1Q → D1.M1R) (s : Salt) (e : Trace),
          InTime (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1) 1 H s e) ∧
      Assumptions.HonestClock ((outputPriced D1.M1Q D1.M1R Salt).adversaries M1.sixteen)
        (pricedHonest (D1.m1 D1.m1Wref)) (Witness.Clock.docket Salt Trace) (Witness.Clock.stamps D1.M1Q D1.M1R Salt 0 1)
        honestRun (realWindow Witness.Clock.site Witness.Clock.key 1) :=
  ⟨M1.sixteen_inDomain, honest_clock, honest_honestClock⟩

/-- **`WorkBetweenSaltAndDeadline`'s bound fails where the audit accepts everything**: one sample passes with
probability 1, above `(1 − 1/2)^1 + 0`, while the run spends nothing. -/
theorem WorkBetweenSaltAndDeadline_nontrivial :
    ¬ pr (fun x : (Unit → Unit) × Unit × (Fin 1 → Ticket (acceptAll.Wref dot.layout) dot.layout.N) =>
        AcceptsBy Witness.Clock.site (Witness.Clock.docket Unit Trace) (Witness.Clock.stamps Unit Unit Unit 0 0) 0 x.1
            x.2.1 (commitAll.trace dot x.1 x.2.1) (correctSet acceptAll dot x.1 x.2.1 (commitAll.trace dot x.1 x.2.1))
            x.2.2 ∧
          (0 : ℝ) < (1 - reductionGap 0 1) * (1 - 1 / 2) * totalWref acceptAll dot.layout ∧
          ¬ commitAll.Breaks dot x.1 x.2.1)
      ≤ (1 - 1 / 2) ^ 1 + 0 := by
  rw [pr_eq_one, not_le]
  · norm_num
  · intro x
    exact ⟨⟨Witness.Clock.site_ok, Witness.Clock.inTime 0 0 0 le_rfl le_rfl _ _ _,
      fun i => dot_correct x.1 x.2.1 (x.2.2 i).1⟩, dot_certified,
      not_breaks_of_hashless freeModel_hashless _ _ _ _⟩

/-- **`SpareCapacityAtDeadline`'s bound fails where the audit accepts everything**: with no capacity and no spare work,
one sample passes with probability 1, above `(1 − 1/2)^1 + 0`. -/
theorem SpareCapacityAtDeadline_nontrivial :
    ¬ pr (fun x : (Unit → Unit) × Unit × (Fin 1 → Ticket (acceptAll.Wref dot.layout) dot.layout.N) =>
        AcceptsBy Witness.Clock.site (Witness.Clock.docket Unit Trace) (Witness.Clock.stamps Unit Unit Unit 0 0) 0 x.1
            x.2.1 (commitAll.trace dot x.1 x.2.1) (correctSet acceptAll dot x.1 x.2.1 (commitAll.trace dot x.1 x.2.1))
            x.2.2 ∧
          0 * realWindow Witness.Clock.site Witness.Clock.key 0 - (1 - reductionGap 0 1) * (1 - 1 / 2) * totalWref acceptAll dot.layout < 0 ∧
          ¬ commitAll.Breaks dot x.1 x.2.1)
      ≤ (1 - 1 / 2) ^ 1 + 0 := by
  rw [pr_eq_one, not_le]
  · norm_num
  · intro x
    exact ⟨⟨Witness.Clock.site_ok, Witness.Clock.inTime 0 0 0 le_rfl le_rfl _ _ _,
      fun i => dot_correct x.1 x.2.1 (x.2.2 i).1⟩, by linarith [dot_certified],
      not_breaks_of_hashless freeModel_hashless _ _ _ _⟩

end Pouw.SecurityProofs
