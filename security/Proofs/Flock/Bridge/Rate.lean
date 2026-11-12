module

public import Proofs.Flock.Bridge.Draw

/-!
# A sampling design at a rate samples at least the rate of its units

`LawOn law ns kd` says how many units a sampling design over strata at `ns` samples: `kd`, the sum of its strata's
counts. A selection rule that names a rate fixes each count from the rate and the registered sizes
(`Spot.Select.Rule.over`, `Spot.Select.Rule.tableOver`; `Properties.Spot.RuleAtRate`), so the share `kd / Σ ns` is at
least the rate at whatever sizes the prover registers. This file holds that step:

* `RateOn num den law n`: each of `law`'s counts is at least `num/den` of what it is taken from: a subset's `k` of the
  `n` units, and each stratum's count under a stratified or work design (the rule's, `stratumK` or `workK`) of the
  stratum's units. A Bernoulli design is none;
* `rateOn_subset`, `rateOn_stratified`, `rateOn_work`: the designs a rate names are `RateOn`: `subset k` at
  `k = Spot.Rate.count`, `stratified K` at `K = Spot.Rate.stratK` over the strata's sizes, and `work K` at floors
  `Spot.Rate.floor`;
* `rateOn_count`: under `LawOn law ns kd` and `RateOn num den law (Σ ns)`, `num · Σ ns ≤ den · kd`; `rateOn_share`:
  `num/den ≤ kd / Σ ns` in the reals (`rate_le_share`).

The statements over a period of `SamplingBlock`s take a share `q ≤ kd/n` as a hypothesis and sample a uniform subset,
so they read this at `subset k` alone (`Proofs.Flock.Bridge.RatePeriod`).
-/

@[expose] public section

namespace Flock.Bridge

/-- **A sampling design whose every count is at least `num/den` of what it is taken from**: a subset's `k` of the `n`
units; under a stratified design each stratum's count `stratumK K n_s n` of its `n_s` units; under a work design each
stratum's count `workK` of its units. A Bernoulli design is none. -/
def RateOn (num den : ℕ) (law : Flock.Draw.Law) (n : ℕ) : Prop :=
  match law with
  | .subset k => num * n ≤ den * k
  | .bernoulli _ _ => False
  | .stratified K St => ∀ s ∈ St.toList, num * s.size ≤ den * Flock.Draw.stratumK K s.size n
  | .work K ws => ∀ s ∈ ws.toList,
      num * s.size ≤ den * Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor

/-- A rate's subset, `Spot.Rate.count` of the `n` units, is `RateOn`. -/
theorem rateOn_subset (num den tables n : ℕ) (hd : 0 < den) (ht : 0 < tables) :
    RateOn num den (.subset (Spot.Rate.count num den tables n)) n :=
  Spot.Rate.count_ge num den tables n hd ht

/-- A stratified rate's design, `stratified K` at `K = Spot.Rate.stratK` over strata of sizes `ns`, each at least one
unit, is `RateOn`. -/
theorem rateOn_stratified {num den : ℕ} {St : Array Flock.Draw.Stratum} {ns : List ℕ}
    (hsz : St.toList.map (·.size) = ns) (hn : ∀ x ∈ ns, 1 ≤ x) (hd : 0 < den) (hq : num ≤ den) :
    RateOn num den (.stratified (Spot.Rate.stratK num den ns ns.sum) St) ns.sum := by
  show ∀ s ∈ St.toList, num * s.size ≤ den * Flock.Draw.stratumK (Spot.Rate.stratK num den ns ns.sum) s.size ns.sum
  intro s hs
  have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
  exact Spot.Rate.stratK_ge num den s.size ns.sum ns hm (hn _ hm) (List.le_sum_of_mem hm) hd hq

/-- A work rate's design, each stratum's floor `Spot.Rate.floor` at its own units, is `RateOn`, at any `K`. -/
theorem rateOn_work {num den K n : ℕ} {ws : Array Flock.Draw.WorkStratum} (hd : 0 < den) (hq : num ≤ den)
    (hf : ∀ s ∈ ws.toList, ∃ f, s.floor = Spot.Rate.floor num den s.size f) :
    RateOn num den (.work K ws) n := by
  show ∀ s ∈ ws.toList,
    num * s.size ≤ den * Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor
  intro s hs
  obtain ⟨f, hf⟩ := hf s hs
  rw [hf]
  exact Spot.Rate.workK_ge num den s.size f K s.work (Flock.Draw.totalWork ws) hd hq

/-- Counts each at least `a/b` of their sizes sum to at least `a/b` of the sizes' sum. -/
theorem sum_mul_le {α : Type} (f g : α → ℕ) (a b : ℕ) :
    ∀ l : List α, (∀ s ∈ l, a * f s ≤ b * g s) → a * (l.map f).sum ≤ b * (l.map g).sum
  | [], _ => by simp
  | s :: l, h => by
    simp only [List.map_cons, List.sum_cons, Nat.mul_add]
    exact Nat.add_le_add (h s (by simp)) (sum_mul_le f g a b l fun s hs => h s (List.mem_cons_of_mem _ hs))

/-- **A sampling design at a rate samples at least the rate of its units**: under `LawOn law ns kd`, where every count
of `law` is at least `num/den` of what it is taken from (`RateOn`), `num · Σ ns ≤ den · kd`. -/
theorem rateOn_count {law : Flock.Draw.Law} {ns : List ℕ} {kd num den : ℕ} (h : LawOn law ns kd)
    (hr : RateOn num den law ns.sum) : num * ns.sum ≤ den * kd := by
  cases law with
  | subset k =>
    obtain ⟨rfl, -⟩ := h
    exact hr
  | bernoulli a b => exact h.elim
  | stratified K St =>
    obtain ⟨⟨-, hsz⟩, -, -, hk, hsum⟩ := h
    subst hsz
    subst hsum
    refine sum_mul_le _ _ num den St.toList fun s hs => ?_
    show num * s.size ≤ den * s.k
    rw [hk s hs]
    exact hr s hs
  | work K ws =>
    obtain ⟨⟨-, hsz⟩, hk, hsum⟩ := h
    subst hsz
    subst hsum
    rw [List.map_map]
    refine sum_mul_le _ _ num den ws.toList fun s hs => ?_
    show num * s.size ≤ den * s.k
    rw [hk s hs]
    exact hr s hs

/-- A count `kd` of `n` units, `n` not zero, with `num · n ≤ den · kd`, is at least the share `num/den` of them. -/
theorem rate_le_share {num den n kd : ℕ} (h : num * n ≤ den * kd) (hd : 0 < den) (hn : 0 < n) :
    (num : ℝ) / den ≤ (kd : ℝ) / n := by
  rw [div_le_div_iff₀ (by exact_mod_cast hd) (by exact_mod_cast hn), mul_comm (kd : ℝ)]
  exact_mod_cast h

/-- **The rate as a share**: under `rateOn_count`'s hypotheses, over a population that is not empty,
`num/den ≤ kd / Σ ns`. -/
theorem rateOn_share {law : Flock.Draw.Law} {ns : List ℕ} {kd num den : ℕ} (h : LawOn law ns kd)
    (hr : RateOn num den law ns.sum) (hd : 0 < den) (hn : 0 < ns.sum) :
    (num : ℝ) / den ≤ (kd : ℝ) / (ns.sum : ℕ) :=
  rate_le_share (rateOn_count h hr) hd hn

end Flock.Bridge
