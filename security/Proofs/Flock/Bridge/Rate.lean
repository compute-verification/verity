module

public import Proofs.Flock.Bridge.Draw

/-! A sampling design at a rate samples at least that rate of its units. -/

@[expose] public section

namespace Flock.Bridge

def RateOn (num den : ℕ) (samplingDesign : Flock.Draw.SamplingDesign) (n : ℕ) : Prop :=
  match samplingDesign with
  | .subset k => num * n ≤ den * k
  | .bernoulli _ _ => False
  | .stratified K St => ∀ s ∈ St.toList, num * s.size ≤ den * Flock.Draw.stratumK K s.size n
  | .work K ws => ∀ s ∈ ws.toList,
      num * s.size ≤ den * Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor

theorem rateOn_subset (num den tables n : ℕ) (hd : 0 < den) (ht : 0 < tables) :
    RateOn num den (.subset (Spot.Rate.count num den tables n)) n :=
  Spot.Rate.count_ge num den tables n hd ht

theorem rateOn_stratified {num den : ℕ} {St : Array Flock.Draw.Stratum} {ns : List ℕ}
    (hsz : St.toList.map (·.size) = ns) (hn : ∀ x ∈ ns, 1 ≤ x) (hd : 0 < den) (hq : num ≤ den) :
    RateOn num den (.stratified (Spot.Rate.stratK num den ns ns.sum) St) ns.sum := by
  show ∀ s ∈ St.toList, num * s.size ≤ den * Flock.Draw.stratumK (Spot.Rate.stratK num den ns ns.sum) s.size ns.sum
  intro s hs
  have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
  exact Spot.Rate.stratK_ge num den s.size ns.sum ns hm (hn _ hm) (List.le_sum_of_mem hm) hd hq

theorem rateOn_work {num den K n : ℕ} {ws : Array Flock.Draw.WorkStratum} (hd : 0 < den) (hq : num ≤ den)
    (hf : ∀ s ∈ ws.toList, ∃ f, s.floor = Spot.Rate.floor num den s.size f) :
    RateOn num den (.work K ws) n := by
  show ∀ s ∈ ws.toList,
    num * s.size ≤ den * Flock.Draw.workK K s.work s.size (Flock.Draw.totalWork ws) s.floor
  intro s hs
  obtain ⟨f, hf⟩ := hf s hs
  rw [hf]
  exact Spot.Rate.workK_ge num den s.size f K s.work (Flock.Draw.totalWork ws) hd hq

theorem sum_mul_le {α : Type} (f g : α → ℕ) (a b : ℕ) :
    ∀ l : List α, (∀ s ∈ l, a * f s ≤ b * g s) → a * (l.map f).sum ≤ b * (l.map g).sum
  | [], _ => by simp
  | s :: l, h => by
    simp only [List.map_cons, List.sum_cons, Nat.mul_add]
    exact Nat.add_le_add (h s (by simp)) (sum_mul_le f g a b l fun s hs => h s (List.mem_cons_of_mem _ hs))

theorem rateOn_count {samplingDesign : Flock.Draw.SamplingDesign} {ns : List ℕ} {kd num den : ℕ} (h : SamplingDesignOn samplingDesign ns kd)
    (hr : RateOn num den samplingDesign ns.sum) : num * ns.sum ≤ den * kd := by
  cases samplingDesign with
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

theorem rate_le_share {num den n kd : ℕ} (h : num * n ≤ den * kd) (hd : 0 < den) (hn : 0 < n) :
    (num : ℝ) / den ≤ (kd : ℝ) / n := by
  rw [div_le_div_iff₀ (by exact_mod_cast hd) (by exact_mod_cast hn), mul_comm (kd : ℝ)]
  exact_mod_cast h

theorem rateOn_share {samplingDesign : Flock.Draw.SamplingDesign} {ns : List ℕ} {kd num den : ℕ} (h : SamplingDesignOn samplingDesign ns kd)
    (hr : RateOn num den samplingDesign ns.sum) (hd : 0 < den) (hn : 0 < ns.sum) :
    (num : ℝ) / den ≤ (kd : ℝ) / (ns.sum : ℕ) :=
  rate_le_share (rateOn_count h hr) hd hn

end Flock.Bridge
