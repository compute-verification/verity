module

public import Proofs.Flock.Claims.Sampling.Reg
public import Proofs.Flock.Claims.Prob

/-!
# The refused second stage, in expectation (`RegTwo.ex_checked_le`)

Where the second stage is refused (`Select.selectSecond` errors, as it does on the empty sample a refused first stage
hands it), the checked proof blocks are every block below `N` (`RegTwo.checked_of_refused`). Where it accepts, they are
the always-checked blocks and each proved unit's block, so a weight at most `M` sums over them to at most the
always-checked blocks' weights plus `M` for each proved unit (`RegTwo.sum_checked_le`). A function of the checked blocks
at most `B ≥ 0` wherever the second stage accepts then averages, over any finite space of coins, at most `B` plus the
refusal's probability times its value at every block below `N` (`RegTwo.ex_checked_le`). With the blocks' needs as the
weight and the second stage's count, that is `Σ_always |needs| + C·M + Pr[refused]·Σ_{u<N} |needs u|`
(`RegTwo.ex_sum_checked_le`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims

theorem sum_filter_or_le (xs : List ℕ) (p q : ℕ → Bool) (w : ℕ → ℕ) :
    ((xs.filter fun v => p v || q v).map w).sum ≤ ((xs.filter p).map w).sum + ((xs.filter q).map w).sum := by
  induction xs with
  | nil => simp
  | cons a xs ih =>
    by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq] <;> omega

/-- An average is at most `B ≥ 0` plus an event's probability times `C`, where the function is at most `B` off the
event and at most `C` on it. -/
theorem Ex_le_add_pr_mul {Ω : Type} [Fintype Ω] [Nonempty Ω] (f : Ω → ℝ) (R : Ω → Prop) {B C : ℝ} (hB : 0 ≤ B)
    (hoff : ∀ ω, ¬ R ω → f ω ≤ B) (hon : ∀ ω, R ω → f ω ≤ C) : Ex f ≤ B + Pr R * C := by
  classical
  have hpt : ∀ ω, f ω ≤ B + C * (if R ω then (1 : ℝ) else 0) := by
    intro ω
    by_cases h : R ω
    · simp only [h, ↓reduceIte, mul_one]
      linarith [hon ω h]
    · simp only [h, ↓reduceIte, mul_zero, add_zero]
      exact hoff ω h
  have hind : Ex (fun ω => C * (if R ω then (1 : ℝ) else 0)) = C * Pr R := by
    rw [Pr_eq_sum R]
    unfold Ex
    rw [← Finset.mul_sum, mul_div_assoc]
  calc Ex f ≤ Ex (fun ω => B + C * (if R ω then (1 : ℝ) else 0)) := Ex_mono hpt
    _ = B + C * Pr R := by rw [Ex_add (fun _ => B), Ex_const, hind]
    _ = B + Pr R * C := by ring

/-- Where the second stage is refused, every proof block below `N` is checked. -/
theorem RegTwo.checked_of_refused (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray)
    (h : Except.toBool (Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂) = false) :
    g.checked N src₁ src₂ = List.range N := by
  unfold RegTwo.checked
  cases hs : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ with
  | ok p =>
    rw [hs] at h
    exact absurd h (by simp [Except.toBool])
  | error e => rfl

/-- **The checked blocks where the second stage accepts**: a weight at most `M` sums over them to at most the
always-checked blocks' weights below `N` plus `M` for each proved unit. -/
theorem RegTwo.sum_checked_le (g : RegTwo) (N : ℕ) (src₁ src₂ : ByteArray) {draw : Lean.Json} {n : ℕ}
    (hsel : Spot.Select.selectSecond (g.sample src₁) g.ns g.ks src₂ = .ok (draw, n)) (w : ℕ → ℕ) (M : ℕ)
    (hM : ∀ u, w u ≤ M) :
    ((g.checked N src₁ src₂).map w).sum ≤
      (((List.range N).filter fun v => g.always v).map w).sum + (Spot.Draw.proved draw).size * M := by
  have hc : g.checked N src₁ src₂ = (List.range N).filter fun v =>
      g.always v || (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v := by
    unfold RegTwo.checked
    rw [hsel]
  rw [hc]
  refine (sum_filter_or_le _ _ _ w).trans (Nat.add_le_add_left ?_ _)
  have hs := List.sum_le_length_nsmul (((List.range N).filter fun v =>
      (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v).map w) M (by
    intro y hy
    obtain ⟨v, _, rfl⟩ := List.mem_map.1 hy
    exact hM v)
  rw [List.length_map, smul_eq_mul] at hs
  refine hs.trans (Nat.mul_le_mul_right _ ?_)
  have hsub : ((List.range N).filter fun v =>
      (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v).toFinset ⊆
      ((Spot.Draw.proved draw).toList.map fun j => g.blockAt (g.sample src₁) j.toNat).toFinset := by
    intro v hv
    obtain ⟨-, hq⟩ := List.mem_filter.1 (List.mem_toFinset.1 hv)
    rw [← Array.any_toList] at hq
    obtain ⟨j, hj, hjv⟩ := List.any_eq_true.1 hq
    exact List.mem_toFinset.2 (List.mem_map.2 ⟨j, hj, beq_iff_eq.1 hjv⟩)
  calc ((List.range N).filter fun v =>
        (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v).length
      = ((List.range N).filter fun v =>
          (Spot.Draw.proved draw).any fun j => g.blockAt (g.sample src₁) j.toNat == v).toFinset.card :=
        (List.toFinset_card_of_nodup (List.nodup_range.filter _)).symm
    _ ≤ ((Spot.Draw.proved draw).toList.map fun j => g.blockAt (g.sample src₁) j.toNat).toFinset.card :=
        Finset.card_le_card hsub
    _ ≤ ((Spot.Draw.proved draw).toList.map fun j => g.blockAt (g.sample src₁) j.toNat).length :=
        List.toFinset_card_le _
    _ = (Spot.Draw.proved draw).size := by rw [List.length_map, Array.length_toList]

/-- **The refused second stage, in expectation**: over coins `ω` with first-stage bytes `src₁ ω` and second-stage
bytes `src₂ ω`, a function `F` of the checked blocks at most `B ≥ 0` wherever the second stage accepts averages at most
`B` plus the refusal's probability times `F` at every block below `N`. -/
theorem RegTwo.ex_checked_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (g : RegTwo) (N : ℕ) (src₁ src₂ : Ω → ByteArray)
    (F : List ℕ → ℝ) {B : ℝ} (hB : 0 ≤ B)
    (hok : ∀ ω draw n, Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) = .ok (draw, n) →
      F (g.checked N (src₁ ω) (src₂ ω)) ≤ B) :
    Ex (fun ω => F (g.checked N (src₁ ω) (src₂ ω))) ≤
      B + Pr (fun ω => Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false) *
        F (List.range N) :=
  Ex_le_add_pr_mul _ _ hB
    (fun ω hR => by
      cases hs : Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) with
      | ok p => exact hok ω p.1 p.2 hs
      | error e => exact absurd (by rw [hs]; rfl) hR)
    (fun ω hR => le_of_eq (by rw [g.checked_of_refused N _ _ hR]))

/-- **The checked blocks' weights, in expectation**: for a weight at most `M`, with the second stage proving at most `C`
units wherever it accepts, the checked blocks' weights average at most the always-checked blocks' weights below `N`
plus `C·M`, plus the refusal's probability times every block's weight below `N`. -/
theorem RegTwo.ex_sum_checked_le {Ω : Type} [Fintype Ω] [Nonempty Ω] (g : RegTwo) (N : ℕ)
    (src₁ src₂ : Ω → ByteArray) (w : ℕ → ℕ) (M C : ℕ) (hM : ∀ u, w u ≤ M)
    (hC : ∀ ω draw n, Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω) = .ok (draw, n) →
      (Spot.Draw.proved draw).size ≤ C) :
    Ex (fun ω => (((g.checked N (src₁ ω) (src₂ ω)).map w).sum : ℝ)) ≤
      ((((List.range N).filter fun v => g.always v).map w).sum + C * M : ℕ) +
        Pr (fun ω => Except.toBool (Spot.Select.selectSecond (g.sample (src₁ ω)) g.ns g.ks (src₂ ω)) = false) *
          (((List.range N).map w).sum : ℝ) :=
  g.ex_checked_le N src₁ src₂ (fun l => ((l.map w).sum : ℝ)) (Nat.cast_nonneg _) fun ω draw n hsel => by
    exact_mod_cast (g.sum_checked_le N _ _ hsel w M hM).trans
      (Nat.add_le_add_left (Nat.mul_le_mul_right M (hC ω draw n hsel)) _)

end SamplingDeploy
