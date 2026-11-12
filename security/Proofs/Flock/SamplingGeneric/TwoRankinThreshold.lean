module

public import Proofs.Flock.SamplingGeneric.TwoRankin

/-!
# The patterns' factor is monotone in the requests' thresholds

`rankinFactorPatterns` (`Definitions.Claims.SamplingTwoRankin`) reads request `r`'s class at the threshold
`servedBits req r`. `rankinFactorPatternsAt` is the same sum at any thresholds `n : R → ℕ`, and it only grows when the
thresholds of the requests that read in the replay block grow.

* `coarsen`, `coarsen_lightOf`: a set's class at a threshold fixes its class at any lower one.
* `lightBits_lightOf_mono`: a class's bits `min(b(U), n)` grow with the threshold.
* `rankinFactorPatternsAt_mono`: **the factor is monotone in the thresholds**. A class `c` at the lower thresholds is
  the union of the classes `c'` at the higher ones that coarsen to it. Its bits are at most each `c'`'s, and the
  largest pass probability over its sets is the largest over those `c'`, at most their sum.
* `rankinFactorPatterns_le_at`: the patterns' factor of requests each serving at most `S` bits is at most the factor
  at the threshold `S`.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-- **The patterns' factor at thresholds `n`**: `rankinFactorPatterns` with request `r`'s class read at `n r` in place of
`servedBits req r`. -/
noncomputable def rankinFactorPatternsAt (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (n : R → ℕ) (q θ : ℝ) (ρ : ℕ) : ℝ :=
  ∑ c ∈ (blocksIn Pt d.part ρ).powerset.image
      (fun U r => lightOf s Pt (n r) (blocksTo s.prog Pt req U r)),
    (2 : ℝ) ^ (∑ r, lightBits s Pt (n r) (c r)) *
      ⨆ U : ((blocksIn Pt d.part ρ).powerset.filter
          fun U => (fun r => lightOf s Pt (n r) (blocksTo s.prog Pt req U r)) = c),
        d.passes Pt q (counts Pt d.part U.1) ρ ^ θ

theorem rankinFactorPatternsAt_servedBits (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type}
    [Fintype R] [DecidableEq R] (req : Fin s.prog.outs.length → R) (q θ : ℝ) (ρ : ℕ) :
    rankinFactorPatternsAt s Pt d req (servedBits req) q θ ρ = rankinFactorPatterns s Pt d req q θ ρ := rfl

/-- A class at a higher threshold, read at the threshold `n`. -/
def coarsen (s : AuditStatement) (Pt : Partition) (n : ℕ) : Option (Option (Finset ℕ)) → Option (Option (Finset ℕ))
  | none => none
  | some none => some none
  | some (some A) => if n ≤ bitsOf s Pt A then some none else some (some A)

/-- **A set's class at a threshold fixes its class at a lower one.** -/
theorem coarsen_lightOf (s : AuditStatement) (Pt : Partition) {n n' : ℕ} (V : Finset ℕ)
    (h : V.Nonempty → n ≤ n') : coarsen s Pt n (lightOf s Pt n' V) = lightOf s Pt n V := by
  unfold lightOf
  by_cases h0 : V = ∅
  · simp [h0, coarsen]
  · have hle := h (nonempty_iff_ne_empty.2 h0)
    by_cases h2 : n' ≤ bitsOf s Pt V
    · have h1 : n ≤ bitsOf s Pt V := hle.trans h2
      simp [h0, h1, h2, coarsen]
    · by_cases h1 : n ≤ bitsOf s Pt V <;> simp [h0, h1, h2, coarsen]

/-- **A class's bits grow with the threshold.** -/
theorem lightBits_lightOf_mono (s : AuditStatement) (Pt : Partition) {n n' : ℕ} (V : Finset ℕ)
    (h : V.Nonempty → n ≤ n') :
    lightBits s Pt n (lightOf s Pt n V) ≤ lightBits s Pt n' (lightOf s Pt n' V) := by
  rw [lightBits_lightOf, lightBits_lightOf]
  rcases V.eq_empty_or_nonempty with h0 | h0
  · subst h0
    simp [bitsOf]
  · exact min_le_min le_rfl (h h0)

/-- **The patterns' factor is monotone in the thresholds** of the requests that read in the replay block. -/
theorem rankinFactorPatternsAt_mono (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) {n n' : R → ℕ} {q θ : ℝ} (ρ : ℕ) (hq1 : q ≤ 1)
    (hn : ∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty → n r ≤ n' r) :
    rankinFactorPatternsAt s Pt d req n q θ ρ ≤ rankinFactorPatternsAt s Pt d req n' q θ ρ := by
  classical
  let P : Finset (Finset ℕ) := (blocksIn Pt d.part ρ).powerset
  let pat : (R → ℕ) → Finset ℕ → R → Option (Option (Finset ℕ)) := fun m U r =>
    lightOf s Pt (m r) (blocksTo s.prog Pt req U r)
  let wt : (R → ℕ) → (R → Option (Option (Finset ℕ))) → ℕ := fun m c => ∑ r, lightBits s Pt (m r) (c r)
  let cls : (R → ℕ) → (R → Option (Option (Finset ℕ))) → Finset (Finset ℕ) := fun m c =>
    P.filter fun U => pat m U = c
  let g : (R → ℕ) → (R → Option (Option (Finset ℕ))) → ℝ := fun m c =>
    ⨆ U : cls m c, d.passes Pt q (counts Pt d.part U.1) ρ ^ θ
  let co : (R → Option (Option (Finset ℕ))) → R → Option (Option (Finset ℕ)) := fun c r =>
    coarsen s Pt (n r) (c r)
  have hf0 : ∀ U : Finset ℕ, 0 ≤ d.passes Pt q (counts Pt d.part U) ρ ^ θ := fun U =>
    Real.rpow_nonneg (d.passes_nonneg Pt hq1 _ _) θ
  have hg0 : ∀ m c, 0 ≤ g m c := fun m c => Real.iSup_nonneg fun U => hf0 U.1
  have hV : ∀ U ∈ P, ∀ r, (blocksTo s.prog Pt req U r).Nonempty → n r ≤ n' r := fun U hU r h =>
    hn r (h.mono (blocksTo_mono req (mem_powerset.1 hU) r))
  have hco : ∀ U ∈ P, co (pat n' U) = pat n U := fun U hU =>
    funext fun r => coarsen_lightOf s Pt _ (hV U hU r)
  have hsup : ∀ c, g n c ≤ ∑ c' ∈ (P.image (pat n')).filter (fun c' => co c' = c), g n' c' := by
    intro c
    show (⨆ U : cls n c, d.passes Pt q (counts Pt d.part U.1) ρ ^ θ) ≤ _
    refine Real.iSup_le (fun U => ?_) (sum_nonneg fun c' _ => hg0 n' c')
    obtain ⟨U, hU⟩ := U
    obtain ⟨hUP, hUc⟩ := mem_filter.1 hU
    have hmem : pat n' U ∈ (P.image (pat n')).filter (fun c' => co c' = c) :=
      mem_filter.2 ⟨mem_image_of_mem _ hUP, (hco U hUP).trans hUc⟩
    have hmemU : U ∈ cls n' (pat n' U) := mem_filter.2 ⟨hUP, rfl⟩
    have hle : d.passes Pt q (counts Pt d.part U) ρ ^ θ ≤ g n' (pat n' U) :=
      le_ciSup (f := fun V : cls n' (pat n' U) => d.passes Pt q (counts Pt d.part V.1) ρ ^ θ)
        (Set.finite_range _).bddAbove ⟨U, hmemU⟩
    exact hle.trans (single_le_sum (f := fun c' => g n' c') (fun c' _ => hg0 n' c') hmem)
  have hw : ∀ c' ∈ P.image (pat n'), (2 : ℝ) ^ wt n (co c') ≤ (2 : ℝ) ^ wt n' c' := by
    intro c' hc'
    obtain ⟨U, hU, rfl⟩ := mem_image.1 hc'
    refine pow_le_pow_right₀ one_le_two (sum_le_sum fun r _ => ?_)
    show lightBits s Pt (n r) (coarsen s Pt (n r) (lightOf s Pt (n' r) (blocksTo s.prog Pt req U r))) ≤
      lightBits s Pt (n' r) (lightOf s Pt (n' r) (blocksTo s.prog Pt req U r))
    rw [coarsen_lightOf s Pt _ (hV U hU r)]
    exact lightBits_lightOf_mono s Pt _ (hV U hU r)
  show ∑ c ∈ P.image (pat n), (2 : ℝ) ^ wt n c * g n c ≤ ∑ c' ∈ P.image (pat n'), (2 : ℝ) ^ wt n' c' * g n' c'
  calc ∑ c ∈ P.image (pat n), (2 : ℝ) ^ wt n c * g n c
      ≤ ∑ c ∈ P.image (pat n), ∑ c' ∈ (P.image (pat n')).filter (fun c' => co c' = c),
          (2 : ℝ) ^ wt n' c' * g n' c' := by
        refine sum_le_sum fun c _ => ?_
        have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ wt n c := pow_nonneg zero_le_two _
        refine (mul_le_mul_of_nonneg_left (hsup c) h2).trans ?_
        rw [mul_sum]
        refine sum_le_sum fun c' hc' => ?_
        obtain ⟨hc'I, hc'c⟩ := mem_filter.1 hc'
        rw [← hc'c]
        exact mul_le_mul_of_nonneg_right (hw c' hc'I) (hg0 n' c')
    _ = ∑ c' ∈ P.image (pat n'), (2 : ℝ) ^ wt n' c' * g n' c' :=
        sum_fiberwise_of_maps_to (g := co) (fun c' hc' => by
          obtain ⟨U, hU, rfl⟩ := mem_image.1 hc'
          exact mem_image.2 ⟨U, hU, (hco U hU).symm⟩) fun c' => (2 : ℝ) ^ wt n' c' * g n' c'

/-- **Requests serving at most `S` bits**: the patterns' factor is at most the factor at the threshold `S`. -/
theorem rankinFactorPatterns_le_at (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) {q θ : ℝ} (ρ : ℕ) (hq1 : q ≤ 1) {S : ℕ}
    (hS : ∀ r, (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty → servedBits req r ≤ S) :
    rankinFactorPatterns s Pt d req q θ ρ ≤ rankinFactorPatternsAt s Pt d req (fun _ => S) q θ ρ :=
  rankinFactorPatternsAt_mono s Pt d req (n := servedBits req) (n' := fun _ => S) ρ hq1 hS

end Claims.Sampling
