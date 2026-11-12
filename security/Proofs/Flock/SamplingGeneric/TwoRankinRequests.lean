module

public import Proofs.Flock.SamplingGeneric.TwoRankin

/-!
# Rankin's bound on the patterns' count, at requests that read several replay blocks

`AdmissibleTwoRankinPatterns` (`Definitions.Claims.SamplingTwoRankin`) takes a split `req` of the served outputs into
requests each reading the proof blocks of one replay block. Its inequality holds at every split: the hypothesis is not
needed. Write `T_r` for the proof blocks of a set `T` that request `r` reads, and `T ∩ ρ` for `T`'s piece in replay
block `ρ` (`blocksIn`).

* `blocksTo_inter`: a request's read of a piece is its read's piece, `(T ∩ ρ)_r = T_r ∩ ρ`.
* `lightOf_eq_some_none_iff`, `lightOf_some_none_of_inter`, `eq_of_lightOf_eq`: a class is `some none` exactly at a
  nonempty set committing at least `n` bits, so a set with such a piece has that class; and two sets of one class other
  than `some none` are equal.
* `lightOf_eq_of_inter`: **the pieces' classes fix the class**. Two sets whose pieces in every replay block have the
  same class have the same class: a piece of class `some none` makes both `some none`, and with none, each piece's
  class records the piece, and a set is the union of its pieces (`biUnion_inter_blocksIn`).
* `lightBits_le_sum_inter`: **a class costs at most its pieces' classes**, `min(b(U), n) ≤ Σ_ρ min(b(U ∩ ρ), n)`
  (`lightBits_lightOf`, `bitsOf_eq_sum`, `min_sum_le_sum_min`).
* `admissibleTwo_rankin_patterns_requests`: **`AdmissibleTwoRankinPatterns`' inequality at every split**. Each
  admissible pattern `p` is the pattern of some admissible set `T_p`. Its weight is at most the product of the weights
  of `T_p`'s pieces' patterns, so at most `δ^{−θ}` times the product of their terms `2^{bits}·passes^θ`
  (`weight_le_rankin`), each at most its pattern's term in the replay block's patterns' factor. `p ↦ (ρ ↦ the pattern
  of T_p ∩ ρ)` is injective, since the pieces' patterns fix the pattern, into the product of the replay blocks'
  patterns, over which the sum of the product is the product of the sums (`Finset.prod_univ_sum`).
* `admissibleTwoRankinPatterns_of_requests`: `AdmissibleTwoRankinPatterns` follows, its hypothesis on the requests
  unread.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

/-- **A request's read of a piece is its read's piece**: `(T ∩ B)_r = T_r ∩ B`. -/
theorem blocksTo_inter {C : Prog} {Pt : Partition} {R : Type} (req : Fin C.outs.length → R) (T B : Finset ℕ) (r : R) :
    blocksTo C Pt req (T ∩ B) r = blocksTo C Pt req T r ∩ B := by
  ext v
  constructor
  · intro h
    obtain ⟨hv, hr⟩ := (mem_blocksTo req).1 h
    exact mem_inter.2 ⟨(mem_blocksTo req).2 ⟨(mem_inter.1 hv).1, hr⟩, (mem_inter.1 hv).2⟩
  · intro h
    obtain ⟨hv, hB⟩ := mem_inter.1 h
    obtain ⟨hvT, hr⟩ := (mem_blocksTo req).1 hv
    exact (mem_blocksTo req).2 ⟨mem_inter.2 ⟨hvT, hB⟩, hr⟩

/-- A class is `some none` exactly at a nonempty set committing at least `n` bits. -/
theorem lightOf_eq_some_none_iff {s : AuditStatement} {Pt : Partition} {n : ℕ} {U : Finset ℕ} :
    lightOf s Pt n U = some none ↔ U ≠ ∅ ∧ n ≤ bitsOf s Pt U := by
  unfold lightOf
  split_ifs with h₁ h₂ <;> simp [h₁, h₂]

/-- A set with a piece of class `some none` has class `some none`: `b` is a sum over the set. -/
theorem lightOf_some_none_of_inter {s : AuditStatement} {Pt : Partition} {n : ℕ} {U B : Finset ℕ}
    (h : lightOf s Pt n (U ∩ B) = some none) : lightOf s Pt n U = some none := by
  rw [lightOf_eq_some_none_iff] at h ⊢
  refine ⟨fun h0 => h.1 (by rw [h0, empty_inter]), h.2.trans ?_⟩
  unfold bitsOf
  exact sum_le_sum_of_subset inter_subset_left

/-- Two sets of one class other than `some none` are equal: `none` is the empty set's, and `some (some U)` records
`U`. -/
theorem eq_of_lightOf_eq {s : AuditStatement} {Pt : Partition} {n : ℕ} {U U' : Finset ℕ}
    (h : lightOf s Pt n U = lightOf s Pt n U') (hl : lightOf s Pt n U ≠ some none) : U = U' := by
  match hc : lightOf s Pt n U with
  | none => exact (lightOf_eq_none hc).trans (lightOf_eq_none (h.symm.trans hc)).symm
  | some none => exact absurd hc hl
  | some (some V) => exact (lightOf_eq_some_some hc).trans (lightOf_eq_some_some (h.symm.trans hc)).symm

/-- **The pieces' classes fix the class**: in scope, two sets of proof blocks whose pieces in every replay block have
the same class have the same class. A piece of class `some none` makes both sets' class `some none`; with none, the
pieces are equal, and a set is the union of its pieces. -/
theorem lightOf_eq_of_inter {s : AuditStatement} {Pt Pρ : Partition}
    (hcov : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome) (n : ℕ) {U U' : Finset ℕ}
    (hU : U ⊆ range Pt.blocks.length) (hU' : U' ⊆ range Pt.blocks.length)
    (h : ∀ i : Fin Pρ.blocks.length,
      lightOf s Pt n (U ∩ blocksIn Pt Pρ i) = lightOf s Pt n (U' ∩ blocksIn Pt Pρ i)) :
    lightOf s Pt n U = lightOf s Pt n U' := by
  by_cases hheavy : ∃ i : Fin Pρ.blocks.length, lightOf s Pt n (U ∩ blocksIn Pt Pρ i) = some none
  · obtain ⟨i, hi⟩ := hheavy
    rw [lightOf_some_none_of_inter hi, lightOf_some_none_of_inter ((h i).symm.trans hi)]
  · have hpieces : ∀ i : Fin Pρ.blocks.length, U ∩ blocksIn Pt Pρ i = U' ∩ blocksIn Pt Pρ i := fun i =>
      eq_of_lightOf_eq (h i) fun hi => hheavy ⟨i, hi⟩
    rw [← biUnion_inter_blocksIn hcov hU, ← biUnion_inter_blocksIn hcov hU']
    exact biUnion_congr rfl fun i _ => hpieces i

/-- **A class costs at most its pieces' classes**: in scope, `min(b(U), n) ≤ Σ_ρ min(b(U ∩ ρ), n)`. -/
theorem lightBits_le_sum_inter (s : AuditStatement) {Pt Pρ : Partition}
    (hcov : ∀ v < Pt.blocks.length, (replayOf Pt Pρ v).isSome) (n : ℕ) {U : Finset ℕ}
    (hU : U ⊆ range Pt.blocks.length) :
    lightBits s Pt n (lightOf s Pt n U) ≤
      ∑ i : Fin Pρ.blocks.length, lightBits s Pt n (lightOf s Pt n (U ∩ blocksIn Pt Pρ i)) := by
  simp only [lightBits_lightOf]
  rw [bitsOf_eq_sum s hcov hU]
  exact min_sum_le_sum_min _ _ _

/-- **`AdmissibleTwoRankinPatterns`' inequality at every split of the served outputs into requests**: on a statement
in scope whose first stage misses each set `T` of replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, at rates
`q_ρ ≤ 1`, at every `δ > 0`, `θ ≥ 0` and split `req`, the admissible sets' patterns' sum `Σ_p 2^{Σ_r bits(p r)}` is at
most `δ^{−θ}` times the product of the replay blocks' patterns' factors. A request may read the proof blocks of any
number of replay blocks. -/
theorem admissibleTwo_rankin_patterns_requests (s : AuditStatement) (Pt : Partition) (d : Design.Two) (q : ℕ → ℝ)
    (δ θ : ℝ) (R : Type) [Fintype R] [DecidableEq R] (req : Fin s.prog.outs.length → R) (hs : InScopeTwo Pt d s)
    (hq : ∀ T ⊆ Finset.range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) (hq1 : ∀ ρ, q ρ ≤ 1)
    (hδ : 0 < δ) (hθ : 0 ≤ θ) :
    ((∑ p ∈ (admissibleTwo Pt d δ).image
          (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
        2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤
      δ ^ (-θ) * ∏ ρ ∈ Finset.range d.part.blocks.length, rankinFactorPatterns s Pt d req (q ρ) θ ρ := by
  classical
  have hcov := hs.2.2.2.1
  let pat : Finset ℕ → R → Option (Option (Finset ℕ)) := fun U r =>
    lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)
  let wt : (R → Option (Option (Finset ℕ))) → ℕ := fun c => ∑ r, lightBits s Pt (servedBits req r) (c r)
  let Φ : Finset ℕ → Fin d.part.blocks.length → R → Option (Option (Finset ℕ)) := fun T i =>
    pat (T ∩ blocksIn Pt d.part i)
  let cls : Fin d.part.blocks.length → (R → Option (Option (Finset ℕ))) → Finset (Finset ℕ) := fun i c =>
    (blocksIn Pt d.part i).powerset.filter fun U => pat U = c
  let g : Fin d.part.blocks.length → (R → Option (Option (Finset ℕ))) → ℝ := fun i c =>
    (2 : ℝ) ^ wt c * ⨆ U : cls i c, d.passes Pt (q i) (counts Pt d.part U.1) i ^ θ
  have hg0 : ∀ i c, 0 ≤ g i c := fun i c =>
    mul_nonneg (by positivity) (Real.iSup_nonneg fun U => Real.rpow_nonneg (d.passes_nonneg Pt (hq1 i) _ _) θ)
  have hfix : ∀ T ⊆ range Pt.blocks.length, ∀ T' ⊆ range Pt.blocks.length, Φ T = Φ T' → pat T = pat T' := by
    intro T hT T' hT' h
    funext r
    show lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r) =
      lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T' r)
    refine lightOf_eq_of_inter hcov _ ((blocksTo_subset s.prog Pt req T r).trans hT)
      ((blocksTo_subset s.prog Pt req T' r).trans hT') fun i => ?_
    rw [← blocksTo_inter, ← blocksTo_inter]
    exact congr_fun (congr_fun h i) r
  have hwt : ∀ T ⊆ range Pt.blocks.length, wt (pat T) ≤ ∑ i, wt (Φ T i) := by
    intro T hT
    simp only [wt, Φ, pat]
    rw [sum_comm]
    refine sum_le_sum fun r _ => ?_
    simp only [blocksTo_inter]
    exact lightBits_le_sum_inter s hcov _ ((blocksTo_subset s.prog Pt req T r).trans hT)
  have hrep : ∀ p ∈ (admissibleTwo Pt d δ).image pat, ∃ T ∈ admissibleTwo Pt d δ, pat T = p := fun p hp =>
    mem_image.1 hp
  choose! rep hrepA hrepP using hrep
  let Ψ : (R → Option (Option (Finset ℕ))) → Fin d.part.blocks.length → R → Option (Option (Finset ℕ)) :=
    fun p => Φ (rep p)
  have key : ∀ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p ≤ δ ^ (-θ) * ∏ i, g i (Ψ p i) := by
    intro p hp
    have hT := hrepA p hp
    have hTN := mem_powerset.1 (mem_filter.1 hT).1
    have hle : wt p ≤ ∑ i : Fin d.part.blocks.length, wt (pat (rep p ∩ blocksIn Pt d.part i)) := by
      have h := hwt (rep p) hTN
      rw [hrepP p hp] at h
      exact h
    have hx : (2 : ℝ) ^ wt p ≤
        ∏ i : Fin d.part.blocks.length, (2 : ℝ) ^ wt (pat (rep p ∩ blocksIn Pt d.part i)) := by
      rw [prod_pow_eq_pow_sum]
      exact pow_le_pow_right₀ one_le_two hle
    refine (weight_le_rankin hq hq1 hδ hθ (w := fun _ U => (2 : ℝ) ^ wt (pat U))
      (fun _ _ => pow_nonneg zero_le_two _) hT hx).trans ?_
    refine mul_le_mul_of_nonneg_left (prod_le_prod₀ (fun i _ => mul_nonneg (pow_nonneg zero_le_two _)
      (Real.rpow_nonneg (d.passes_nonneg Pt (hq1 i) _ _) θ)) fun i _ => ?_) (Real.rpow_nonneg hδ.le _)
    refine mul_le_mul_of_nonneg_left ?_ (pow_nonneg zero_le_two _)
    have hmem : rep p ∩ blocksIn Pt d.part i ∈ cls i (Ψ p i) :=
      mem_filter.2 ⟨mem_powerset.2 inter_subset_right, rfl⟩
    exact le_ciSup (f := fun U : cls i (Ψ p i) => d.passes Pt (q i) (counts Pt d.part U.1) i ^ θ)
      (Set.finite_range _).bddAbove ⟨_, hmem⟩
  have hinj : Set.InjOn Ψ ((admissibleTwo Pt d δ).image pat) := by
    intro p hp p' hp' h
    have hp₁ := mem_coe.1 hp
    have hp₂ := mem_coe.1 hp'
    exact (hrepP p hp₁).symm.trans ((hfix _ (mem_powerset.1 (mem_filter.1 (hrepA p hp₁)).1) _
      (mem_powerset.1 (mem_filter.1 (hrepA p' hp₂)).1) h).trans (hrepP p' hp₂))
  have hpi : ((admissibleTwo Pt d δ).image pat).image Ψ ⊆
      Fintype.piFinset fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset.image pat := by
    intro x hx
    obtain ⟨p, -, rfl⟩ := mem_image.1 hx
    exact Fintype.mem_piFinset.2 fun i => mem_image.2
      ⟨rep p ∩ blocksIn Pt d.part i, mem_powerset.2 inter_subset_right, rfl⟩
  push_cast
  show ∑ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p ≤ _
  calc ∑ p ∈ (admissibleTwo Pt d δ).image pat, (2 : ℝ) ^ wt p
      ≤ ∑ p ∈ (admissibleTwo Pt d δ).image pat, δ ^ (-θ) * ∏ i, g i (Ψ p i) := sum_le_sum key
    _ = δ ^ (-θ) * ∑ x ∈ ((admissibleTwo Pt d δ).image pat).image Ψ, ∏ i, g i (x i) := by
        rw [sum_image hinj, mul_sum]
    _ ≤ δ ^ (-θ) * ∑ x ∈ Fintype.piFinset
          (fun i : Fin d.part.blocks.length => (blocksIn Pt d.part i).powerset.image pat), ∏ i, g i (x i) :=
        mul_le_mul_of_nonneg_left (sum_le_sum_of_subset_of_nonneg hpi fun x _ _ => prod_nonneg fun i _ => hg0 i (x i))
          (Real.rpow_nonneg hδ.le _)
    _ = δ ^ (-θ) * ∏ ρ ∈ range d.part.blocks.length, rankinFactorPatterns s Pt d req (q ρ) θ ρ := by
        rw [← prod_univ_sum, ← Fin.prod_univ_eq_prod_range]
        rfl

/-- **`AdmissibleTwoRankinPatterns`** follows from the inequality at every split: its hypothesis that each request
reads the proof blocks of one replay block is not read. -/
theorem admissibleTwoRankinPatterns_of_requests : AdmissibleTwoRankinPatterns :=
  fun s Pt d q δ θ R _ _ req hs hq hq1 hδ hθ _ =>
    admissibleTwo_rankin_patterns_requests s Pt d q δ θ R req hs hq hq1 hδ hθ

end Claims.Sampling
