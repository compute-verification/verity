module

public import Definitions.Claims.SamplingTwoRankin
public import Proofs.Flock.Claims.Witnesses.SamplingTwo
public import Proofs.Flock.SamplingGeneric.TwoRankin

/-!
# Rankin's bound on the two-stage count says something

`AdmissibleTwoRankin` and `AdmissibleTwoRankinPatterns` are guarantees that the lock's `claims.generic` names, each held
to an honest and a nontrivial witness.

* `rankinTwo`: `twoStmt`'s two proof blocks in one replay block (`onePart`), its first stage `bernoulli:1/2`, `k = 1`.
  It is in scope, its first stage meets the rates' hypothesis at `q_ρ = 1/2` (`bernoulli_Pa_le`), and a request reading
  both outputs reads in that one replay block (`rankinTwo_honest`, `rankinTwoPatterns_honest`).
* `nothingTwo`: `toyStmt`'s one proof block, its first stage `bernoulli:0/1`, which samples nothing, so `P̄_a = 1` at
  every footprint and both sets of proof blocks are admissible at `δ = 9/10`. Claimed at the rate `1/2` it doesn't meet,
  the bound fails at `θ = 1`: the count is `1 + 2 = 3` and `δ^{−1}·G = (10/9)·(1·1 + 2·(1/2)) = 20/9`
  (`rankinTwo_nontrivial`). With one request reading the one output, the patterns are the sets' and the numbers the same
  (`rankinTwoPatterns_nontrivial`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Finset
open Claims.Sampling
open Claims.Witness.Sampling (toyStmt toyPart toyProg toy_boundary)

/-! ## The honest instance -/

/-- One replay block holding both of `twoPart`'s proof blocks. -/
def onePart : Partition := ⟨[[[[0, 1]]]]⟩

/-- `twoStmt`'s design: its proof blocks in one replay block, sampled `bernoulli:1/2`, `k = 1`. -/
def rankinTwo : Design.Two := ⟨onePart, ⟨[.bernoulli 1 2]⟩, [1]⟩

theorem inScopeTwo_rankin : InScopeTwo twoPart rankinTwo twoStmt := by
  refine ⟨inScope_two.1, inScope_two.2.1, by decide, by decide, ⟨⟨rfl, ?_⟩, by decide, by decide⟩⟩
  intro p hp
  simp only [rankinTwo, onePart, List.zip_cons_cons, List.zip_nil_right, List.mem_singleton] at hp
  subst hp
  simp [OneStage.WellFormed]

/-- **`AdmissibleTwoRankin`'s honest witness**: `rankinTwo` is in scope, and its first stage misses each set of replay
blocks with probability at most `(1/2)^{|T|}`. -/
theorem rankinTwo_honest : InScopeTwo twoPart rankinTwo twoStmt ∧
    ∀ T ⊆ range rankinTwo.part.blocks.length, rankinTwo.replay.Pa rankinTwo.part T ≤ ∏ _ρ ∈ T, (1 - (1 / 2 : ℝ)) := by
  refine ⟨inScopeTwo_rankin, fun T hT => ?_⟩
  have h := (bernoulli_Pa_le inScopeTwo_rankin (num := 1) (den := 2) rfl).1 T hT
  norm_num at h ⊢
  exact h

/-- **`AdmissibleTwoRankinPatterns`'s honest witness**: `rankinTwo_honest`'s, and a request reading both of `twoStmt`'s
outputs reads in its one replay block. -/
theorem rankinTwoPatterns_honest : InScopeTwo twoPart rankinTwo twoStmt ∧
    (∀ T ⊆ range rankinTwo.part.blocks.length, rankinTwo.replay.Pa rankinTwo.part T ≤ ∏ _ρ ∈ T, (1 - (1 / 2 : ℝ))) ∧
    ∀ r : Unit, ∃ ρ, ∀ v ∈ blocksTo twoStmt.prog twoPart (fun _ => ()) (range twoPart.blocks.length) r,
      replayOf twoPart rankinTwo.part v = some ρ := by
  refine ⟨rankinTwo_honest.1, rankinTwo_honest.2, fun r => ⟨0, fun v hv => ?_⟩⟩
  have hv2 : v < 2 := mem_range.1 (blocksTo_subset _ _ _ _ _ hv)
  interval_cases v <;> decide

/-! ## The nontrivial instance -/

/-- `toyStmt`'s design with a first stage that samples nothing (`bernoulli:0/1`). -/
def nothingTwo : Design.Two := ⟨toyPart, ⟨[.bernoulli 0 1]⟩, [1]⟩

theorem nothingTwo_replay_Pa (A : Finset ℕ) : nothingTwo.replay.Pa nothingTwo.part A = 1 := by
  simp [nothingTwo, Design.Pa, toyPart, classesPa, OneStage.Pa]

/-- A first stage that samples nothing accepts at every footprint. -/
theorem nothingTwo_Pa (fp : ℕ → ℕ → ℕ) : nothingTwo.Pa toyPart fp = 1 := by
  unfold Design.Two.Pa
  simp only [nothingTwo_replay_Pa, one_mul]
  rw [← prod_add]
  simp

theorem nothingTwo_admissible : admissibleTwo toyPart nothingTwo (9 / 10) = (range 1).powerset := by
  classical
  unfold admissibleTwo
  refine filter_true_of_mem fun T _ => ?_
  show (9 / 10 : ℝ) < nothingTwo.Pa toyPart _
  rw [nothingTwo_Pa]
  norm_num

theorem toy_blocksIn : blocksIn toyPart nothingTwo.part 0 = {0} := by decide

theorem powerset_zero : ({0} : Finset ℕ).powerset = {∅, {0}} := by decide

theorem toy_bitsOf_zero : bitsOf toyStmt toyPart {0} = 1 := by decide

theorem nothingTwo_passes_empty : nothingTwo.passes toyPart (1 / 2) (counts toyPart nothingTwo.part ∅) 0 = 1 := by
  unfold Design.Two.passes Design.Two.esc
  simp [counts, escCell_zero]

theorem nothingTwo_passes_zero : nothingTwo.passes toyPart (1 / 2) (counts toyPart nothingTwo.part {0}) 0 = 1 / 2 := by
  have h : nothingTwo.esc toyPart (counts toyPart nothingTwo.part {0}) 0 = 0 :=
    toyTwo_esc (fp := counts toyPart nothingTwo.part {0}) (by decide)
  unfold Design.Two.passes
  rw [h]
  norm_num

theorem nothingTwo_rankinFactor : rankinFactor toyStmt toyPart nothingTwo (1 / 2) 1 0 = 2 := by
  unfold rankinFactor
  rw [toy_blocksIn, powerset_zero, sum_pair (by decide), nothingTwo_passes_empty, nothingTwo_passes_zero,
    toy_bitsOf_zero]
  norm_num [bitsOf]

theorem nothingTwo_count : (∑ T ∈ admissibleTwo toyPart nothingTwo (9 / 10), 2 ^ bitsOf toyStmt toyPart T : ℕ) = 3 := by
  rw [nothingTwo_admissible]
  decide

/-- **`AdmissibleTwoRankin`'s conclusion fails** at a first stage that samples nothing, claimed at the rate `1/2`:
`3 > (10/9)·2`. -/
theorem rankinTwo_nontrivial :
    ¬ ((∑ T ∈ admissibleTwo toyPart nothingTwo (9 / 10), 2 ^ bitsOf toyStmt toyPart T : ℕ) : ℝ) ≤
      (9 / 10 : ℝ) ^ (-(1 : ℝ)) *
        ∏ ρ ∈ range nothingTwo.part.blocks.length, rankinFactor toyStmt toyPart nothingTwo (1 / 2) 1 ρ := by
  rw [nothingTwo_count, show nothingTwo.part.blocks.length = 1 from rfl, prod_range_one, nothingTwo_rankinFactor,
    Real.rpow_neg_one]
  norm_num

/-- The one request, reading `toyStmt`'s one output. -/
def toyReq : Fin toyStmt.prog.outs.length → Unit := fun _ => ()

theorem toy_blocksTo : blocksTo toyStmt.prog toyPart toyReq {0} () = {0} := by
  refine subset_antisymm (blocksTo_subset _ _ _ _ _) (singleton_subset_iff.2 ?_)
  exact (mem_blocksTo toyReq).2 ⟨mem_singleton_self 0, 0, by decide, by decide, ⟨0, by decide⟩, rfl, Prog.Reads.refl 0⟩

theorem toy_servedBits : servedBits toyReq () = 1 := by decide

theorem toy_pat_empty :
    (fun r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq ∅ r)) =
      fun _ => none := by
  funext r
  rw [show blocksTo toyStmt.prog toyPart toyReq ∅ r = ∅ from subset_empty.1 (blocksTo_subset _ _ _ _ _)]
  exact lightOf_empty _ _ _

theorem toy_pat_zero :
    (fun r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq {0} r)) =
      fun _ => some none := by
  funext r
  rw [toy_blocksTo, toy_servedBits]
  simp [lightOf, toy_bitsOf_zero]

theorem toy_pat_ne : (fun _ : Unit => (none : Option (Option (Finset ℕ)))) ≠ fun _ => some none := by
  intro h
  exact absurd (congr_fun h ()) (by simp)

/-- **`AdmissibleTwoRankinPatterns`'s conclusion fails** at `nothingTwo`, claimed at the rate `1/2`, with one request
reading the one output: two patterns, `1 + 2 = 3 > (10/9)·2`. -/
theorem rankinTwoPatterns_nontrivial :
    ¬ ((∑ p ∈ (admissibleTwo toyPart nothingTwo (9 / 10)).image
          (fun T r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq T r)),
          2 ^ ∑ r, lightBits toyStmt toyPart (servedBits toyReq r) (p r) : ℕ) : ℝ) ≤
      (9 / 10 : ℝ) ^ (-(1 : ℝ)) *
        ∏ ρ ∈ range nothingTwo.part.blocks.length,
          rankinFactorPatterns toyStmt toyPart nothingTwo toyReq (1 / 2) 1 ρ := by
  have himg : ∀ S : Finset (Finset ℕ), S = {∅, {0}} →
      S.image (fun T r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq T r)) =
        {fun _ => none, fun _ => some none} := by
    rintro S rfl
    rw [image_insert, image_singleton, toy_pat_empty, toy_pat_zero]
  have hcount : (∑ p ∈ (admissibleTwo toyPart nothingTwo (9 / 10)).image
        (fun T r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq T r)),
        2 ^ ∑ r, lightBits toyStmt toyPart (servedBits toyReq r) (p r) : ℕ) = 3 := by
    rw [himg _ (by rw [nothingTwo_admissible]; decide), sum_pair toy_pat_ne]
    simp [lightBits, toy_servedBits]
  have hfac : rankinFactorPatterns toyStmt toyPart nothingTwo toyReq (1 / 2) 1 0 ≤ 2 := by
    unfold rankinFactorPatterns
    rw [himg _ (by rw [toy_blocksIn, powerset_zero]), sum_pair toy_pat_ne]
    have hs : ∑ r : Unit, servedBits toyReq r = 1 := (Fintype.sum_unique _).trans toy_servedBits
    simp only [lightBits, hs, sum_const_zero, pow_zero, one_mul, pow_one]
    have h1 : ⨆ U : ((blocksIn toyPart nothingTwo.part 0).powerset.filter fun U =>
        (fun r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq U r)) =
          fun _ => none), nothingTwo.passes toyPart (1 / 2) (counts toyPart nothingTwo.part U.1) 0 ^ (1 : ℝ) ≤ 1 :=
      Real.iSup_le (fun U => by
        rw [Real.rpow_one]
        exact nothingTwo.passes_le_one toyPart (by norm_num) _ _) zero_le_one
    have h2 : ⨆ U : ((blocksIn toyPart nothingTwo.part 0).powerset.filter fun U =>
        (fun r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq U r)) =
          fun _ => some none), nothingTwo.passes toyPart (1 / 2) (counts toyPart nothingTwo.part U.1) 0 ^ (1 : ℝ) ≤
          1 / 2 := by
      have key : ∀ V ∈ (blocksIn toyPart nothingTwo.part 0).powerset.filter (fun U =>
          (fun r => lightOf toyStmt toyPart (servedBits toyReq r) (blocksTo toyStmt.prog toyPart toyReq U r)) =
            fun _ => some none),
          nothingTwo.passes toyPart (1 / 2) (counts toyPart nothingTwo.part V) 0 ^ (1 : ℝ) ≤ 1 / 2 := by
        intro V hV
        obtain ⟨hU, hpat⟩ := mem_filter.1 hV
        rw [toy_blocksIn, mem_powerset] at hU
        rcases subset_singleton_iff.1 hU with h0 | h0
        · rw [h0, toy_pat_empty] at hpat
          exact absurd hpat toy_pat_ne
        · simp only [h0, Real.rpow_one, nothingTwo_passes_zero, le_refl]
      exact Real.iSup_le (fun U => key U.1 U.2) (by norm_num)
    linarith
  rw [hcount, show nothingTwo.part.blocks.length = 1 from rfl, prod_range_one, Real.rpow_neg_one]
  intro h
  have : (3 : ℝ) ≤ (9 / 10)⁻¹ * 2 :=
    h.trans (mul_le_mul_of_nonneg_left hfac (by norm_num))
  norm_num at this

end Claims.Witness.SamplingGeneric
