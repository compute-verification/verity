module

public import Proofs.Properties.SampledProofs.FactorUpper.Patterns

/-!
# Rankin's factor is below `factorUpper`

`Properties.SampledProofs.FactorUpperSound` (the sets' form, `FactorUpper/Sets.lean`) and `FactorUpperPatternsSound`
(the patterns' form over requests, `FactorUpper/Patterns.lean`), on the arithmetic of `FactorUpper/Bounds.lean`.

Each bound returns at a deployed-shaped replay block, by the kernel's evaluation (`_honest`), and each conclusion fails
for a bound that returns `−1` whatever it reads: at a replay block with no proof blocks both factors are `1`
(`_nontrivial`).
-/

@[expose] public section

namespace Properties.SampledProofs.SecurityProofs

open Claims.Sampling
open _root_.SampledProofs.FactorUpper

/-- **`lnFactorUpper` is above Rankin's factor.** -/
theorem factorUpperSound : FactorUpperSound := FactorUpperProof.factorUpper_sound

/-- **`lnFactorUpperPatterns` is above Rankin's patterns' factor.** -/
theorem factorUpperPatternsSound : FactorUpperPatternsSound := FactorUpperProof.factorUpperPatterns_sound

section Witnesses

namespace FactorUpperProof

/-- A partition with no proof blocks. -/
def emptyPart : Partition := ⟨[]⟩

/-- A one-gate statement. -/
def emptyStmt : AuditStatement := ⟨[], ⟨[.input], [0]⟩, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- A two-stage design over `emptyPart`. -/
def emptyDesign : Design.Two := ⟨emptyPart, ⟨[]⟩, []⟩

theorem emptyPart_covers (ρ : ℕ) : Covers emptyStmt emptyPart emptyDesign ρ [] :=
  ⟨rfl, fun t c h => by simp at h⟩

theorem emptyPart_rankinFactor (q θ : ℝ) (ρ : ℕ) : rankinFactor emptyStmt emptyPart emptyDesign q θ ρ = 1 := by
  simp [rankinFactor, blocksIn, emptyPart, Partition.strata, Design.Two.passes, Design.Two.esc, bitsOf]

theorem iSup_one_of_mem {S : Finset (Finset ℕ)} {x : Finset ℕ} (hx : x ∈ S) : (⨆ _U : S, (1 : ℝ)) = 1 :=
  haveI : Nonempty S := ⟨⟨x, hx⟩⟩
  ciSup_const

theorem emptyPart_rankinFactorPatterns (q θ : ℝ) (ρ : ℕ) :
    rankinFactorPatterns emptyStmt emptyPart emptyDesign (fun _ => ()) q θ ρ = 1 := by
  have hB : blocksIn emptyPart emptyDesign.part ρ = ∅ := by simp [blocksIn, emptyPart, Partition.strata]
  have hE : ∀ r, blocksTo emptyStmt.prog emptyPart (fun _ => ()) ∅ r = ∅ := fun r =>
    Finset.subset_empty.1 (blocksTo_sub_self ∅ r)
  have hP : ∀ fp, emptyDesign.passes emptyPart q fp ρ = 1 := fun fp => by
    simp [Design.Two.passes, Design.Two.esc, emptyPart, Partition.strata]
  unfold rankinFactorPatterns
  rw [hB]
  simp only [Finset.powerset_empty, Finset.image_singleton, Finset.sum_singleton, hE, hP, Real.one_rpow]
  rw [iSup_one_of_mem (x := ∅) ?_]
  · simp [lightOf, lightBits]
  · simp [hE]

theorem exp_neg_one_lt_one : Real.exp (-1) < 1 :=
  calc Real.exp (-1) < Real.exp 0 := Real.exp_lt_exp.2 (by norm_num)
    _ = 1 := Real.exp_zero

/-- The deployed-shaped replay block: one stratum of 64 proof blocks of 16 bits, `k = 4`. -/
def honestCells : List Cell := [⟨4, [(64, 16)]⟩]

/-- `q = 1/2`, `θ = 1`, the chords `optimize.CHORDS` and precision `32`. -/
def honestParams : Params := ⟨1 / 2, 1, chords, 32⟩

end FactorUpperProof

open FactorUpperProof

/-- **`FactorUpperSound`'s honest witness**: the bound returns at a deployed-shaped replay block. -/
theorem factorUpperSound_honest : (lnFactorUpper honestCells honestParams).isSome = true := by
  decide +kernel

/-- **`FactorUpperSound`'s conclusion fails** for a bound of `−1`: Rankin's factor of a replay block with no proof
blocks, which `[]` covers (`emptyPart_covers`), is `1`. -/
theorem factorUpperSound_nontrivial : ¬ rankinFactor emptyStmt emptyPart emptyDesign (1 / 2) 1 0 ≤ Real.exp (-1) := by
  rw [emptyPart_rankinFactor, not_le]
  exact exp_neg_one_lt_one

/-- **`FactorUpperPatternsSound`'s honest witness**: the bound returns at the deployed-shaped replay block with each of
at most 4 requests reading 8 of its proof blocks and serving 64 bits. -/
theorem factorUpperPatternsSound_honest :
    (lnFactorUpperPatterns honestCells [[(8, 16)]] 4 64 honestParams).isSome = true := by
  decide +kernel

/-- **`FactorUpperPatternsSound`'s conclusion fails** for a bound of `−1`: the patterns' factor of a replay block with
no proof blocks is `1`. -/
theorem factorUpperPatternsSound_nontrivial :
    ¬ rankinFactorPatterns emptyStmt emptyPart emptyDesign (fun _ => ()) (1 / 2) 1 0 ≤ Real.exp (-1) := by
  rw [emptyPart_rankinFactorPatterns, not_le]
  exact exp_neg_one_lt_one

end Witnesses

end Properties.SampledProofs.SecurityProofs
