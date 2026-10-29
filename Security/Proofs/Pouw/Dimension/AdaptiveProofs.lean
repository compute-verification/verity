import Definitions.Pouw.Dimension.Ssn

/-!
# A program chosen from the coins, in `MH100w` (proofs; draft semantics, pending review)

`MH100w`'s programs are straight-line and fixed before the coins. A program that reads coin bits and branches is, at
each grid point `x`, the straight-line program `prog x` it chose there, run on every grid point where it makes the same
choice: its cell `{y | prog y = prog x}`. `DistinctWritesMH100Floor` holds for every set of grid points, so it applies
cell by cell (`adaptive_floor`): the chosen program pays `c` for each target that stays pairwise distinct and non-free
on its cell.

What reading costs is not here: a branch or predicate per coin bit read is priced separately (`adaptive_total`). The
targets that merge on a cell are the design's: a word that equals another on every point of a cell, or a free value
there, earns nothing at the points of that cell. For served sparse noise (`ssnWordB_eq_of_zero`), a row's words after
two steps with equal data constants merge on the cell where the bits between them are 0: reading one bit merges a
row's `n` words of a step into the step before on half the points.
-/

namespace Pouw.Dimension.H100

variable {G : Type}

/-- **The floor, cell by cell**: a well-formed `MH100w` program chosen at each grid point from the coins, whose choice at
`x` holds every target of `T` on the points where it is chosen, pays at `x` at least `c` for each target of a subset
`D` pairwise distinct and non-free on that cell. -/
theorem adaptive_floor (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {F : Set (G → ℚ)} (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
    (hL : ∀ f ∈ L, f ∈ F) (prog : G → List (Op100 G)) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (T : Finset (G → ℚ)) (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L T)
    (x : G) (D : Finset (G → ℚ)) (hD : D ⊆ T) (hDd : DistinctOn {y | prog y = prog x} F D) :
    c * D.card ≤ cost100 pr (prog x) :=
  hDW c pr hc G F hF {y | prog y = prog x} fpSem L (prog x) D hL (hwf x) hDd fun t ht => hout x t (hD ht)

/-- **With the reads priced**: if choosing at `x` costs `read x` (its branches and predicates), the total at `x` is at
least `read x` plus `c` for each target distinct and non-free on the cell. -/
theorem adaptive_total (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {F : Set (G → ℚ)} (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ))
    (hL : ∀ f ∈ L, f ∈ F) (prog : G → List (Op100 G)) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (T : Finset (G → ℚ)) (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L T) (read : G → ℕ)
    (x : G) (D : Finset (G → ℚ)) (hD : D ⊆ T) (hDd : DistinctOn {y | prog y = prog x} F D) :
    read x + c * D.card ≤ read x + cost100 pr (prog x) :=
  Nat.add_le_add_left (adaptive_floor hDW hc hF fpSem L hL prog hwf T hout x D hD hDd) _

end Pouw.Dimension.H100

namespace Pouw.Dimension.Ssn

variable {m S C L : ℕ}

/-- **Two of a row's words merge where the bits they weigh differently are 0**: if row `i`'s words of column `j` after
steps `t` and `t'` have equal data constants, they are equal at every coin point where each bit with different weights
in the two is 0. For the 4-lane layout and `t' = t − 1` that is the one bit of step `t`'s lane, so a program that reads
it merges the row's `n` words of step `t` into step `t − 1`'s on half the points. -/
theorem ssnWordB_eq_of_zero (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ)
    (t t' : Fin S) (hκ : κ i j t = κ i j t') (x : Bits m C)
    (hx : ∀ k, bitCoef src used pos W i j t k ≠ bitCoef src used pos W i j t' k → x k = 0) :
    ssnWordB src used pos κ W i j t x = ssnWordB src used pos κ W i j t' x := by
  unfold ssnWordB aff
  rw [hκ]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases h : bitCoef src used pos W i j t k = bitCoef src used pos W i j t' k
  · rw [h]
  · simp [hx k h]

end Pouw.Dimension.Ssn
