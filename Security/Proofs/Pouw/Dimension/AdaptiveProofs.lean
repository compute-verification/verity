import Proofs.Pouw.Dimension.GrindProofs

/-!
# A program chosen from the coins, in `MH100w` (proofs; draft semantics, pending review)

`MH100w`'s programs are straight-line and fixed before the coins. A program that reads coin bits and branches is, at
each grid point `x`, the straight-line program `prog x` it chose there, run on every grid point where it makes the same
choice: its cell `{y | prog y = prog x}`. `DistinctWritesMH100Floor` holds for every set of grid points, so it applies
cell by cell (`adaptive_floor`): the chosen program pays `c` for each target that stays pairwise distinct and non-free
on its cell.

With `lost_le`'s window conditions (`adaptive_lost`), a cell of measure `q^−g` loses at most `g·M + |Ec|` targets, `M`
the most words whose window holds one block of digits: the adaptive bound is `c·(|Ω| − g·M − |Ec|)` at every point.

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

open Classical in
/-- **The adaptive bound for affine targets**: on the coin digits `ι → Fin q`, a well-formed `MH100w` program chosen from
the coins, whose choice at `x` holds every target word on its cell, pays at `x` at least `c·(|Ω| − g·M − |Ec|)` when
that cell has measure at least `q^−g`, under `lost_le`'s window conditions. Each digit's worth of reading can merge
up to `M` words, so the bound is empty once a program has read `|Ω|/M` digits' worth. -/
theorem adaptive_lost {ι : Type} [Fintype ι] [DecidableEq ι] {q : ℕ} (hq : 1 < q)
    (hDW : DistinctWritesMH100Floor) {c : ℕ} {pr : Prices100} (hc : Floor c pr)
    {K U E : Type} [DecidableEq K] [DecidableEq U] [DecidableEq E]
    (blk : ι → K) (ord : K → ℕ) (hord : Function.Injective ord)
    (Ω : Finset U) (ν : U → ℕ) (hν : Set.InjOn ν Ω) (wc : U → ℚ) (wa : U → ι → ℚ) (win : U → Finset K)
    (hwin : ∀ u ∈ Ω, ∀ u' ∈ Ω, ν u' < ν u → ∃ k, blk k ∈ win u ∧ wa u k ≠ wa u' k ∧
      ∀ k', ord (blk k) < ord (blk k') → wa u k' = wa u' k')
    (hwin0 : ∀ u ∈ Ω, ∃ k, blk k ∈ win u ∧ wa u k ≠ 0 ∧ ∀ k', ord (blk k) < ord (blk k') → wa u k' = 0)
    (Ec : Finset E) (ec : E → ℚ) (ea : E → ι → ℚ)
    (F : Set ((ι → Fin q) → ℚ)) (hF : ∀ f ∈ F, (∃ c : ℚ, f = fun _ => c) ∨ ∃ e ∈ Ec, f = Ssn.aff (ec e) (ea e))
    (hFc : ∀ r : ℚ, (fun _ : ι → Fin q => r) ∈ F)
    (M : ℕ) (hM : ∀ b, (Ω.filter fun u => b ∈ win u).card ≤ M)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List ((ι → Fin q) → ℚ)) (hL : ∀ f ∈ L, f ∈ F)
    (prog : (ι → Fin q) → List (Op100 (ι → Fin q))) (hwf : ∀ x, ∀ o ∈ prog x, o.WFw pr F)
    (hout : ∀ x, OutputsOn fpSem {y | prog y = prog x} (prog x) L (Ω.image fun u => Ssn.aff (wc u) (wa u)))
    (x : ι → Fin q) (g : ℕ)
    (hcell : q ^ Fintype.card ι ≤ q ^ g * (Finset.univ.filter fun y => prog y = prog x).card) :
    c * (Ω.card - (g * M + Ec.card)) ≤ cost100 pr (prog x) := by
  obtain ⟨D, hDΩ, hDd, hDc, hΩ⟩ := Grind.lost_le hq _ g hcell blk ord hord Ω ν hν wc wa win hwin hwin0 Ec ec ea F
    hF M hM
  have hcoe : (↑(Finset.univ.filter fun y => prog y = prog x) : Set (ι → Fin q)) = {y | prog y = prog x} := by
    ext y; simp
  rw [hcoe] at hDd
  have h := adaptive_floor hDW hc hFc fpSem L hL prog hwf _ hout x _ (Finset.image_subset_image hDΩ) hDd
  rw [hDc] at h
  exact le_trans (Nat.mul_le_mul_left c (by omega)) h

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
