import Definitions.Nci.Guarantees.CoverChained
import Proofs.Nci.CoveringTwoCosts
import Proofs.Nci.Cover.Gates
import Proofs.Nci.Cover.Training

/-!
# The cover theorem on a compliant declaration, counting the chained multiplication blocks, proved

`chained_le_of_blocks` is the covering bound's count (`Nci.SecurityProofs.CoveringBoundTwoCosts`) over a family of
blocks each at its own rate: blocks that compute every gate with a predecessor, block `u` handed `k_u ≤ α σ_u` gates
outside `θ` with `σ_u ≤ Γ`, have `c_w c_a |𝓜*| ≤ (α² Γ + c_w c_a α) Σ σ_u`. Each chained block is charged to a block
computing its output gate, each block takes at most `(k_u / c_w)(k_u / c_a) + k_u` of them
(`CoveringTwoCosts.card_charged_le`), and the per-block bounds add up at the NCI policy's rate (`sum_area_le`).

On `c`'s bits as a declaration (`bitDecl c`) the gate view is the bit view (`ofGates_bitDecl`), so the block of a set of
`c`'s gates (`block`) is a subcircuit of `bitDecl c` (`bitBlock`) that is handed what the bit view says and computes
what it says. Pull a compliant division of `d` back to `c` (`Nci.SecurityProofs.pull`): piece `u`'s block has at most
`s` bits and is handed at most `α σ_u` bits outside the free bits (`translation`, `card_block_le`), and every bit with a
predecessor is computed by the block of the piece holding its gate (`mem_computed_block_of_pred`). So
`chained_le_of_blocks` applies, with `Σ σ_u = size(d)`, and the slowdown form follows (`slowdown_of_count`).
-/

namespace Nci.Cover.SecurityProofs

open Finset

/-- **The covering bound's count over blocks each at its own rate**: if blocks `S u` of a `(s, c_w)`-parameter-hard `e`
compute every gate with a predecessor, each with at most `s` gates and handed `k_u ≤ α σ_u` gates outside `θ`, where
`0 ≤ σ_u ≤ Γ`, `α Γ < c_w D` and `c_a > 0`, then the multiplication blocks chained at `(D, c_a)` number
`c_w c_a |𝓜*| ≤ (α² Γ + c_w c_a α) Σ σ_u`. -/
theorem chained_le_of_blocks {e : Verity.Declaration} (B : Nci.Covering.MulBlocks e) {n : ℕ}
    (S : Fin n → Nci.Covering.Subcircuit e.D) (σ : Fin n → ℝ) {Γ s D : ℕ} {α cw ca : ℝ}
    (hα : 0 ≤ α) (hca : 0 < ca) (hhard : Nci.Assumptions.ParameterHardAt e B s cw) (hk : α * Γ < cw * D)
    (hs : ∀ u, (S u).gates.card ≤ s) (hσ0 : ∀ u, 0 ≤ σ u) (hσΓ : ∀ u, σ u ≤ Γ)
    (hku : ∀ u, (((S u).ins \ e.θ).card : ℝ) ≤ α * σ u)
    (hcov : ∀ g, g ∉ Nci.Covering.sources e.D → ∃ u, g ∈ (S u).inner) :
    cw * ca * (Nci.Assumptions.chainedBlocksAt e B D ca).card ≤ (α * α * Γ + cw * ca * α) * ∑ u, σ u := by
  classical
  set Mc := Nci.Assumptions.chainedBlocksAt e B D ca with hMc
  have hcw : 0 < cw := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D), mul_nonneg hα (Nat.cast_nonneg Γ : (0 : ℝ) ≤ Γ)]
  have hpos : 0 < cw * ca := mul_pos hcw hca
  have hkD : ∀ u, ((((S u).ins \ e.θ).card : ℕ) : ℝ) < cw * D := fun u =>
    ((hku u).trans (mul_le_mul_of_nonneg_left (hσΓ u) hα)).trans_lt hk
  have hf : ∀ u, cw * ca * ((Mc.filter fun p => B.o p ∈ (S u).inner).card : ℝ) ≤
      (((S u).ins \ e.θ).card : ℝ) * ((S u).ins \ e.θ).card + cw * ca * ((S u).ins \ e.θ).card := by
    intro u
    have h := Nci.SecurityProofs.CoveringTwoCosts.card_charged_le B (hkD u) hca hhard (S := S u) ⟨hs u, le_rfl⟩
    calc cw * ca * ((Mc.filter fun p => B.o p ∈ (S u).inner).card : ℝ)
        ≤ cw * ca * ((((S u).ins \ e.θ).card : ℝ) / cw * ((((S u).ins \ e.θ).card : ℝ) / ca) +
            ((S u).ins \ e.θ).card) := mul_le_mul_of_nonneg_left h hpos.le
      _ = (((S u).ins \ e.θ).card : ℝ) * ((S u).ins \ e.θ).card + cw * ca * ((S u).ins \ e.θ).card := by
        field_simp
  have hsum := sum_area_le univ (f := fun u => ((Mc.filter fun p => B.o p ∈ (S u).inner).card : ℝ))
    (k := fun u => (((S u).ins \ e.θ).card : ℝ)) (σ := σ) (Γ := (Γ : ℝ)) hpos.le hα
    (fun u _ => Nat.cast_nonneg _) (fun u _ => hσ0 u) (fun u _ => hf u) (fun u _ => hku u) (fun u _ => hσΓ u)
  have hMsub : Mc ⊆ univ.biUnion fun u => Mc.filter fun p => B.o p ∈ (S u).inner := by
    intro p hp
    have hpP : p ∈ B.P := (mem_filter.1 hp).1
    obtain ⟨u, hu⟩ := hcov _ fun h => disjoint_left.1 (B.m_computed p hpP) (B.o_mem p hpP) h
    exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨hp, hu⟩⟩
  have hcard : (Mc.card : ℝ) ≤ ∑ u, ((Mc.filter fun p => B.o p ∈ (S u).inner).card : ℝ) := by
    exact_mod_cast (card_le_card hMsub).trans card_biUnion_le
  calc cw * ca * (Mc.card : ℝ) ≤ cw * ca * ∑ u, ((Mc.filter fun p => B.o p ∈ (S u).inner).card : ℝ) :=
        mul_le_mul_of_nonneg_left hcard hpos.le
    _ ≤ (α * α * Γ + cw * ca * α) * ∑ u, σ u := hsum

/-- **The count's slowdown form**: `c_w c_a x ≤ (α² Γ + c_w c_a α) size(d)` gives
`(x / size(c)) Γ / ((α Γ / c_w)(α Γ / c_a) + α Γ) ≤ size(d) / size(c)`. -/
theorem slowdown_of_count {x Γ α cw ca sd sc : ℝ} (hcw : 0 < cw) (hca : 0 < ca) (hΓ : 0 ≤ Γ) (hα : 0 ≤ α)
    (hsd : 0 ≤ sd) (hsc : 0 ≤ sc) (h : cw * ca * x ≤ (α * α * Γ + cw * ca * α) * sd) :
    x / sc * Γ / (α * Γ / cw * (α * Γ / ca) + α * Γ) ≤ sd / sc := by
  have hpos : 0 < cw * ca := mul_pos hcw hca
  have hK0 : 0 ≤ α * Γ / cw * (α * Γ / ca) + α * Γ := by positivity
  have hxK : x * Γ ≤ (α * Γ / cw * (α * Γ / ca) + α * Γ) * sd := by
    have e : cw * ca * ((α * Γ / cw * (α * Γ / ca) + α * Γ) * sd) = Γ * ((α * α * Γ + cw * ca * α) * sd) := by
      field_simp
    have h1 : cw * ca * (x * Γ) ≤ cw * ca * ((α * Γ / cw * (α * Γ / ca) + α * Γ) * sd) := by
      rw [e]
      nlinarith [mul_le_mul_of_nonneg_left h hΓ]
    exact le_of_mul_le_mul_left h1 hpos
  rw [div_mul_eq_mul_div, div_right_comm]
  rcases hK0.lt_or_eq with hK | hK
  · exact div_le_div_of_nonneg_right ((div_le_iff₀ hK).2 (by linarith)) hsc
  · rw [← hK, div_zero, zero_div]
    exact div_nonneg hsd hsc

/-- The gate view of `c`'s bits as a declaration is its bit view. -/
theorem ofGates_bitDecl (c : Verity.Declaration) : ofGates (bitDecl c) = Circuit.ofDecl c := rfl

/-- **The block of a set `R` of `c`'s gates, as a subcircuit of `bitDecl c`**: its bits, with the inputs the bit view
hands it. -/
def bitBlock (c : Verity.Declaration) (R : Finset (Fin c.D.N)) : Nci.Covering.Subcircuit (bitDecl c).D :=
  Gates.toSubcircuit (bitDecl c) (block c R)

theorem bitBlock_gates (c : Verity.Declaration) (R : Finset (Fin c.D.N)) : (bitBlock c R).gates = block c R := rfl

theorem bitBlock_ins (c : Verity.Declaration) (R : Finset (Fin c.D.N)) :
    (bitBlock c R).ins = (Circuit.ofDecl c).inputs (block c R) := rfl

theorem bitBlock_inner (c : Verity.Declaration) (R : Finset (Fin c.D.N)) :
    (bitBlock c R).inner = (Circuit.ofDecl c).computed (block c R) := rfl

theorem bitDecl_θ (c : Verity.Declaration) : (bitDecl c).θ = (Circuit.ofDecl c).θ := rfl

/-- A bit of `bitDecl c` that is not a source reads something in the bit view. -/
theorem pred_nonempty_of_not_source {c : Verity.Declaration} {y : Fin (bitDecl c).D.N}
    (hy : y ∉ Nci.Covering.sources (bitDecl c).D) : ((Circuit.ofDecl c).pred y).Nonempty :=
  nonempty_iff_ne_empty.2 fun h => hy (Gates.mem_sources.2 h)

/-- A computed gate of `c` lies in a piece of a division pulled back to `c`. -/
theorem exists_pull_gates {c d : Verity.Declaration} (M : Verity.Containment c.D d.D) {n : ℕ}
    (P : Verity.Partition d.D n) {g : Fin c.D.N} (hg : g ∈ c.D.computed) :
    ∃ u, g ∈ (Nci.SecurityProofs.pull M P).gates u := by
  cases h : (Nci.SecurityProofs.pull M P).unit g with
  | none =>
    rw [(Nci.SecurityProofs.pull M P).input_iff] at h
    simp [Verity.Circuit.computed, h] at hg
  | some u => exact ⟨u, ((Nci.SecurityProofs.pull M P).mem_gates).2 h⟩

theorem CoverTrainingChained : Guarantees.CoverTrainingChained := by
  intro Γ α c d M B s D w₀ cw ca hα hC hB hH hk hca hw hs
  have hcw : 0 < cw := by
    by_contra h
    nlinarith [not_lt.1 h, (Nat.cast_nonneg D : (0 : ℝ) ≤ D), mul_nonneg hα (Nat.cast_nonneg Γ : (0 : ℝ) ≤ Γ)]
  obtain ⟨n, P, hP⟩ := hC
  set R : Fin n → Finset (Fin c.D.N) := fun u => (Nci.SecurityProofs.pull M P).gates u with hR
  set σ : Fin n → ℝ := fun u => (d.D.size (P.gates u) : ℝ) with hσdef
  have hσΓ : ∀ u, σ u ≤ Γ := fun u => by
    show (d.D.size (P.gates u) : ℝ) ≤ Γ
    exact_mod_cast (hP u).1
  have hnov : ∀ u, (novelReads c (R u) : ℝ) ≤ α * σ u := fun u =>
    (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_pull M hB P u)).trans (hP u).2
  have hku : ∀ u, ((((bitBlock c (R u)).ins \ (bitDecl c).θ).card : ℕ) : ℝ) ≤ α * σ u := fun u => by
    rw [bitBlock_ins, bitDecl_θ]
    exact (Nat.cast_le.2
      (translation (R u) ((Nci.SecurityProofs.pull M P).gates_subset_computed u)).2.2).trans (hnov u)
  have hcard : ∀ u, (bitBlock c (R u)).gates.card ≤ s := fun u => by
    rw [bitBlock_gates]
    have h1 : ((block c (R u)).card : ℝ) ≤ w₀ * (R u).card + novelReads c (R u) + c.D.width c.θ := by
      exact_mod_cast card_block_le hw (R u)
    have h2 : ((R u).card : ℝ) ≤ Γ := by
      exact_mod_cast (Nci.SecurityProofs.card_pull_gates_le u).trans (hP u).1
    have h3 : (w₀ : ℝ) * (R u).card ≤ w₀ * Γ := mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg _)
    have h4 : α * σ u ≤ α * Γ := mul_le_mul_of_nonneg_left (hσΓ u) hα
    have h5 : ((block c (R u)).card : ℝ) ≤ s := by linarith [hnov u]
    exact_mod_cast h5
  have hcov : ∀ y, y ∉ Nci.Covering.sources (bitDecl c).D → ∃ u, y ∈ (bitBlock c (R u)).inner := by
    intro y hy
    have hne := pred_nonempty_of_not_source hy
    obtain ⟨u, hu⟩ := exists_pull_gates M P (computed_of_pred hne)
    exact ⟨u, by rw [bitBlock_inner]; exact mem_computed_block_of_pred hne hu⟩
  have hcount := chained_le_of_blocks B (fun u => bitBlock c (R u)) σ hα hca hH hk hcard
    (fun u => Nat.cast_nonneg _) hσΓ hku hcov
  have hσsum : ∑ u, σ u = d.D.size d.D.computed := by
    simp only [hσdef, Verity.Circuit.size]
    exact_mod_cast P.sum_card_gates
  rw [hσsum] at hcount
  exact ⟨hcount, slowdown_of_count hcw hca (Nat.cast_nonneg _) hα (Nat.cast_nonneg _) (Nat.cast_nonneg _) hcount⟩

end Nci.Cover.SecurityProofs
