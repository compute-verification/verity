import Definitions.Nci.Guarantees.CoverTraining
import Proofs.Nci.Pull
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.Bits
import Proofs.Nci.Cover.Sum

/-!
# The cover theorem on a declaration that complies with the NCI policy, proved

Pull a compliant division of `d` back to `c` (`pull`): piece `u` has at most `σ_u ≤ Γ` gates (`card_pull_gates_le`) and
reads at most `α σ_u` bits of novel information (`novelReads_pull`). Its block in `c`'s bit view (`block`) is handed
`k_u ≤ α σ_u` bits outside the free bits (`translation`) and has at most `s` bits (`card_block_le`), so it finishes at
most `(k_u² + b k_u) / b` counted products (`intensity`). Every counted product's output bit reads something,
so its gate is a computed gate of `c`, in some piece, whose block computes it (`mem_computed_block_of_pred`). Summed over
the IUs (`sum_area_le`), `b #P ≤ (α² Γ + b α) size(d)`. `training_of_block_bound` is this route for any per-block
bound of the same form, which the whole-word variant (`Aligned`) reuses.
-/

namespace Nci.Cover.SecurityProofs

open Finset

/-- **The counted products are finished by the pieces' blocks**: every counted product's output bit is computed by the
block of the piece that holds its gate. -/
theorem card_P_le_sum_finished {c d : Verity.Declaration} (M : Verity.Containment c.D d.D) {n : ℕ}
    (P : Verity.Partition d.D n) {J L : Type} [DecidableEq J] [DecidableEq L]
    (Mb : MulBlocks (Circuit.ofDecl c) J L) :
    Mb.P.card ≤ ∑ u, (Mb.finished (block c ((Nci.SecurityProofs.pull M P).gates u))).card := by
  refine (card_le_card fun p hp => ?_).trans (card_biUnion_le (s := univ)
    (t := fun u => Mb.finished (block c ((Nci.SecurityProofs.pull M P).gates u))))
  have hpAll := Mb.P_subset hp
  have hne := Mb.m_computed p hpAll _ (o_mem Mb hpAll)
  have hg := computed_of_pred hne
  obtain ⟨u, hu⟩ : ∃ u, (Nci.SecurityProofs.pull M P).unit ((benc c).symm (Mb.o p)).1 = some u := by
    cases h : (Nci.SecurityProofs.pull M P).unit ((benc c).symm (Mb.o p)).1 with
    | none =>
      rw [(Nci.SecurityProofs.pull M P).input_iff] at h
      simp [Verity.Circuit.computed, h] at hg
    | some u => exact ⟨u, rfl⟩
  have hgR : ((benc c).symm (Mb.o p)).1 ∈ (Nci.SecurityProofs.pull M P).gates u :=
    ((Nci.SecurityProofs.pull M P).mem_gates).2 hu
  exact mem_biUnion.2 ⟨u, mem_univ _, mem_filter.2 ⟨hp, mem_computed_block_of_pred hne hgR⟩⟩

/-- **The per-block bounds summed over a compliant division pulled back to `c`**: if the block of every set of `c`'s
gates, when it has at most `s` bits and is handed `k < b D` bits outside the free bits, finishes at most `(k² + e k) / e`
counted products, then `α₀ e size(c) ≤ (α² Γ + e α) size(d)`. `CoverTraining` is this at `e = b`, from `intensity`. -/
theorem training_of_block_bound {Γ : ℕ} {α b α₀ e : ℝ} {c d : Verity.Declaration} (M : Verity.Containment c.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl c) J L) {s D w₀ : ℕ}
    (he : 0 ≤ e) (hα : 0 ≤ α) (hC : Nci.Complies Γ α d)
    (hB : Nci.Assumptions.ThetaRegistered c d (Nci.PartialContainment.ofContainment M)) (hk : α * Γ < b * D)
    (hw : ∀ g, c.D.w g ≤ w₀) (hs : w₀ * Γ + α * Γ + c.D.width c.θ ≤ s)
    (hα₀ : α₀ * c.D.size c.D.computed ≤ Mb.P.card)
    (hblk : ∀ R : Finset (Fin c.D.N), (block c R).card ≤ s →
      (((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card : ℝ) < b * D →
      e * (Mb.finished (block c R)).card ≤
        ((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card *
          ((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card +
        e * ((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card) :
    α₀ * e * c.D.size c.D.computed ≤ (α * α * Γ + e * α) * d.D.size d.D.computed := by
  obtain ⟨n, P, hP⟩ := hC
  set R : Fin n → Finset (Fin c.D.N) := fun u => (Nci.SecurityProofs.pull M P).gates u with hR
  set σ : Fin n → ℝ := fun u => (d.D.size (P.gates u) : ℝ) with hσdef
  set k : Fin n → ℕ := fun u =>
    ((Circuit.ofDecl c).inputs (block c (R u)) \ (Circuit.ofDecl c).θ).card with hkdef
  have hσΓ : ∀ u, σ u ≤ Γ := fun u => by
    show (d.D.size (P.gates u) : ℝ) ≤ Γ
    exact_mod_cast (hP u).1
  have hnov : ∀ u, (novelReads c (R u) : ℝ) ≤ α * σ u := fun u =>
    (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_pull M hB P u)).trans (hP u).2
  have hku : ∀ u, (k u : ℝ) ≤ α * σ u := fun u =>
    (Nat.cast_le.2 (translation (R u) ((Nci.SecurityProofs.pull M P).gates_subset_computed u)).2.2).trans
      (hnov u)
  have hαΓ : ∀ u, α * σ u ≤ α * Γ := fun u => mul_le_mul_of_nonneg_left (hσΓ u) hα
  have hkD : ∀ u, (k u : ℝ) < b * D := fun u => ((hku u).trans (hαΓ u)).trans_lt hk
  have hcard : ∀ u, (block c (R u)).card ≤ s := fun u => by
    have h1 : ((block c (R u)).card : ℝ) ≤ w₀ * (R u).card + novelReads c (R u) + c.D.width c.θ := by
      exact_mod_cast card_block_le hw (R u)
    have h2 : ((R u).card : ℝ) ≤ Γ := by
      exact_mod_cast (Nci.SecurityProofs.card_pull_gates_le u).trans (hP u).1
    have h3 : (w₀ : ℝ) * (R u).card ≤ w₀ * Γ := mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg _)
    have h4 := (hnov u).trans (hαΓ u)
    have h5 : ((block c (R u)).card : ℝ) ≤ s := by linarith
    exact_mod_cast h5
  have hI : ∀ u, e * ((Mb.finished (block c (R u))).card : ℝ) ≤ k u * k u + e * k u := fun u =>
    hblk (R u) (hcard u) (hkD u)
  have hsum := sum_area_le univ (f := fun u => ((Mb.finished (block c (R u))).card : ℝ))
    (k := fun u => (k u : ℝ)) (σ := σ) he hα (fun u _ => Nat.cast_nonneg _) (fun u _ => Nat.cast_nonneg _)
    (fun u _ => hI u) (fun u _ => hku u) (fun u _ => hσΓ u)
  have hσsum : ∑ u, σ u = d.D.size d.D.computed := by
    simp only [hσdef, Verity.Circuit.size]
    exact_mod_cast P.sum_card_gates
  have hPcard : (Mb.P.card : ℝ) ≤ ∑ u, ((Mb.finished (block c (R u))).card : ℝ) := by
    exact_mod_cast card_P_le_sum_finished M P Mb
  rw [hσsum] at hsum
  calc α₀ * e * c.D.size c.D.computed = e * (α₀ * c.D.size c.D.computed) := by ring
    _ ≤ e * Mb.P.card := mul_le_mul_of_nonneg_left hα₀ he
    _ ≤ e * ∑ u, ((Mb.finished (block c (R u))).card : ℝ) := mul_le_mul_of_nonneg_left hPcard he
    _ ≤ (α * α * Γ + e * α) * d.D.size d.D.computed := hsum

theorem CoverTraining : Guarantees.CoverTraining := by
  intro Γ α b α₀ c d M J L _ _ Mb A s D w₀ hb hα hC hB hH hU hk hw hs hα₀
  exact training_of_block_bound M Mb hb.le hα hC hB hk hw hs hα₀ fun R hR hkR =>
    intensity Mb A hb hH hU hkR (I := block c R) ⟨hR, le_rfl⟩

end Nci.Cover.SecurityProofs
