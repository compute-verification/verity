import Proofs.Nci.Cover.Training

/-!
# The width variant on whole-word activations, proved

The width variant charges each use of a block's whole products `f` gates of its region. On a declaration's bit view
that holds of the blocks the bridge uses, which are whole words (`block c R = wordBits c (R ∪ in(R))`), when the regions
are whole words at least `f` bits wide: a block handed a bit of such a region (`whole_use_entry`) is handed all of that
word's bits (`width_aligned`), so `aligned_charge`.

With the charge alone, under `UpdateHardness`, the uses cost `f #J ≤ k` (`uses_charge_le`: the regions are disjoint and
outside `θ`) and the updated parameters `b #L ≤ k`, from the same inputs, so `b f #J #L ≤ k²` and a block finishes at
most `k² / (b f) + k` counted products (`finished_le_of_charge_base`). Summed over a compliant division pulled back
(`training_of_block_bound`), that is `CoverTrainingAligned`.
-/

namespace Nci.Cover.SecurityProofs

open Finset BitView

/-- **The uses of a block's whole products, `c` gates each**: if each is handed at least `c` gates of its region, the
regions disjoint and outside `θ`, then `c #J ≤ #(inputs I \ θ)`. -/
theorem uses_charge_le {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {D : ℕ} (hU : M.Upstream A D) {I : Finset (Fin C.N)} {c : ℝ}
    (hc : ∀ j ∈ (M.whole I).image Prod.fst, c ≤ (C.inputs I ∩ A j).card) :
    c * ((M.whole I).image Prod.fst).card ≤ (C.inputs I \ C.θ).card := by
  have hJ : (M.whole I).image Prod.fst ⊆ M.uses := image_subset_image (filter_subset _ _)
  have hT : ((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j) ⊆ C.inputs I \ C.θ := by
    intro x hx
    obtain ⟨j, hj, hx⟩ := mem_biUnion.1 hx
    obtain ⟨hxI, hxA⟩ := mem_inter.1 hx
    exact mem_sdiff.2 ⟨hxI, fun hθ => disjoint_left.1 (hU.disjoint_θ j (hJ hj)) hxA hθ⟩
  have hTcard : (((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j)).card =
      ∑ j ∈ (M.whole I).image Prod.fst, (C.inputs I ∩ A j).card :=
    card_biUnion fun j hj j' hj' hne =>
      (hU.disjoint (hJ hj) (hJ hj') hne).mono inter_subset_right inter_subset_right
  calc c * ((M.whole I).image Prod.fst).card
      = ∑ _j ∈ (M.whole I).image Prod.fst, c := by rw [sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ j ∈ (M.whole I).image Prod.fst, ((C.inputs I ∩ A j).card : ℝ) := sum_le_sum hc
    _ = (((M.whole I).image Prod.fst).biUnion (fun j => C.inputs I ∩ A j)).card := by
      rw [hTcard, Nat.cast_sum]
    _ ≤ (C.inputs I \ C.θ).card := by exact_mod_cast card_le_card hT

/-- **Intensity from a per-use charge, under `UpdateHardness` alone**: a set of at most `s` gates handed at most `k`
bits besides `θ`, each use of whose whole products is handed at least `c ≥ 0` gates of its region, finishes at most
`k² / (b c) + k` counted products: `b c · #(finished I) ≤ k² + b c k`. -/
theorem finished_le_of_charge_base {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L] (M : MulBlocks C J L)
    (A : J → Finset (Fin C.N)) {s k D : ℕ} {b c : ℝ} (hb : 0 < b) (hc0 : 0 ≤ c)
    (hH : Assumptions.UpdateHardness M s b) (hU : M.Upstream A D) {I : Finset (Fin C.N)} (hs : I.card ≤ s)
    (hI : (C.inputs I \ C.θ).card ≤ k) (hc : ∀ j ∈ (M.whole I).image Prod.fst, c ≤ (C.inputs I ∩ A j).card) :
    b * c * (M.finished I).card ≤ k * k + b * c * k := by
  have hK : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI
  have hJ := (uses_charge_le M A hU hc).trans hK
  have hL := (hH I hs (M.whole I) (fun p hp => M.P_subset (mem_filter.1 hp).1)
    (fun p hp => (mem_filter.1 hp).2)).trans hK
  have hJL : (c * ((M.whole I).image Prod.fst).card) * (b * ((M.whole I).image Prod.snd).card) ≤ k * k :=
    mul_le_mul hJ hL (by positivity) (Nat.cast_nonneg _)
  have hW : ((M.whole I).card : ℝ) ≤ ((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card := by
    exact_mod_cast (card_le_card subset_product).trans (card_product _ _).le
  have hrest : ((M.finished I \ M.whole I).card : ℝ) ≤ k :=
    (by exact_mod_cast finished_sdiff_whole_card_le M (I := I) : ((M.finished I \ M.whole I).card : ℝ) ≤ _).trans hK
  have hfin : ((M.finished I).card : ℝ) ≤ (M.finished I \ M.whole I).card + (M.whole I).card := by
    exact_mod_cast card_le_card_sdiff_add_card
  have hbc : 0 ≤ b * c := by positivity
  have h1 := mul_le_mul_of_nonneg_left hfin hbc
  have h2 := mul_le_mul_of_nonneg_left hrest hbc
  have h3 := mul_le_mul_of_nonneg_left hW hbc
  have h4 : b * c * (((M.whole I).image Prod.fst).card * ((M.whole I).image Prod.snd).card : ℝ) =
      (c * ((M.whole I).image Prod.fst).card) * (b * ((M.whole I).image Prod.snd).card) := by ring
  linarith

/-- **A block of whole words is charged `f` bits per use** of its whole products, when each use's activation region is
the bits of whole words at least `f` bits wide. -/
theorem aligned_charge (c : Verity.RegisteredCircuit) {J L : Type} [DecidableEq J] [DecidableEq L]
    (Mb : MulBlocks (Circuit.ofDecl c) J L) (Aw : J → Finset (Fin c.D.N)) {s k D f : ℕ} {b : ℝ} (hb : 0 < b)
    (hH : Assumptions.UpdateHardness Mb s b) (hU : Mb.Upstream (fun j => wordBits c (Aw j)) D)
    (hk : (k : ℝ) < b * D) (hf : ∀ j ∈ Mb.uses, ∀ a ∈ Aw j, f ≤ c.D.w a) (R : Finset (Fin c.D.N))
    (hs : (block c R).card ≤ s) (hR : ((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card ≤ k) :
    ∀ j ∈ (Mb.whole (block c R)).image Prod.fst,
      (f : ℝ) ≤ ((Circuit.ofDecl c).inputs (block c R) ∩ wordBits c (Aw j)).card := by
  intro j hj
  have hne := whole_use_entry Mb (fun j => wordBits c (Aw j)) hb hH hU hk (I := block c R) ⟨hs, hR⟩ j hj
  have hjU : j ∈ Mb.uses := image_subset_image (filter_subset _ _) hj
  exact Nat.cast_le.2 (width_aligned (R ∪ c.D.into R) (Aw j) f (hf j hjU) hne)

/-- **The intensity of a block of whole words on whole-word activations**, under `UpdateHardness` alone: a block of at
most `s` bits handed `k < b D` bits outside the free bits finishes at most `k² / (b f) + k` counted products. -/
theorem alignedIntensity (c : Verity.RegisteredCircuit) {J L : Type} [DecidableEq J] [DecidableEq L]
    (Mb : MulBlocks (Circuit.ofDecl c) J L) (Aw : J → Finset (Fin c.D.N)) {s k D f : ℕ} {b : ℝ} (hb : 0 < b)
    (hH : Assumptions.UpdateHardness Mb s b) (hU : Mb.Upstream (fun j => wordBits c (Aw j)) D)
    (hk : (k : ℝ) < b * D) (hf : ∀ j ∈ Mb.uses, ∀ a ∈ Aw j, f ≤ c.D.w a) (R : Finset (Fin c.D.N))
    (hs : (block c R).card ≤ s) (hR : ((Circuit.ofDecl c).inputs (block c R) \ (Circuit.ofDecl c).θ).card ≤ k) :
    b * f * (Mb.finished (block c R)).card ≤ k * k + b * f * k :=
  finished_le_of_charge_base Mb _ hb (Nat.cast_nonneg f) hH hU hs hR
    (aligned_charge c Mb Aw hb hH hU hk hf R hs hR)

theorem CoverTrainingAligned : Guarantees.CoverTrainingAligned := by
  intro Γ α b α₀ c d M J L _ _ Mb Aw s D w₀ f hb hα hC hB hH hU hk hf hw hs hα₀
  have h := training_of_block_bound (e := b * f) M Mb (by positivity) hα hC hB hk hw hs hα₀ fun R hR hkR =>
    alignedIntensity c Mb Aw hb hH hU hkR hf R hR le_rfl
  calc α₀ * b * f * c.D.size c.D.computed = α₀ * (b * f) * c.D.size c.D.computed := by ring
    _ ≤ (α * α * Γ + b * f * α) * d.D.size d.D.computed := h

end Nci.Cover.SecurityProofs
