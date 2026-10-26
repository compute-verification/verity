import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Nat
import Definitions.Accounting.Guarantees.Requirement

/-!
# A declared circuit's requirement, from what the audit credits

The values a layout packs in one block total at most the block's `b` bits (`bits_filter_block_le`): their bit ranges are
disjoint intervals of `[0, b)`. So the values of a set `S`, each in a block of `R`, outside the blocks `H` total at most
`|R \ H|·b` bits (`bits_le_bitsAt_add`), and those that lie in one block per group at most `b` per group
(`bits_le_groups`). `Accounting.SecurityProofs.RequirementHeld` applies the first to the credited values and the
resident blocks, `Accounting.SecurityProofs.RequirementMoved` to the credited reads and the plan, and
`Accounting.SecurityProofs.RequirementUncredited` and `Accounting.SecurityProofs.RequirementUnplanned` the second to
the values in incomplete blocks. Each uses only that every credited block is resident, or planned: other resident or
planned blocks hold none of `W`'s credited values. `Accounting.SecurityProofs.RequirementHeldLive` and
`Accounting.SecurityProofs.RequirementDense` sum the credited bits block by block. An execution that follows its wiring
reads only live values (`reads_subset_live`).
-/

namespace Accounting.Requirement

variable {G : Type} [DecidableEq G]

/-- A step reads only values live when it begins. -/
theorem reads_subset_live {W : Wiring G} {e : Execution G} (h : e.Follows W) (t : ℕ) :
    reads W e t ⊆ live W e t := by
  intro v hv
  simp only [reads, live, Finset.mem_filter] at hv ⊢
  obtain ⟨hv, hb, g, hg, hgt, hu⟩ := hv
  exact ⟨hv, hb, g, hg, hgt.ge, (h.2 g hg).1 hu⟩

variable (W : Wiring G) (ℓ : Layout G)

omit [DecidableEq G] in
/-- A subset of a packed set is packed. -/
theorem packs_mono {b : ℕ} {S T : Finset G} (h : Packs W ℓ b S) (hT : T ⊆ S) : Packs W ℓ b T :=
  fun u hu => ⟨(h u (hT hu)).1, fun v hv => (h u (hT hu)).2 v (hT hv)⟩

omit [DecidableEq G] in
/-- The values of a packed set in one block total at most the block's `b` bits. -/
theorem bits_filter_block_le {b : ℕ} {S T : Finset G} (hT : T ⊆ S) (h : Packs W ℓ b S) (j : ℕ) :
    bits W (T.filter fun v => ℓ.block v = j) ≤ b := by
  set U := T.filter fun v => ℓ.block v = j
  let I : G → Finset ℕ := fun v => Finset.Ico (ℓ.offset v) (ℓ.offset v + W.width v)
  have hdisj : (U : Set G).PairwiseDisjoint I := by
    intro u hu v hv huv
    have hu' := Finset.mem_filter.mp (Finset.mem_coe.mp hu)
    have hv' := Finset.mem_filter.mp (Finset.mem_coe.mp hv)
    refine Finset.disjoint_left.mpr fun x hx hx' => ?_
    simp only [I, Finset.mem_Ico] at hx hx'
    rcases (h u (hT hu'.1)).2 v (hT hv'.1) huv (hu'.2.trans hv'.2.symm) with hle | hle <;> omega
  calc bits W U = ∑ v ∈ U, (I v).card := by simp [bits, I]
    _ = (U.biUnion I).card := (Finset.card_biUnion hdisj).symm
    _ ≤ (Finset.range b).card := by
        refine Finset.card_le_card fun x hx => ?_
        obtain ⟨v, hv, hx⟩ := Finset.mem_biUnion.mp hx
        have hfit := (h v (hT (Finset.mem_filter.mp hv).1)).1
        simp only [I, Finset.mem_Ico] at hx
        exact Finset.mem_range.mpr (by omega)
    _ = b := Finset.card_range b

omit [DecidableEq G] in
/-- The values of a packed set `S`, each in a block of `R`, total at most their bits in the blocks `H` and `b` for each
block of `R` outside `H`. -/
theorem bits_le_bitsAt_add {b : ℕ} {S : Finset G} {R : Finset ℕ} (H : Finset ℕ) (hR : ∀ v ∈ S, ℓ.block v ∈ R)
    (h : Packs W ℓ b S) : bits W S ≤ bitsAt W ℓ S H + (R \ H).card * b := by
  have hmaps : ∀ v ∈ S.filter (fun v => ℓ.block v ∉ H), ℓ.block v ∈ R \ H := fun v hv =>
    Finset.mem_sdiff.mpr ⟨hR v (Finset.mem_filter.mp hv).1, (Finset.mem_filter.mp hv).2⟩
  have hout : bits W (S.filter fun v => ℓ.block v ∉ H) ≤ (R \ H).card * b := by
    calc bits W (S.filter fun v => ℓ.block v ∉ H)
        = ∑ j ∈ R \ H, bits W ((S.filter fun v => ℓ.block v ∉ H).filter fun v => ℓ.block v = j) :=
          (Finset.sum_fiberwise_of_maps_to hmaps _).symm
      _ ≤ ∑ _j ∈ R \ H, b := Finset.sum_le_sum fun j _ => bits_filter_block_le W ℓ (Finset.filter_subset _ _) h j
      _ = (R \ H).card * b := Finset.sum_const_nat fun _ _ => rfl
  calc bits W S = bitsAt W ℓ S H + bits W (S.filter fun v => ℓ.block v ∉ H) :=
        (Finset.sum_filter_add_sum_filter_not S _ _).symm
    _ ≤ bitsAt W ℓ S H + (R \ H).card * b := Nat.add_le_add_left hout _

omit [DecidableEq G] in
/-- The values of a subset `T` of a packed set that lie in one block per group of `g` total at most `b` bits per
group. -/
theorem bits_le_groups {K : Type} [DecidableEq K] {b : ℕ} {S T : Finset G} (hT : T ⊆ S) (h : Packs W ℓ b S)
    (g : G → K) (hone : OneBlockEach ℓ g T) : bits W T ≤ (T.image g).card * b := by
  have hmaps : ∀ v ∈ T, g v ∈ T.image g := fun v hv => Finset.mem_image_of_mem g hv
  have hk : ∀ k, bits W (T.filter fun v => g v = k) ≤ b := by
    intro k
    rcases (T.filter fun v => g v = k).eq_empty_or_nonempty with he | ⟨v₀, hv₀⟩
    · rw [he]
      simp [bits]
    · have hj : ∀ v ∈ T.filter (fun v => g v = k), ℓ.block v = ℓ.block v₀ := fun v hv =>
        Finset.card_le_one.mp (hone k) _ (Finset.mem_image_of_mem _ hv) _ (Finset.mem_image_of_mem _ hv₀)
      rw [← Finset.filter_true_of_mem hj]
      exact bits_filter_block_le W ℓ ((Finset.filter_subset _ _).trans hT) h _
  calc bits W T = ∑ k ∈ T.image g, bits W (T.filter fun v => g v = k) :=
        (Finset.sum_fiberwise_of_maps_to hmaps _).symm
    _ ≤ ∑ _k ∈ T.image g, b := Finset.sum_le_sum fun k _ => hk k
    _ = (T.image g).card * b := Finset.sum_const_nat fun _ _ => rfl

end Accounting.Requirement

namespace Accounting.SecurityProofs

open Accounting.Requirement

theorem RequirementHeld : Accounting.Guarantees.RequirementHeld := by
  intro G _ W e ℓ σ b t H ⟨hres, hpack, _⟩
  exact bits_le_bitsAt_add W ℓ H (fun v hv => hres (Finset.mem_image_of_mem _ hv))
    (packs_mono W ℓ hpack (Finset.filter_subset _ _))

theorem RequirementHeldLive : Accounting.Guarantees.RequirementHeldLive := by
  intro G _ W e ℓ σ b t H ⟨hres, _, _⟩
  have hmaps : ∀ v ∈ (credited W e ℓ t).filter (fun v => ℓ.block v ∉ H), ℓ.block v ∈ σ.resident t \ H :=
    fun v hv => Finset.mem_sdiff.mpr
      ⟨hres (Finset.mem_image_of_mem _ (Finset.mem_filter.mp hv).1), (Finset.mem_filter.mp hv).2⟩
  rw [creditedBits, bits, ← Finset.sum_filter_add_sum_filter_not (credited W e ℓ t) (fun v => ℓ.block v ∈ H),
    ← Finset.sum_fiberwise_of_maps_to hmaps]
  refine congrArg _ (Finset.sum_congr rfl fun j hj => ?_)
  have hjH : j ∉ H := (Finset.mem_sdiff.mp hj).2
  rw [bitsIn, bits, Finset.filter_filter]
  exact Finset.sum_congr (Finset.filter_congr fun v _ => ⟨fun h => h.2, fun h => ⟨h ▸ hjH, h⟩⟩) fun _ _ => rfl

theorem RequirementDense : Accounting.Guarantees.RequirementDense := by
  intro G _ W e ℓ σ b t D ⟨hres, _, _⟩ hD
  have hmaps : ∀ v ∈ credited W e ℓ t, ℓ.block v ∈ σ.resident t := fun v hv => hres (Finset.mem_image_of_mem _ hv)
  calc (σ.resident t).card * D = ∑ _j ∈ σ.resident t, D := (Finset.sum_const_nat fun _ _ => rfl).symm
    _ ≤ ∑ j ∈ σ.resident t, bitsIn W ℓ (credited W e ℓ t) j := Finset.sum_le_sum hD
    _ = creditedBits W e ℓ t := Finset.sum_fiberwise_of_maps_to hmaps _

theorem RequirementMoved : Accounting.Guarantees.RequirementMoved := by
  intro G _ W e ℓ plan b t M ⟨hplan, hpack⟩
  exact bits_le_bitsAt_add W ℓ M (fun v hv => hplan (Finset.mem_image_of_mem _ hv))
    (packs_mono W ℓ hpack (Finset.filter_subset _ _))

theorem RequirementUncredited : Accounting.Guarantees.RequirementUncredited := by
  intro G _ K _ W e ℓ σ b t g ⟨_, hpack, _⟩ hone
  have hsub : credited W e ℓ t ⊆ live W e t := Finset.filter_subset _ _
  calc liveBits W e t = creditedBits W e ℓ t + bits W (live W e t \ credited W e ℓ t) := by
        simp only [liveBits, creditedBits, bits]
        rw [add_comm, Finset.sum_sdiff hsub]
    _ ≤ _ := Nat.add_le_add_left (bits_le_groups W ℓ Finset.sdiff_subset hpack g hone) _

theorem RequirementUnplanned : Accounting.Guarantees.RequirementUnplanned := by
  intro G _ K _ W e ℓ plan b t g ⟨_, hpack⟩ hone
  have hsub : creditedReads W e ℓ t ⊆ reads W e t := Finset.filter_subset _ _
  calc readBits W e t = creditedReadBits W e ℓ t + bits W (reads W e t \ creditedReads W e ℓ t) := by
        simp only [readBits, creditedReadBits, bits]
        rw [add_comm, Finset.sum_sdiff hsub]
    _ ≤ _ := Nat.add_le_add_left (bits_le_groups W ℓ Finset.sdiff_subset hpack g hone) _

end Accounting.SecurityProofs
