module

public import Definitions.Claims.SamplingTwo

/-!
# The two-stage extraction's unions: over footprints, and on the replay sample

Each wait of `CompilerKSTwo`'s extraction pays a union bound, `|S|^{1/k}` over the sets `S` it unions
(`wait_bad_le`), so its count `k` must cover `log₂ |S|`. Both unions are smaller than all `2^{n_PU}` sets of proof
blocks:

* `card_image_le_of_counts`, `prod_cell_le_footBits`, `footBits_le`: **the outer union, over footprints**. A function
  of a set's counts per cell (`counts`) on the grid of replay blocks and strata takes at most
  `∏_{ρ,t}(n_{ρ,t} + 1) ≤ 2^M` values over any family of sets, `M` the footprint bits (`footBits`), and `M` is at most
  the proof blocks: the cells are disjoint, and `n + 1 ≤ 2^n`.
* `cell_subset_sampleBlocks`, `inter_sampleBlocks_cell`, `card_sampleBlocks_le`, `card_image_inter_le`: **the inner
  union, on the replay sample**. In a replay block of the sample `S`, a set `X` and its projection
  `X ∩ sampleBlocks S` have the same proof blocks, so the second stage at `S` misses them alike, and the projections
  of any family on a set `B` number at most `2^{|B|}`.
* `cell_subset`, `replayOf_of_mem_cell`, `cell_disj`, `sum_card_cell_le`: cells are disjoint sets of proof blocks.
-/

@[expose] public section

namespace Claims.Sampling

open Finset

theorem cell_subset (Pt Pρ : Partition) (ρ t : ℕ) : cell Pt Pρ ρ t ⊆ range Pt.blocks.length :=
  filter_subset _ _

theorem replayOf_of_mem_cell {Pt Pρ : Partition} {ρ t v : ℕ} (h : v ∈ cell Pt Pρ ρ t) : replayOf Pt Pρ v = some ρ :=
  (mem_filter.1 h).2.1

theorem cell_disj {Pt Pρ : Partition} {ρ t ρ' t' v : ℕ} (h : v ∈ cell Pt Pρ ρ t) (h' : v ∈ cell Pt Pρ ρ' t') :
    ρ = ρ' ∧ t = t' :=
  ⟨Option.some.inj ((replayOf_of_mem_cell h).symm.trans (replayOf_of_mem_cell h')),
    (mem_filter.1 h).2.2.symm.trans (mem_filter.1 h').2.2⟩

/-- **The cells' sizes sum to at most the proof blocks**, over any grid: the cells are disjoint sets of proof
blocks. -/
theorem sum_card_cell_le (Pt Pρ : Partition) (G : Finset (ℕ × ℕ)) :
    ∑ p ∈ G, (cell Pt Pρ p.1 p.2).card ≤ Pt.blocks.length := by
  have hd : (G : Set (ℕ × ℕ)).PairwiseDisjoint fun p => cell Pt Pρ p.1 p.2 := by
    intro p _ q _ hpq
    show Disjoint (cell Pt Pρ p.1 p.2) (cell Pt Pρ q.1 q.2)
    exact disjoint_left.2 fun v hv hv' => hpq (Prod.ext_iff.2 (cell_disj hv hv'))
  rw [← card_biUnion hd]
  exact (card_le_card (biUnion_subset.2 fun p _ => cell_subset Pt Pρ p.1 p.2)).trans (card_range _).le

/-- **What reads counts on the grid takes at most `∏_{ρ,t}(n_{ρ,t} + 1)` values** over any family of sets of proof
blocks: a function of a set's counts per cell on the grid of replay blocks and strata, each at most its cell's size, is
a function of a vector of bounded counts. -/
theorem card_image_le_of_counts {β : Type*} [DecidableEq β] (Pt : Partition) (d : Design.Two) (S : Finset (Finset ℕ))
    (f : Finset ℕ → β)
    (hf : ∀ Y Y', (∀ ρ < d.part.blocks.length, ∀ t < Pt.strata.length,
      counts Pt d.part Y ρ t = counts Pt d.part Y' ρ t) → f Y = f Y') :
    (S.image f).card ≤
      ∏ p ∈ range d.part.blocks.length ×ˢ range Pt.strata.length, ((cell Pt d.part p.1 p.2).card + 1) := by
  classical
  let G := range d.part.blocks.length ×ˢ range Pt.strata.length
  let key : Finset ℕ → (p : ℕ × ℕ) → p ∈ G → ℕ := fun Y p _ => counts Pt d.part Y p.1 p.2
  calc (S.image f).card ≤ (G.pi fun p => range ((cell Pt d.part p.1 p.2).card + 1)).card := by
        refine card_le_card_of_surjOn (fun c => f (Function.invFunOn key S c)) fun b hb => ?_
        obtain ⟨Y, hY, rfl⟩ := mem_image.1 (mem_coe.1 hb)
        obtain ⟨-, h⟩ := Function.invFunOn_pos (f := key) (s := (S : Set (Finset ℕ))) ⟨Y, hY, rfl⟩
        refine ⟨key Y,
          mem_coe.2 (mem_pi.2 fun p _ => mem_range.2 (Nat.lt_succ_of_le (card_le_card inter_subset_right))),
          hf _ _ fun ρ hρ t ht => ?_⟩
        exact congrFun (congrFun h (ρ, t)) (mem_product.2 ⟨mem_range.2 hρ, mem_range.2 ht⟩)
    _ = ∏ p ∈ G, ((cell Pt d.part p.1 p.2).card + 1) := by
        rw [card_pi]
        exact prod_congr rfl fun p _ => card_range _

/-- **The footprint bits cover the footprints**: `∏_{ρ,t}(n_{ρ,t} + 1) ≤ 2^M`. -/
theorem prod_cell_le_footBits (Pt : Partition) (d : Design.Two) :
    ∏ p ∈ range d.part.blocks.length ×ˢ range Pt.strata.length, ((cell Pt d.part p.1 p.2).card + 1) ≤
      2 ^ footBits Pt d :=
  Nat.le_pow_clog (by norm_num) _

/-- **The footprint bits are at most the proof blocks**: `n + 1 ≤ 2^n` in each cell, and the cells are disjoint. -/
theorem footBits_le (Pt : Partition) (d : Design.Two) : footBits Pt d ≤ Pt.blocks.length := by
  unfold footBits
  rw [Nat.clog_le_iff_le_pow (by norm_num)]
  calc ∏ p ∈ range d.part.blocks.length ×ˢ range Pt.strata.length, ((cell Pt d.part p.1 p.2).card + 1)
      ≤ ∏ p ∈ range d.part.blocks.length ×ˢ range Pt.strata.length, 2 ^ (cell Pt d.part p.1 p.2).card :=
        prod_le_prod fun p _ => Nat.lt_two_pow_self
    _ = 2 ^ ∑ p ∈ range d.part.blocks.length ×ˢ range Pt.strata.length, (cell Pt d.part p.1 p.2).card :=
        prod_pow_eq_pow_sum _ _ _
    _ ≤ 2 ^ Pt.blocks.length := Nat.pow_le_pow_right (by norm_num) (sum_card_cell_le Pt d.part _)

/-- **A replay sample's proof blocks hold its replay blocks' cells**. -/
theorem cell_subset_sampleBlocks {Pt Pρ : Partition} {S : List ℕ} {ρ : ℕ} (hρ : ρ ∈ S) (t : ℕ) :
    cell Pt Pρ ρ t ⊆ sampleBlocks Pt Pρ S := fun _ hv =>
  mem_filter.2 ⟨cell_subset Pt Pρ ρ t hv, ρ, hρ, replayOf_of_mem_cell hv⟩

/-- **Projected on the replay sample, a set keeps its cells there**: in a replay block of `S`, `X ∩ sampleBlocks S`
has `X`'s proof blocks. -/
theorem inter_sampleBlocks_cell {Pt Pρ : Partition} {S : List ℕ} {ρ : ℕ} (hρ : ρ ∈ S) (X : Finset ℕ) (t : ℕ) :
    X ∩ sampleBlocks Pt Pρ S ∩ cell Pt Pρ ρ t = X ∩ cell Pt Pρ ρ t := by
  rw [inter_assoc, inter_eq_right.2 (cell_subset_sampleBlocks hρ t)]

theorem card_sampleBlocks_le (Pt Pρ : Partition) (S : List ℕ) : (sampleBlocks Pt Pρ S).card ≤ Pt.blocks.length :=
  (card_le_card (filter_subset _ _)).trans (card_range _).le

/-- **The projections on a set `B` number at most `2^{|B|}`**, over any family of sets. -/
theorem card_image_inter_le (Xs : Finset (Finset ℕ)) (B : Finset ℕ) : (Xs.image (· ∩ B)).card ≤ 2 ^ B.card := by
  rw [← card_powerset]
  exact card_le_card (image_subset_iff.2 fun X _ => mem_powerset.2 inter_subset_right)

end Claims.Sampling
