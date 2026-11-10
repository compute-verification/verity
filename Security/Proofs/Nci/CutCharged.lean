module

public import Proofs.Nci.CutWitness

/-!
# Blocks charged whole: what they hold, and that `HECharged` says something

A set of gates holds whole at most its own gates over the fewest gates of a MAC, MACs being disjoint
(`card_inside_mul_le`): what a block charged whole holds. `HECharged_honest` and `HECharged_nontrivial` are at the toy
`Nci.Cut.Toy`. With a block for each MAC, the first under the cap `k = 2` and the second charged whole, `HECharged`'s
premises hold together. With one block holding both MACs and no block charged, the block reads three gates, and at the
total `T = 0` the bound is `0`, below `N = 2`.
-/

@[expose] public section

namespace Nci.Cut.SecurityProofs

open Verity Finset

/-- A set `A` of gates holds whole at most `|A| / γ` MACs of at least `γ` gates each: MACs are disjoint. -/
theorem card_inside_mul_le {C : Circuit} {ι : Type} [Fintype ι] (W : Work C ι) (A : Finset (Fin C.N)) {γ : ℕ}
    (hγ : ∀ s, γ ≤ (W.mac s).card) : (W.inside A).card * γ ≤ A.card :=
  calc (W.inside A).card * γ = ∑ _s ∈ W.inside A, γ := by rw [sum_const, smul_eq_mul]
    _ ≤ ∑ s ∈ W.inside A, (W.mac s).card := sum_le_sum fun s _ => hγ s
    _ = ((W.inside A).biUnion W.mac).card := (card_biUnion fun s _ t _ hst => W.mac_disjoint s t hst).symm
    _ ≤ A.card := card_le_card (biUnion_subset.2 fun _ hs => (mem_filter.1 hs).2)

theorem toy_card_P2 : ∀ u, (Toy.P2.gates u).card ≤ 1 := by decide

theorem toy_card_mac : ∀ s, 1 ≤ (Toy.W.mac s).card := by decide

theorem HECharged_honest : Guarantees.HECharged_honest := by
  have h0 := toy_reads_P2 0
  have h1 := toy_reads_P2 1
  refine ⟨toy_values, toy_noSharedPair, toy_hardness 2, toy_exports 1, fun u _ => toy_reads_P2 u, ?_, ?_,
    toy_card_mac, fun u _ => toy_card_P2 u⟩
  · rw [Fin.sum_univ_two]
    omega
  · rw [sum_singleton]
    exact (Nat.pow_le_pow_left h0 2).trans (by norm_num)

theorem HECharged_nontrivial : Guarantees.HECharged_nontrivial := by
  refine ⟨one_pos, toy_values, toy_noSharedPair, toy_hardness 0, toy_exports 1, fun h => ?_, ?_⟩
  · have h3 := toy_reads_P1
    rw [Fin.sum_univ_one] at h
    omega
  · rw [toy_N, compl_univ, sum_empty]
    norm_num [workHEOn]

end Nci.Cut.SecurityProofs
