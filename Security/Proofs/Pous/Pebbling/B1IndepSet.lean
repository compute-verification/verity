import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Combinatorics.SimpleGraph.Finite
import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# B1: independent sets under a degree bound

The draft `PousProofs.b1_indep_set`, stated verbatim and proved. A maximum independent set `s` dominates the graph:
a vertex outside `s` with no neighbour in `s` could be added to it. So `n ≤ |s| + Σ_{u ∈ s} deg u ≤ (Δ + 1)|s|`.
-/

namespace PousPebbling

/-- B1. A graph of maximum degree `Δ` on `n` vertices has an independent set of at least `n/(Δ + 1)` vertices. -/
theorem b1_indep_set (n Δ : ℕ) (G : SimpleGraph (Fin n)) [DecidableRel G.Adj] (hΔ : ∀ v, G.degree v ≤ Δ) :
    ∃ s : Finset (Fin n), G.IsIndepSet (s : Set (Fin n)) ∧ n ≤ (Δ + 1) * s.card := by
  obtain ⟨s, hs⟩ := G.maximumIndepSet_exists (α := Fin n)
  refine ⟨s, hs.isIndepSet, ?_⟩
  have hcover : (Finset.univ : Finset (Fin n)) ⊆ s ∪ s.biUnion (fun u => G.neighborFinset u) := by
    intro v _
    by_contra hv
    simp only [Finset.mem_union, Finset.mem_biUnion, SimpleGraph.mem_neighborFinset, not_or, not_exists,
      not_and] at hv
    have hind : G.IsIndepSet ((insert v s : Finset (Fin n)) : Set (Fin n)) := by
      rw [Finset.coe_insert, SimpleGraph.IsIndepSet, Set.pairwise_insert]
      refine ⟨hs.isIndepSet, fun u hu _ => ⟨fun h => hv.2 u hu h.symm, fun h => hv.2 u hu h⟩⟩
    have hle := hs.maximum _ hind
    rw [Finset.card_insert_of_notMem hv.1] at hle
    omega
  calc n = (Finset.univ : Finset (Fin n)).card := by simp
    _ ≤ (s ∪ s.biUnion (fun u => G.neighborFinset u)).card := Finset.card_le_card hcover
    _ ≤ s.card + (s.biUnion (fun u => G.neighborFinset u)).card := Finset.card_union_le _ _
    _ ≤ s.card + ∑ u ∈ s, (G.neighborFinset u).card := by gcongr; exact Finset.card_biUnion_le
    _ ≤ s.card + ∑ _u ∈ s, Δ := Nat.add_le_add_left (Finset.sum_le_sum fun u _ => hΔ u) _
    _ = (Δ + 1) * s.card := by rw [Finset.sum_const, smul_eq_mul, Nat.succ_mul, Nat.mul_comm]; omega

/-! ### Proposed corollary

`construction-paths.md` §4 names the corollary only in prose: "the free fraction under local re-derivation". This is
one reading of it, for review. -/

/-- The free fraction under local re-derivation. In a loop-free digraph on `n` vertices where every vertex has at
most `Δ` neighbours (parents and children together), some set `S` of at least `n/(Δ + 1)` vertices spans no edge.
Every vertex of `S` then has all its parents outside `S`, so an adversary that stores every label outside `S`
re-derives all of `S` in one parallel round. -/
theorem b1_free_fraction (n Δ : ℕ) (E : Fin n → Fin n → Prop) [DecidableRel E] (hirr : ∀ v, ¬ E v v)
    (hΔ : ∀ v, (Finset.univ.filter fun u => E u v ∨ E v u).card ≤ Δ) :
    ∃ S : Finset (Fin n), (∀ u ∈ S, ∀ v ∈ S, ¬ E u v) ∧ n ≤ (Δ + 1) * S.card := by
  classical
  obtain ⟨s, hs, hcard⟩ := b1_indep_set n Δ (SimpleGraph.fromRel E) fun v => by
    refine le_trans (Finset.card_le_card fun u hu => ?_) (hΔ v)
    rw [SimpleGraph.mem_neighborFinset, SimpleGraph.fromRel_adj] at hu
    simpa [or_comm] using hu.2
  refine ⟨s, fun u hu v hv huv => ?_, hcard⟩
  by_cases h : u = v
  · exact hirr u (h ▸ huv)
  · exact hs hu hv h ((SimpleGraph.fromRel_adj _ _ _).2 ⟨h, Or.inl huv⟩)

end PousPebbling
