module

public import Proofs.Flock.SamplingGeneric.Fails
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fin.Embedding
import Mathlib.Data.List.GetD

/-!
# The sessions' weights at the gates their blocks need, counted by block (`sum_sessions_le`)

A run's extractions at gate `g` run each session some block of which needs `g`, once, at the session's first slot
(`sessSlots`). Two sessions' first slots with one key are one slot (`first_eq_of_key`), so distinct sessions hold
disjoint slots. A weight `e_s ≤ e_max` on each session, summed over a set of gates and, at each gate, over the sessions
some block of which needs it, is then at most `e_max` times the sample's blocks' needs, summed:
`Σ_g Σ_{s ∋ g} e_s = Σ_s e_s·|needs(s)| ≤ e_max·Σ_{u sampled} |needs u|`.
-/

@[expose] public section

namespace Claims.SamplingGeneric

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

theorem sum_range_getD_map (l : List ℕ) (f : ℕ → ℕ) :
    ∑ j ∈ Finset.range l.length, f (l.getD j 0) = (l.map f).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.length_cons, Finset.sum_range_succ', List.map_cons, List.sum_cons]
    simp only [List.getD_cons_succ, List.getD_cons_zero, ih]
    omega

/-- The gates of `G` some block of session `i` needs are at most the needs of the blocks at its slots, summed. -/
theorem card_filter_sess_le (L : Compiled Coin A) (x : Msg) (l : List ℕ) (needs : ℕ → Finset ℕ) (i : ℕ) {N : ℕ}
    (G : Finset (Fin N)) :
    (G.filter fun g => ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u).card ≤
      ∑ j ∈ (L.sessSlots x l i).toFinset, (needs (l.getD j 0)).card := by
  have hsub : (G.filter fun g => ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u).map Fin.valEmbedding ⊆
      (L.sessSlots x l i).toFinset.biUnion fun j => needs (l.getD j 0) := by
    intro y hy
    obtain ⟨g, hg, rfl⟩ := Finset.mem_map.1 hy
    obtain ⟨-, u, hu, hgu⟩ := Finset.mem_filter.1 hg
    obtain ⟨j, hj, hju⟩ := List.mem_filterMap.1 hu
    refine Finset.mem_biUnion.2 ⟨j, List.mem_toFinset.2 hj, ?_⟩
    rw [List.getD_eq_getElem?_getD, hju, Option.getD_some]
    exact hgu
  calc (G.filter fun g => ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u).card
      = ((G.filter fun g => ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u).map Fin.valEmbedding).card :=
        (Finset.card_map _).symm
    _ ≤ ((L.sessSlots x l i).toFinset.biUnion fun j => needs (l.getD j 0)).card := Finset.card_le_card hsub
    _ ≤ ∑ j ∈ (L.sessSlots x l i).toFinset, (needs (l.getD j 0)).card := Finset.card_biUnion_le

/-- Distinct sessions' first slots hold disjoint slots. -/
theorem sessSlots_disjoint (L : Compiled Coin A) (x : Msg) (l : List ℕ) (F : Finset ℕ)
    (hF : ∀ i ∈ F, (L.sessSlots x l i).head? = some i) :
    (F : Set ℕ).PairwiseDisjoint fun i => (L.sessSlots x l i).toFinset := by
  intro i hi i' hi' hne
  rw [Function.onFun, Finset.disjoint_left]
  intro j hj hj'
  obtain ⟨u, -, -, hk⟩ := mem_sessSlots.1 (List.mem_toFinset.1 hj)
  obtain ⟨u', -, -, hk'⟩ := mem_sessSlots.1 (List.mem_toFinset.1 hj')
  exact hne (first_eq_of_key (hF i (Finset.mem_coe.1 hi)) (hF i' (Finset.mem_coe.1 hi')) (hk.symm.trans hk'))

/-- **The sessions' weights at the gates their blocks need, counted by block**: for sessions named by their first
slots `F` in the sample `l`, a weight `e i ≤ e_max` on each, and gates `G`, summing over `G` the weights of each gate's
sessions (those some block of which needs it, as a run's extractions at the gate run them) gives at most `e_max` times
the sample's blocks' needs, summed. -/
theorem sum_sessions_le (L : Compiled Coin A) (x : Msg) (l : List ℕ) (needs : ℕ → Finset ℕ) (F : Finset ℕ)
    (hF : ∀ i ∈ F, (L.sessSlots x l i).head? = some i) (e : ℕ → ℕ) (emax : ℕ) (he : ∀ i ∈ F, e i ≤ emax)
    {N : ℕ} (G : Finset (Fin N)) :
    ∑ g ∈ G, ∑ i ∈ F, (if ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u then e i else 0) ≤
      emax * (l.map fun u => (needs u).card).sum := by
  have hrow : ∀ i ∈ F, ∑ g ∈ G, (if ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u then e i else 0) ≤
      (∑ j ∈ (L.sessSlots x l i).toFinset, (needs (l.getD j 0)).card) * emax := by
    intro i hi
    rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
    exact Nat.mul_le_mul (L.card_filter_sess_le x l needs i G) (he i hi)
  have hsub : F.biUnion (fun i => (L.sessSlots x l i).toFinset) ⊆ Finset.range l.length := by
    intro j hj
    obtain ⟨i, -, hj⟩ := Finset.mem_biUnion.1 hj
    obtain ⟨u, hu, -, -⟩ := mem_sessSlots.1 (List.mem_toFinset.1 hj)
    obtain ⟨hlt, -⟩ := List.getElem?_eq_some_iff.1 hu
    exact Finset.mem_range.2 hlt
  rw [Finset.sum_comm]
  calc ∑ i ∈ F, ∑ g ∈ G, (if ∃ u ∈ L.sessBlocks x l i, g.val ∈ needs u then e i else 0)
      ≤ ∑ i ∈ F, (∑ j ∈ (L.sessSlots x l i).toFinset, (needs (l.getD j 0)).card) * emax :=
        Finset.sum_le_sum hrow
    _ = (∑ j ∈ F.biUnion (fun i => (L.sessSlots x l i).toFinset), (needs (l.getD j 0)).card) * emax := by
        rw [Finset.sum_biUnion (L.sessSlots_disjoint x l F hF), Finset.sum_mul]
    _ ≤ (∑ j ∈ Finset.range l.length, (needs (l.getD j 0)).card) * emax :=
        Nat.mul_le_mul_right _ (Finset.sum_le_sum_of_subset hsub)
    _ = emax * (l.map fun u => (needs u).card).sum := by
        rw [sum_range_getD_map l (fun u => (needs u).card), Nat.mul_comm]

end Compiled

end Claims.SamplingGeneric
