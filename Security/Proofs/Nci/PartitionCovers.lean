import Definitions.Nci.Guarantees.PartitionCovers
import Proofs.Core.Circuit

/-!
# Proof of Lemma A.1: a `(c, k)`-partition's completed proof blocks cover the circuit

Proof block `u`'s completion `I ∪ in(I)` with inputs `in(I)` is a subcircuit: a gate of `I` reads only gates of `I ∪ in(I)`,
and a source in it is in `in(I)`, since every gate of `I` is an operation gate and so reads some gate (positive fan-in).
It computes exactly `I`, so the completions compute every operation gate.
-/

namespace Nci.SecurityProofs

open Verity Nci.Covering Finset

/-- Proof block `u`'s completion `I ∪ in(I)`, with inputs `in(I)`, on a circuit of positive fan-in. -/
def completion {d : Declaration} {B : ℕ} (P : Partition d.D B)
    (hfan : ∀ g, d.D.isInput g = false → (d.D.reads g).Nonempty) (u : Fin B) : Subcircuit d.D where
  gates := P.gates u ∪ d.D.into (P.gates u)
  ins := d.D.into (P.gates u)
  ins_sub := subset_union_right
  closed := by
    intro g hg hgi x hx
    have hgu : g ∈ P.gates u := (mem_union.1 hg).resolve_right hgi
    by_cases hxu : x ∈ P.gates u
    · exact mem_union_left _ hxu
    · exact mem_union_right _ (by simp only [Circuit.into, mem_filter, mem_univ, true_and]; exact ⟨hxu, g, hgu, hx⟩)
  sources_ins := by
    intro g hg hs
    rcases mem_union.1 hg with hgu | hgi
    · have hi : d.D.isInput g = false := by simpa [Circuit.computed] using P.gates_subset_computed u hgu
      have hne := hfan g hi
      simp only [sources, mem_filter, mem_univ, true_and] at hs
      rw [hs] at hne
      exact absurd hne (by simp)
    · exact hgi

theorem PartitionCovers : Guarantees.PartitionCovers := by
  intro d B c k P hP
  refine ⟨completion P hP.boolean.fanin_pos, ⟨fun u => ⟨?_, hP.reads u⟩, ?_⟩, fun u => ⟨rfl, rfl⟩⟩
  · calc (P.gates u ∪ d.D.into (P.gates u)).card ≤ (P.gates u).card + (d.D.into (P.gates u)).card :=
          card_union_le _ _
      _ ≤ c + univ.sup fun u => (d.D.into (P.gates u)).card := by
          rw [hP.card u]
          exact Nat.add_le_add_left (le_sup (f := fun u => (d.D.into (P.gates u)).card) (mem_univ u)) c
  · intro g hg
    have hi : d.D.isInput g = false := by
      cases h : d.D.isInput g
      · rfl
      · exact absurd (by simp [sources, d.D.input_reads g h]) hg
    obtain ⟨u, hu⟩ : ∃ u, P.unit g = some u :=
      Option.ne_none_iff_exists'.1 fun h => by simp [(P.input_iff g).1 h] at hi
    refine ⟨u, ?_⟩
    simp [Subcircuit.inner, completion, Circuit.into, P.mem_gates, hu]

end Nci.SecurityProofs
