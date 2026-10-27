import Definitions.InferenceOnly.Reads
import Proofs.InferenceOnly.Pull

/-!
# Reads in a set of gates, and the pull into the contained circuit

* `novelReads_le_readsIn`: what a set reads novelly is at most all it reads, `readsIn d R univ`.
* `novelReads_pull_le_readsIn`: through a containment `M : c ≼ d`, a piece of a unit of `d` reads novelly at most what
  its unit reads in `M`'s image, baseline included, so no condition on the baselines is needed: a value the piece reads
  is a gate of `c` outside the piece or an input of `c`, and `M` sends either to a gate of its image that the unit reads.
* `novelReads_piece_le_readsIn`: the same through a partial containment, for the piece of `S` a unit computes, with the
  image of `S` and of the gates `S` reads in place of `M`'s image.
-/

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem novelReads_le_readsIn (d : Declaration) (R : Finset (Fin d.D.N)) : novelReads d R ≤ readsIn d R univ := by
  unfold novelReads readsIn
  rw [inter_univ]
  exact sum_le_sum_of_subset sdiff_subset

/-- **The pull into the contained circuit**: through a containment `M : c ≼ d`, the piece of `c` that a unit of `d`
induces (`pull`) reads at most what the unit reads in `M`'s image, baseline included. -/
theorem novelReads_pull_le_readsIn {c d : Declaration} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c ((pull M P).gates u) ≤ readsIn d (P.gates u) (univ.map ⟨M.emb, M.emb_inj⟩) := by
  unfold novelReads readsIn
  calc c.D.width (c.D.into ((pull M P).gates u) \ c.θ) ≤ c.D.width (c.D.into ((pull M P).gates u)) :=
        sum_le_sum_of_subset sdiff_subset
    _ = d.D.width ((c.D.into ((pull M P).gates u)).map ⟨M.emb, M.emb_inj⟩) := width_map (M := M) _
    _ ≤ d.D.width (d.D.into (P.gates u) ∩ univ.map ⟨M.emb, M.emb_inj⟩) := by
        refine sum_le_sum_of_subset fun b hb => ?_
        obtain ⟨a, -, rfl⟩ := mem_map.1 hb
        exact mem_inter.2 ⟨into_pull_map u hb, mem_map_of_mem _ (mem_univ a)⟩

/-- **The pull through a partial containment**: the piece of `c` that a unit of `d` computes (`piece`) reads novelly at
most what the unit reads in the image of `S` and of the gates `S` reads, baseline included: a value the piece reads is a
gate of `S` or a gate outside `S` that `S` reads, and `M` sends either to a gate the unit reads, of the same width. -/
theorem novelReads_piece_le_readsIn {c d : Declaration} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c (piece M P u) ≤ readsIn d (P.gates u) ((M.S ∪ c.D.into M.S).image M.emb) := by
  have hA : ∀ a ∈ c.D.into (piece M P u) \ c.θ, a ∈ M.S ∪ c.D.into M.S := fun a ha =>
    into_piece_subset u (mem_sdiff.1 ha).1
  have hmaps : ∀ a ∈ c.D.into (piece M P u) \ c.θ,
      M.emb a ∈ d.D.into (P.gates u) ∩ (M.S ∪ c.D.into M.S).image M.emb := by
    intro a ha
    refine mem_inter.2 ⟨?_, mem_image_of_mem _ (hA a ha)⟩
    have hain := (mem_sdiff.1 ha).1
    simp only [Circuit.into, mem_filter, mem_univ, true_and] at hain
    obtain ⟨hna, g, hg, hag⟩ := hain
    obtain ⟨hgS, hgu⟩ := mem_piece.1 hg
    simp only [Circuit.into, mem_filter, mem_univ, true_and]
    refine ⟨fun hau => ?_, M.emb g, hgu, ?_⟩
    · by_cases haS : a ∈ M.S
      · exact hna (mem_piece.2 ⟨haS, hau⟩)
      · have hc := P.gates_subset_computed u hau
        simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hc
        rw [M.input_emb a ((mem_union.1 (hA a ha)).resolve_left haS)] at hc
        exact Bool.false_ne_true hc.symm
    · rw [M.reads_emb g hgS]
      exact mem_image_of_mem _ hag
  unfold novelReads readsIn Circuit.width
  calc ∑ a ∈ c.D.into (piece M P u) \ c.θ, c.D.w a
      = ∑ b ∈ (c.D.into (piece M P u) \ c.θ).image M.emb, d.D.w b := by
        rw [sum_image fun a ha b hb h => M.emb_inj a (hA a (mem_coe.1 ha)) b (hA b (mem_coe.1 hb)) h]
        exact sum_congr rfl fun a ha => (M.emb_w a (hA a ha)).symm
    _ ≤ ∑ b ∈ d.D.into (P.gates u) ∩ (M.S ∪ c.D.into M.S).image M.emb, d.D.w b :=
        sum_le_sum_of_subset (image_subset_iff.2 hmaps)

end InferenceOnly.SecurityProofs
