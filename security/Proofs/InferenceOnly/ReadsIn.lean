module

public import Definitions.InferenceOnly.Reads
public import Proofs.InferenceOnly.Pull
public import Proofs.Slowdown.Threads

/-! What a set of gates reads bounds its fresh input, and a piece pulled into the contained circuit reads at most
what its unit reads in the image. -/

@[expose] public section

open Slowdown Slowdown.SecurityProofs

namespace InferenceOnly.SecurityProofs

open Verity Finset

theorem freshInput_le_readsIn (d : RegisteredCircuit) (R : Finset (Fin d.D.N)) : freshInput d R ≤ readsIn d R univ := by
  unfold freshInput readsIn
  rw [inter_univ]
  exact sum_le_sum_of_subset sdiff_subset

theorem freshInput_pull_le_readsIn {c d : RegisteredCircuit} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : freshInput c ((pull M P).gates u) ≤ readsIn d (P.gates u) (univ.map ⟨M.emb, M.emb_inj⟩) := by
  unfold freshInput readsIn
  calc c.D.width (c.D.into ((pull M P).gates u) \ c.θ) ≤ c.D.width (c.D.into ((pull M P).gates u)) :=
        sum_le_sum_of_subset sdiff_subset
    _ = d.D.width ((c.D.into ((pull M P).gates u)).map ⟨M.emb, M.emb_inj⟩) := width_map (M := M) _
    _ ≤ d.D.width (d.D.into (P.gates u) ∩ univ.map ⟨M.emb, M.emb_inj⟩) := by
        refine sum_le_sum_of_subset fun b hb => ?_
        obtain ⟨a, -, rfl⟩ := mem_map.1 hb
        exact mem_inter.2 ⟨into_pull_map u hb, mem_map_of_mem _ (mem_univ a)⟩

theorem freshInput_piece_le_readsIn {c d : RegisteredCircuit} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : freshInput c (piece M P u) ≤ readsIn d (P.gates u) ((M.S ∪ c.D.into M.S).image M.emb) := by
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
  unfold freshInput readsIn Circuit.width
  calc ∑ a ∈ c.D.into (piece M P u) \ c.θ, c.D.w a
      = ∑ b ∈ (c.D.into (piece M P u) \ c.θ).image M.emb, d.D.w b := by
        rw [sum_image fun a ha b hb h => M.emb_inj a (hA a (mem_coe.1 ha)) b (hA b (mem_coe.1 hb)) h]
        exact sum_congr rfl fun a ha => (M.emb_w a (hA a ha)).symm
    _ ≤ ∑ b ∈ d.D.into (P.gates u) ∩ (M.S ∪ c.D.into M.S).image M.emb, d.D.w b :=
        sum_le_sum_of_subset (image_subset_iff.2 hmaps)

end InferenceOnly.SecurityProofs
