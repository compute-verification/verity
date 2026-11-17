module

public import Definitions.InferenceOnly.Assumptions.Theta
public import Proofs.Core.Circuit

/-! Pulling a division back through a containment, into pieces that read no more fresh input than their inference
blocks. -/

@[expose] public section

open Slowdown

namespace InferenceOnly.SecurityProofs

open Verity

section Pull

variable {C D : Circuit} (M : Containment C D) {n : ℕ} (P : Partition D n)

def pull : Partition C n where
  unit g := P.unit (M.emb g)
  input_iff g := by rw [P.input_iff, M.isInput_emb]

variable {M P}

theorem mem_pull_gates {u : Fin n} {g : Fin C.N} : g ∈ (pull M P).gates u ↔ M.emb g ∈ P.gates u := by
  rw [Partition.mem_gates, Partition.mem_gates]
  rfl

theorem pull_gates_map (u : Fin n) : ((pull M P).gates u).map ⟨M.emb, M.emb_inj⟩ ⊆ P.gates u := by
  intro b hb
  obtain ⟨g, hg, rfl⟩ := Finset.mem_map.1 hb
  exact mem_pull_gates.1 hg

theorem card_pull_gates_le (u : Fin n) : ((pull M P).gates u).card ≤ (P.gates u).card := by
  rw [← Finset.card_map ⟨M.emb, M.emb_inj⟩]
  exact Finset.card_le_card (pull_gates_map u)

theorem width_map (S : Finset (Fin C.N)) : C.width S = D.width (S.map ⟨M.emb, M.emb_inj⟩) := by
  simp [Circuit.width, Finset.sum_map, M.emb_w]

theorem into_pull_map (u : Fin n) :
    (C.into ((pull M P).gates u)).map ⟨M.emb, M.emb_inj⟩ ⊆ D.into (P.gates u) := by
  intro b hb
  obtain ⟨a, ha, rfl⟩ := Finset.mem_map.1 hb
  simp only [Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
  obtain ⟨hna, g, hg, hag⟩ := ha
  have hgi : C.isInput g = false := by
    rw [mem_pull_gates, Partition.mem_gates] at hg
    cases h : C.isInput g
    · rfl
    · rw [← M.isInput_emb, ← P.input_iff] at h
      rw [h] at hg
      cases hg
  refine ⟨fun h => hna (mem_pull_gates.2 h), M.emb g, mem_pull_gates.1 hg, ?_⟩
  rw [M.reads_emb g hgi]
  exact Finset.mem_map_of_mem _ hag

theorem into_pull_into_computed (u : Fin n) {a : Fin C.N} (ha : a ∈ C.into ((pull M P).gates u))
    (hai : C.isInput a = true) : a ∈ C.into C.computed := by
  simp only [Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and] at ha ⊢
  obtain ⟨-, g, hg, hag⟩ := ha
  have hgi : C.isInput g = false := by
    rw [mem_pull_gates, Partition.mem_gates] at hg
    cases h : C.isInput g
    · rfl
    · rw [← M.isInput_emb, ← P.input_iff] at h
      rw [h] at hg
      cases hg
  refine ⟨fun h => ?_, g, ?_, hag⟩
  · simp [Circuit.computed, hai] at h
  · simp [Circuit.computed, hgi]

end Pull

theorem freshInput_pull {c d : RegisteredCircuit} (M : Containment c.D d.D)
    (hB : Assumptions.ThetaRegistered c d (Slowdown.PartialContainment.ofContainment M)) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : freshInput c ((pull M P).gates u) ≤ freshInput d (P.gates u) := by
  unfold freshInput
  rw [width_map (M := M)]
  refine Finset.sum_le_sum_of_subset fun b hb => ?_
  obtain ⟨a, ha, rfl⟩ := Finset.mem_map.1 hb
  rw [Finset.mem_sdiff] at ha ⊢
  refine ⟨into_pull_map u (Finset.mem_map_of_mem _ ha.1), fun hθd => ha.2 (hB a ?_ hθd)⟩
  refine into_pull_into_computed u ha.1 ?_
  rw [← M.isInput_emb]
  exact d.θ_input _ hθd

end InferenceOnly.SecurityProofs
