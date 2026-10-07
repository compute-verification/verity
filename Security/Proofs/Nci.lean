import Models.Nci.Guarantees
import Proofs.Nci.Inequalities
import Proofs.Nci.LoomisWhitney
import Proofs.Core
import Proofs.Nci.Chain
import Proofs.Nci.Throttling

/-!
# Proofs of NCI's guarantees

The read bound intersects each IU of the containing declaration with the contained circuit's gates (`pull`): the
pieces divide `c` into IUs no larger, and each piece reads no more novel information than its IU, since a novel value it
reads is either a computed gate of `c` outside the piece or a novel input gate of `c`, and the containment carries both
to novel values its IU reads.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Pull

variable {C D : Circuit} (M : Containment C D) {n : ℕ} (P : Partition D n)

/-- The division of `C` that a division of `D` induces through a containment: each gate in the unit of its image. -/
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

/-- A gate outside a piece that the piece reads is, through the containment, a gate outside its IU that the IU reads. -/
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

end Pull

theorem novelReads_pull {c d : Declaration} (M : Containment c.D d.D)
    (hθ : ∀ g, c.D.isInput g = true → M.emb g ∈ d.θ → g ∈ c.θ) {n : ℕ} (P : Partition d.D n) (u : Fin n) :
    novelReads c ((pull M P).gates u) ≤ novelReads d (P.gates u) := by
  unfold novelReads
  rw [width_map (M := M)]
  refine Finset.sum_le_sum_of_subset fun b hb => ?_
  obtain ⟨a, ha, rfl⟩ := Finset.mem_map.1 hb
  rw [Finset.mem_sdiff] at ha ⊢
  refine ⟨into_pull_map u (Finset.mem_map_of_mem _ ha.1), fun hθd => ha.2 (hθ a ?_ hθd)⟩
  rw [← M.isInput_emb]
  exact d.θ_input _ hθd

theorem ReadBound : Guarantees.ReadBound := by
  intro Γ α c d ⟨M, hθ⟩ ⟨n, P, hP⟩
  have hsize : ∀ u, c.D.size ((pull M P).gates u) ≤ Γ := fun u => (card_pull_gates_le u).trans (hP u).1
  have hle : Reads Γ c ≤ ∑ u, novelReads c ((pull M P).gates u) := Nat.sInf_le ⟨n, pull M P, hsize, rfl⟩
  calc (Reads Γ c : ℝ) ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) := by exact_mod_cast hle
    _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) :=
        Finset.sum_le_sum fun u _ => by exact_mod_cast novelReads_pull M hθ P u
    _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := Finset.sum_le_sum fun u _ => (hP u).2
    _ = α * d.D.size d.D.computed := by
        rw [← Finset.mul_sum]
        congr 1
        exact_mod_cast P.sum_card_gates

theorem Inference : Guarantees.Inference := by
  intro Γ α d n P prompt hsize hreads hprompt
  have hc : Complies Γ α d := by
    refine ⟨n, P, fun u => ⟨hsize u, le_trans ?_ (hprompt u)⟩⟩
    unfold novelReads
    refine Nat.cast_le.2 (Finset.sum_le_sum_of_subset fun a ha => ?_)
    rw [Finset.mem_sdiff] at ha
    exact (Finset.mem_union.1 (hreads u ha.1)).resolve_left ha.2
  refine ⟨hc, le_trans (iInf₂_le d ⟨Verity.SecurityProofs.ContainsImplements d d ⟨Containment.refl d.D,
    fun _ _ h => h⟩, hc⟩) ENNReal.div_self_le_one⟩

end Nci.SecurityProofs
