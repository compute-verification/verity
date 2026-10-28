import Definitions.Nci.Assumptions.Baseline
import Proofs.Core.Circuit

/-!
# Pulling a division back through a containment

Each IU of a containing declaration, intersected with the contained circuit's gates, is a piece of the contained circuit
(`pull`): the pieces divide `c` into IUs no larger, and each piece reads no more novel information than its IU, since a
novel value it reads is either a computed gate of `c` outside the piece or a novel input gate of `c`, and the containment
carries both to novel values its IU reads.

Through a partial containment the pieces (`piece`) divide only the gates `S` that `d` computes, and the dark gates are in
none. The same holds of each piece, with one more case for what it reads (`novelReads_piece`): a dark value enters `d` as
an input gate, which lies in no IU, so the IU reads it too, and it is outside `d`'s θ by `ThetaRegistered`. A
total containment is a partial one with no dark gates (`PartialContainment.ofContainment`), and `ThetaRegistered` for
it keeps the novel inputs of `c` that a piece reads out of `d`'s θ (`novelReads_pull`).
-/

namespace Nci.SecurityProofs

open Verity

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

/-- An input gate that a piece reads is read by `C`'s computed gates. -/
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

theorem novelReads_pull {c d : RegisteredCircuit} (M : Containment c.D d.D)
    (hB : Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M)) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c ((pull M P).gates u) ≤ novelReads d (P.gates u) := by
  unfold novelReads
  rw [width_map (M := M)]
  refine Finset.sum_le_sum_of_subset fun b hb => ?_
  obtain ⟨a, ha, rfl⟩ := Finset.mem_map.1 hb
  rw [Finset.mem_sdiff] at ha ⊢
  refine ⟨into_pull_map u (Finset.mem_map_of_mem _ ha.1), fun hθd => ha.2 (hB a ?_ hθd)⟩
  refine into_pull_into_computed u ha.1 ?_
  rw [← M.isInput_emb]
  exact d.θ_input _ hθd

section Piece

open Finset

theorem ofContainment_S {c d : RegisteredCircuit} (M : Containment c.D d.D) :
    (PartialContainment.ofContainment M).S = c.D.computed := rfl

variable {c d : RegisteredCircuit} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)

/-- The piece of `c` that IU `u` of `d` computes: the gates of `S` whose images lie in `u`. -/
def piece (u : Fin n) : Finset (Fin c.D.N) := M.S.filter fun g => P.unit (M.emb g) = some u

variable {M P}

theorem mem_dark {g : Fin c.D.N} : g ∈ M.dark ↔ g ∈ c.D.computed ∧ g ∉ M.S := mem_sdiff

/-- Computed gates outside `S` are dark. -/
theorem card_le_dark_add_inter (M : PartialContainment c d) {A : Finset (Fin c.D.N)} (hA : A ⊆ c.D.computed) :
    A.card ≤ M.dark.card + (A ∩ M.S).card := by
  rw [← card_sdiff_add_card_inter A M.S]
  exact Nat.add_le_add_right (card_le_card (Finset.sdiff_subset_sdiff hA (subset_refl _))) _

theorem mem_piece {u : Fin n} {g : Fin c.D.N} : g ∈ piece M P u ↔ g ∈ M.S ∧ M.emb g ∈ P.gates u := by
  rw [piece, mem_filter, Partition.mem_gates]

theorem piece_subset (u : Fin n) : piece M P u ⊆ M.S := filter_subset _ _

theorem piece_subset_computed (u : Fin n) : piece M P u ⊆ c.D.computed := (piece_subset u).trans M.S_computed

/-- A gate outside a piece that the piece reads is in `S`, or outside `S` and read by it. -/
theorem into_piece_subset (u : Fin n) : c.D.into (piece M P u) ⊆ M.S ∪ c.D.into M.S := by
  intro a ha
  simp only [Circuit.into, mem_filter, mem_univ, true_and] at ha
  obtain ⟨-, g, hg, hag⟩ := ha
  by_cases haS : a ∈ M.S
  · exact mem_union_left _ haS
  · refine mem_union_right _ ?_
    simp only [Circuit.into, mem_filter, mem_univ, true_and]
    exact ⟨haS, g, piece_subset u hg, hag⟩

/-- A piece has no more gates than its IU: `emb` is injective on `S`. -/
theorem card_piece_le (u : Fin n) : (piece M P u).card ≤ (P.gates u).card :=
  card_le_card_of_injOn M.emb (fun g hg => mem_coe.2 (mem_piece.1 (mem_coe.1 hg)).2)
    fun a ha b hb h => M.emb_inj a (mem_union_left _ (piece_subset u (mem_coe.1 ha))) b
      (mem_union_left _ (piece_subset u (mem_coe.1 hb))) h

/-- **A piece reads no more novel information than its IU.** A novel value the piece reads is a gate of `S` outside the
piece, whose image is a computed gate outside the IU, or a gate outside `S`, whose image is an input gate of `d` outside
its θ (`ThetaRegistered`) and so in no IU; either way the IU reads the image, which is as wide. -/
theorem novelReads_piece (hB : Assumptions.ThetaRegistered c d M) (u : Fin n) :
    novelReads c (piece M P u) ≤ novelReads d (P.gates u) := by
  have hA : ∀ a ∈ c.D.into (piece M P u) \ c.θ, a ∈ M.S ∪ c.D.into M.S := fun a ha =>
    into_piece_subset u (mem_sdiff.1 ha).1
  have hmaps : ∀ a ∈ c.D.into (piece M P u) \ c.θ, M.emb a ∈ d.D.into (P.gates u) \ d.θ := by
    intro a ha
    obtain ⟨hain, haθ⟩ := mem_sdiff.1 ha
    have hain' := hain
    simp only [Circuit.into, mem_filter, mem_univ, true_and] at hain'
    obtain ⟨hna, g, hg, hag⟩ := hain'
    obtain ⟨hgS, hgu⟩ := mem_piece.1 hg
    refine mem_sdiff.2 ⟨?_, fun hθ => ?_⟩
    · simp only [Circuit.into, mem_filter, mem_univ, true_and]
      refine ⟨fun hau => ?_, M.emb g, hgu, ?_⟩
      · by_cases haS : a ∈ M.S
        · exact hna (mem_piece.2 ⟨haS, hau⟩)
        · have hc := P.gates_subset_computed u hau
          simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hc
          rw [M.input_emb a ((mem_union.1 (hA a ha)).resolve_left haS)] at hc
          exact Bool.false_ne_true hc.symm
      · rw [M.reads_emb g hgS]
        exact mem_image_of_mem _ hag
    · by_cases haS : a ∈ M.S
      · have hi := d.θ_input _ hθ
        rw [M.computed_emb a haS] at hi
        exact Bool.false_ne_true hi
      · exact haθ (hB a ((mem_union.1 (hA a ha)).resolve_left haS) hθ)
  unfold novelReads Circuit.width
  calc ∑ a ∈ c.D.into (piece M P u) \ c.θ, c.D.w a
      = ∑ b ∈ (c.D.into (piece M P u) \ c.θ).image M.emb, d.D.w b := by
        rw [sum_image fun a ha b hb h => M.emb_inj a (hA a (mem_coe.1 ha)) b (hA b (mem_coe.1 hb)) h]
        exact sum_congr rfl fun a ha => (M.emb_w a (hA a ha)).symm
    _ ≤ ∑ b ∈ d.D.into (P.gates u) \ d.θ, d.D.w b := sum_le_sum_of_subset (image_subset_iff.2 hmaps)

/-- **The pieces divide `S`**: a gate of `S` is in the piece of its image's IU, and in no other. -/
theorem sum_card_filter_piece {β : Type*} (B : Finset β) (f : β → Fin c.D.N) :
    ∑ u, (B.filter fun b => f b ∈ piece M P u).card = (B.filter fun b => f b ∈ M.S).card := by
  rw [card_eq_sum_card_fiberwise (s := B.filter fun b => f b ∈ M.S) (f := fun b => P.unit (M.emb (f b)))
    (t := univ) fun _ _ => mem_univ _, Fintype.sum_option]
  have hnone : ((B.filter fun b => f b ∈ M.S).filter fun b => P.unit (M.emb (f b)) = none) = ∅ := by
    refine filter_eq_empty_iff.2 fun b hb h => ?_
    have hi := (P.input_iff _).1 h
    rw [M.computed_emb _ (mem_filter.1 hb).2] at hi
    exact Bool.false_ne_true hi
  rw [hnone, card_empty, zero_add]
  refine sum_congr rfl fun u _ => congrArg card ?_
  ext b
  simp only [mem_filter, piece, and_assoc]

end Piece

end Nci.SecurityProofs
