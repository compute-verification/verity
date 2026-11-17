module

public import Definitions.InferenceOnly.Assumptions.Theta
public import Proofs.Core.Circuit

/-!
# Pieces of a partial containment

Through a partial containment the pieces (`piece`) divide only the gates `S` that `d` computes, and the dark gates are in
none. The same holds of each piece, with one more case for what it reads (`freshInput_piece`): a dark value enters `d` as
an input gate, which lies in no inference block, so the inference block reads it too, and it is outside `d`'s θ by `ThetaRegistered`.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.SecurityProofs

open Verity

section Piece

open Finset

theorem ofContainment_S {c d : RegisteredCircuit} (M : Containment c.D d.D) :
    (PartialContainment.ofContainment M).S = c.D.computed := rfl

variable {c d : RegisteredCircuit} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)

/-- The piece of `c` that inference block `u` of `d` computes: the gates of `S` whose images lie in `u`. -/
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

/-- A piece has no more gates than its inference block: `emb` is injective on `S`. -/
theorem card_piece_le (u : Fin n) : (piece M P u).card ≤ (P.gates u).card :=
  card_le_card_of_injOn M.emb (fun g hg => mem_coe.2 (mem_piece.1 (mem_coe.1 hg)).2)
    fun a ha b hb h => M.emb_inj a (mem_union_left _ (piece_subset u (mem_coe.1 ha))) b
      (mem_union_left _ (piece_subset u (mem_coe.1 hb))) h

/-- **A piece reads no more fresh input than its inference block.** A fresh input value the piece reads is a gate of `S` outside the
piece, whose image is a computed gate outside the inference block, or a gate outside `S`, whose image is an input gate of `d` outside
its θ (`ThetaRegistered`) and so in no inference block; either way the inference block reads the image, which is as wide. -/
theorem freshInput_piece (hB : Assumptions.ThetaRegistered c d M) (u : Fin n) :
    freshInput c (piece M P u) ≤ freshInput d (P.gates u) := by
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
  unfold freshInput Circuit.width
  calc ∑ a ∈ c.D.into (piece M P u) \ c.θ, c.D.w a
      = ∑ b ∈ (c.D.into (piece M P u) \ c.θ).image M.emb, d.D.w b := by
        rw [sum_image fun a ha b hb h => M.emb_inj a (hA a (mem_coe.1 ha)) b (hA b (mem_coe.1 hb)) h]
        exact sum_congr rfl fun a ha => (M.emb_w a (hA a ha)).symm
    _ ≤ ∑ b ∈ d.D.into (P.gates u) \ d.θ, d.D.w b := sum_le_sum_of_subset (image_subset_iff.2 hmaps)

/-- **The pieces divide `S`**: a gate of `S` is in the piece of its image's inference block, and in no other. -/
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

end Slowdown.SecurityProofs
