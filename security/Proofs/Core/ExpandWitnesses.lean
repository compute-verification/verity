module

public import Definitions.Core.Guarantees.Expand.Witnesses
public import Proofs.Core.AndSize

/-!
# The bit-level expansion's size bounds: honest and nontrivial lemmas

The proofs of `Security/Definitions/Core/Guarantees/Expand/Witnesses.lean`, at `Verity.Expand.AndPair`
(`Security/Definitions/Core/Expand/Instances.lean`). The honest lemmas are `ExpandSize` and `ExpandAndSize` there. The
nontrivial ones show that both bounds hold with equality there:

* `RegisteredCircuit.width_unregistered_le_expand`: when every gate of a unit has a bit, the bits of the unit's unregistered
  reads are unregistered reads of its expansion, so `ExpandSize`'s bound on them is an equality;
* `Expand.AndPair.isAnd`: each bit of `AndPair`'s gate 2 is the AND of a bit of gate 0 and one of gate 1, so it is not
  affine (`Circuit.not_affine_of_and`), and the expansion's AND count is its number of computed bits.
-/

@[expose] public section

namespace Verity

open Finset

theorem Circuit.expand_op_bit (C : Circuit) (g : Fin C.N) (i : Fin (C.w g))
    (v : (a : Fin C.expand.N) → BitVec (C.expand.w a)) :
    C.expand.op (C.bit g i) v =
      BitVec.ofBool ((C.op g fun a => ofBits fun j => (v (C.bit a j)).getLsbD 0).getLsbD i) := by
  show BitVec.ofBool ((C.op (C.unbit (C.bit g i)).1 fun a => ofBits fun j => (v (C.bit a j)).getLsbD 0).getLsbD
    (C.unbit (C.bit g i)).2) = _
  rw [C.unbit_bit]

theorem Circuit.bitZ_ofBool (b : Bool) : bitZ (BitVec.ofBool b) = if b then 1 else 0 := by
  cases b <;> rfl

theorem getLsbD_ofBool_zero (b : Bool) : (BitVec.ofBool b).getLsbD 0 = b := by
  cases b <;> rfl

/-- The bits of a unit's unregistered reads are unregistered reads of its expansion, when each gate of the unit has a
bit to read them. -/
theorem RegisteredCircuit.bits_unregistered_subset (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (u : Fin B)
    (hw : ∀ g ∈ P.gates u, 0 < d.D.w g) :
    univ.filter (fun b => (d.D.unbit b).1 ∈ d.unregistered (P.gates u)) ⊆ d.expand.unregistered (P.expand.gates u) := by
  intro b' hb'
  obtain ⟨-, hb'⟩ := mem_filter.1 hb'
  obtain ⟨hin, hθ⟩ := mem_sdiff.1 hb'
  obtain ⟨-, hout, g, hg, hr⟩ := mem_filter.1 hin
  have hi : d.D.isInput (d.D.unbit (d.D.bit g ⟨0, hw g hg⟩)).1 = false := by
    rw [Circuit.unbit_bit]
    exact P.isInput_of_mem_gates hg
  refine mem_sdiff.2 ⟨mem_filter.2 ⟨mem_univ _, fun h => hout ?_, d.D.bit g ⟨0, hw g hg⟩, ?_, ?_⟩,
    fun h => hθ (mem_filter.1 h).2⟩
  · rw [P.expand_gates, mem_filter] at h
    exact h.2
  · rw [P.expand_gates, mem_filter, Circuit.unbit_bit]
    exact ⟨mem_univ _, hg⟩
  · exact (d.D.mem_expand_reads hi).2 (by rw [Circuit.unbit_bit]; exact hr)

/-- When each gate of a unit has a bit, its expansion's unregistered reads are at least as many as the width of its own:
`ExpandSize`'s bound on them is an equality. -/
theorem RegisteredCircuit.width_unregistered_le_expand (d : RegisteredCircuit) {B : ℕ} (P : Partition d.D B) (u : Fin B)
    (hw : ∀ g ∈ P.gates u, 0 < d.D.w g) :
    d.D.width (d.unregistered (P.gates u)) ≤ (d.expand.unregistered (P.expand.gates u)).card := by
  calc d.D.width (d.unregistered (P.gates u))
      = (univ.filter fun b => (d.D.unbit b).1 ∈ d.unregistered (P.gates u)).card := by
        rw [d.D.card_filter_unbit (· ∈ d.unregistered (P.gates u)), sum_univ_filter_mem]
        rfl
    _ ≤ _ := card_le_card (d.bits_unregistered_subset P u hw)

namespace Expand.AndPair

theorem op_two (V : (a : Fin d.D.N) → BitVec (d.D.w a)) : d.D.op 2 V = V 0 &&& V 1 := rfl

/-- Each bit of the expansion's computed gates is an AND gate: bit `i` of gate 2 is bit `i` of gate 0 and bit `i` of
gate 1. -/
theorem isAnd : ∀ b ∈ d.D.expand.computed, d.D.expand.IsAnd b := by
  intro b hb
  obtain ⟨g, i, rfl⟩ : ∃ g i, d.D.bit g i = b := ⟨_, _, d.D.bit_unbit b⟩
  have hi : d.D.isInput (d.D.unbit (d.D.bit g i)).1 = false := (mem_filter.1 hb).2
  rw [Circuit.unbit_bit] at hi
  obtain rfl : g = 2 := (by decide : ∀ g : Fin d.D.N, d.D.isInput g = false → g = 2) g hi
  have hxy : d.D.bit 0 i ≠ d.D.bit 1 i := ne_of_lt (d.D.bit_lt (by decide) _ _)
  refine ⟨(mem_filter.1 hb).2, Circuit.not_affine_of_and d.D.expand (d.D.bit 2 i) hxy
    (fun p q a => BitVec.ofBool (Circuit.andProbe (d.D.bit 0 i) (d.D.bit 1 i) p q a))
    (fun p q a => Circuit.bitZ_ofBool _) fun p q => ?_⟩
  rw [Circuit.expand_op_bit, Circuit.bitZ_ofBool, op_two, BitVec.getLsbD_and, getLsbD_ofBits, getLsbD_ofBits,
    getLsbD_ofBool_zero, getLsbD_ofBool_zero, Circuit.andProbe_x hxy, Circuit.andProbe_y hxy]

end Expand.AndPair

end Verity

namespace Verity.SecurityProofs

open Finset Verity.Expand

theorem ExpandSize_honest : Guarantees.ExpandSize_honest := ExpandSize AndPair.d 1 AndPair.P

theorem ExpandSize_nontrivial : Guarantees.ExpandSize_nontrivial := by
  have h := AndPair.d.width_unregistered_le_expand AndPair.P 0 fun _ _ => show 0 < 2 by decide
  have hpos : 0 < AndPair.d.D.width (AndPair.d.unregistered (AndPair.P.gates 0)) := by decide
  unfold Guarantees.ExpandSize_nontrivial
  omega

theorem ExpandAndSize_honest : Guarantees.ExpandAndSize_honest := ExpandAndSize AndPair.d 1 AndPair.P

theorem ExpandAndSize_nontrivial : Guarantees.ExpandAndSize_nontrivial := by
  have hand : AndPair.d.D.expand.size AndPair.d.D.expand.computed ≤
      AndPair.d.D.expand.andSize AndPair.d.D.expand.computed := by
    classical
    unfold Circuit.andSize Circuit.size
    exact card_le_card fun b hb => mem_filter.2 ⟨hb, AndPair.isAnd b hb⟩
  have hsize := (ExpandSize AndPair.d 1 AndPair.P).1
  have hpos : 0 < AndPair.d.D.width AndPair.d.D.computed := by decide
  unfold Guarantees.ExpandAndSize_nontrivial
  omega

end Verity.SecurityProofs
