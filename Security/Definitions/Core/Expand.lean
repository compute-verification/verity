import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Definitions.Core.Formalism

/-!
# The bit-level expansion: a multi-bit circuit as a Boolean one

A gate of `w g` bits is a bundle of `w g` one-bit gates (§3). The expansion `C.expand` has one gate per bit, gate `g`'s
bit `i` being `C.bit g i`, numbered gate by gate in `C`'s order and, within a gate, least significant bit first, so a
bundle read in order is `Circuit.bits`. Bit `i` of a computed gate `g` reads every bit of every gate `g` reads and is
bit `i` of `g`'s operation on the values they assemble (`ofBits`). An input bit is an input gate, a registered one when
its gate is (`Declaration.expand`), and a partition puts each bit in its gate's unit (`Partition.expand`).

`expandT X` is a transcript of `C` read bit by bit, and `collapse a` is a transcript of the expansion assembled gate by
gate. Gates of width 0 have no bits.
-/

namespace Verity

open Finset

/-- The value of `n` bits `f`, bit `i` of weight `2^i`. -/
def ofBits {n : ℕ} (f : Fin n → Bool) : BitVec n :=
  (BitVec.ofBoolListLE (List.ofFn f)).cast List.length_ofFn

namespace Circuit

variable (C : Circuit)

/-- Bit `i` of gate `g`, as a gate of the expansion. -/
def bit (g : Fin C.N) (i : Fin (C.w g)) : Fin (∑ g, C.w g) := finSigmaFinEquiv ⟨g, i⟩

/-- The gate and the bit that a gate of the expansion is. -/
def unbit (b : Fin (∑ g, C.w g)) : (g : Fin C.N) × Fin (C.w g) := finSigmaFinEquiv.symm b

theorem bit_unbit (b : Fin (∑ g, C.w g)) : C.bit (C.unbit b).1 (C.unbit b).2 = b :=
  finSigmaFinEquiv.apply_symm_apply b

theorem unbit_bit (g : Fin C.N) (i : Fin (C.w g)) : C.unbit (C.bit g i) = ⟨g, i⟩ :=
  finSigmaFinEquiv.symm_apply_apply _

private def widthUpTo (n : ℕ) : ℕ := ∑ j ∈ range n, if h : j < C.N then C.w ⟨j, h⟩ else 0

private theorem bit_val (g : Fin C.N) (i : Fin (C.w g)) : (C.bit g i : ℕ) = C.widthUpTo g + i := by
  rw [bit, finSigmaFinEquiv_apply, widthUpTo,
    ← Fin.sum_univ_eq_sum_range (fun j => if h : j < C.N then C.w ⟨j, h⟩ else 0)]
  congr 1
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [show (j : ℕ) < C.N from j.2.trans g.2, ↓reduceDIte]
  rfl

/-- The bits of an earlier gate come before the bits of a later one. -/
theorem bit_lt {a g : Fin C.N} (h : a < g) (i : Fin (C.w a)) (j : Fin (C.w g)) : C.bit a i < C.bit g j := by
  rw [Fin.lt_def, bit_val, bit_val]
  have hsucc : C.widthUpTo a + C.w a = C.widthUpTo (a + 1) := by
    rw [widthUpTo, widthUpTo, sum_range_succ]
    simp only [a.2, ↓reduceDIte]
  have hmono : C.widthUpTo (a + 1) ≤ C.widthUpTo g :=
    sum_le_sum_of_subset (range_subset_range.2 (Nat.succ_le_of_lt h))
  have := i.2
  omega

/-- **The bit-level expansion** of `C`: one gate per bit. -/
@[reducible] def expand : Circuit where
  N := ∑ g, C.w g
  w := fun _ => 1
  isInput b := C.isInput (C.unbit b).1
  reads b := if C.isInput (C.unbit b).1 then ∅ else univ.filter fun b' => (C.unbit b').1 ∈ C.reads (C.unbit b).1
  reads_lt b b' hb := by
    split_ifs at hb with hi
    · simp at hb
    · rw [← C.bit_unbit b, ← C.bit_unbit b']
      exact C.bit_lt (C.reads_lt _ _ (mem_filter.1 hb).2) _ _
  input_reads b h := by simp [h]
  op b v := BitVec.ofBool
    ((C.op (C.unbit b).1 fun a => ofBits fun i => (v (C.bit a i)).getLsbD 0).getLsbD (C.unbit b).2)
  op_congr b v v' h := by
    congr 2
    refine C.op_congr _ _ _ fun a ha => ?_
    have hi : C.isInput (C.unbit b).1 = false := by
      cases hi : C.isInput (C.unbit b).1
      · rfl
      · simp [C.input_reads _ hi] at ha
    congr 1
    funext i
    refine congrArg (·.getLsbD 0) (h (C.bit a i) ?_)
    simp only [hi, Bool.false_eq_true, ite_false, mem_filter, mem_univ, true_and, unbit_bit]
    exact ha
  outputs := univ.filter fun b => (C.unbit b).1 ∈ C.outputs

/-- `X` read bit by bit: a transcript of the expansion. -/
def expandT (X : C.Transcript) : C.expand.Transcript :=
  fun b => BitVec.ofBool ((X (C.unbit b).1).getLsbD (C.unbit b).2)

/-- A transcript of the expansion assembled gate by gate: a transcript of `C`. -/
def collapse (a : C.expand.Transcript) : C.Transcript :=
  fun g => ofBits fun i => (a (C.bit g i)).getLsbD 0

/-- **Positive fan-in in bits**: every computed gate of positive width reads a gate of positive width. The expansion
of such a circuit is Boolean. -/
def BitFaninPos : Prop :=
  ∀ g, C.isInput g = false → 0 < C.w g → ∃ a ∈ C.reads g, 0 < C.w a

end Circuit

/-- A declaration's expansion: an input bit is registered when its gate is. -/
@[reducible] def Declaration.expand (d : Declaration) : Declaration where
  D := d.D.expand
  θ := univ.filter fun b => (d.D.unbit b).1 ∈ d.θ
  θ_input _ hb := d.θ_input _ (mem_filter.1 hb).2

/-- A partition's expansion: each bit is in its gate's unit. -/
def Partition.expand {C : Circuit} {B : ℕ} (P : Partition C B) : Partition C.expand B where
  unit b := P.unit (C.unbit b).1
  input_iff _ := P.input_iff _

end Verity
