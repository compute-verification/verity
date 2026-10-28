import Definitions.Nci.BitView
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# A declaration bit by bit, as a declaration

`Circuit.ofDecl d` is `d`'s wiring bit by bit (`Definitions.Nci.BitView`). `bitDecl d` is the same bits as a declaration
(`Verity.Registered`), so that the covering bound's vocabulary (`Nci.Covering`), stated over a declaration's gates,
reads `d` bit by bit: each of its gates is one bit wide, and its counts (`|S|`, `|in(S) \ θ|`, a cut's size) count bits,
the unit of the NCI policy's novel reads and of the covering bound's costs (an activation's width: 16 at bf16, 8 at FP8,
4 at FP4).

* Its gates are the bits `(g, i)` of `d`'s gates, numbered as in the bit view, and each reads what the bit view says
  (`BitView.bitPred`): a real bit reads every real bit of every gate its gate reads, and padding reads nothing.
* Its input gates are the bits that read nothing: the real bits of `d`'s input gates and of its constant gates (computed
  gates that read nothing), and the padding. Its baseline is the bit view's free bits (`BitView.free`): all of those but
  the real bits of `d`'s novel input gates. A constant's value is fixed by the circuit's text, so its bits are
  registered here, as the bit view frees them.
* Bit `(g, i)` computes bit `i` of what `g` computes from the values of the gates it reads, each assembled from its real
  bits (`BitView.word`, `BitView.bitOp`); a padding bit computes zero. Its output gates are the real bits of `d`'s.

Its wiring is the bit view's: the bit view of its gates (`Proofs.Nci.Cover.Gates`' `ofGates (bitDecl d)`) is
`Circuit.ofDecl d` by definition.
-/

namespace Nci.Cover

open Finset

namespace BitView

variable (C : Verity.Circuit)

/-- The value of gate `a`, assembled from the one-bit values `v` of its real bits, bit `j` the low bit of
`v (a, j)`. -/
def word (v : Fin (C.N * W C) → BitVec 1) (a : Fin C.N) : BitVec (C.w a) :=
  BitVec.ofNat (C.w a) (∑ j : Fin (C.w a),
    (v (enc C (a, Fin.castLE (le_sup (f := C.w) (mem_univ a)) j))).toNat * 2 ^ (j : ℕ))

/-- Bit `x` of the gates' values, from the one-bit values `v`: for a real bit, bit `x.2` of what gate `x.1` computes from
the words its reads assemble; zero for padding. -/
def bitOp (x : Bit C) (v : Fin (C.N * W C) → BitVec 1) : BitVec 1 :=
  if (x.2 : ℕ) < C.w x.1 then BitVec.ofBool ((C.op x.1 (word C v)).getLsbD x.2) else 0

end BitView

open BitView in
/-- **A declaration bit by bit, as a declaration**: the bits of `d`'s gates, one bit wide each, wired as the bit view
(`Circuit.ofDecl d`), each bit that reads nothing an input gate, the free bits registered, and each real bit computing
its bit of its gate's value. -/
def bitDecl (d : Verity.Registered) : Verity.Registered :=
  .ofCircuit
    { N := (Circuit.ofDecl d).N
      w := fun _ => 1
      isInput := fun y => decide ((Circuit.ofDecl d).pred y = ∅)
      reads := (Circuit.ofDecl d).pred
      reads_lt := (Circuit.ofDecl d).pred_lt
      input_reads := fun _ h => of_decide_eq_true h
      op := fun y v => bitOp d.D ((enc d.D).symm y) v
      op_congr := by
        intro y v v' h
        obtain ⟨x, rfl⟩ := (enc d.D).surjective y
        have hx : ∀ x' ∈ bitPred d.D x, v (enc d.D x') = v' (enc d.D x') := fun x' hx' =>
          h _ (by
            change _ ∈ (bitPred d.D ((enc d.D).symm (enc d.D x))).map (enc d.D).toEmbedding
            rw [Equiv.symm_apply_apply]
            exact mem_map_of_mem _ hx')
        change bitOp d.D ((enc d.D).symm (enc d.D x)) v = bitOp d.D ((enc d.D).symm (enc d.D x)) v'
        rw [Equiv.symm_apply_apply]
        unfold bitOp
        split_ifs with hr
        · congr 2
          refine d.D.op_congr x.1 _ _ fun a ha => ?_
          unfold word
          congr 1
          refine sum_congr rfl fun j _ => ?_
          rw [hx]
          simp only [bitPred, hr, ↓reduceIte, bits, mem_filter, mem_product, mem_univ, and_true]
          exact ⟨ha, j.isLt⟩
        · rfl
      outputs := wordBits d d.D.outputs }
    (Circuit.ofDecl d).θ fun y hy => decide_eq_true ((Circuit.ofDecl d).θ_input y hy)

end Nci.Cover
