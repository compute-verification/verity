import Definitions.Core.Registered
import Definitions.Nci.Cover
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Data.Fintype.Prod
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# A declaration's circuit, bit by bit

The cover theorem is stated over a circuit of one-bit gates (`Nci.Cover.Circuit`); a declaration (`Verity.Registered`)
has gates of `w g` bits each. `Circuit.ofDecl d` is `d`'s circuit bit by bit. It keeps the wiring and nothing of what the
gates compute:

* bit `i` of gate `g` (`BitView.Bit`, a pair `(g, i)` with `i < W`, the widest gate's width) is real when `i < w g`; the
  others pad each gate to `W` bits;
* a real bit reads every real bit of every gate its gate reads (`BitView.bitPred`), so it may depend on all of the bits
  its gate's value does; a padding bit reads nothing;
* the registered parameters are the free bits (`BitView.free`): those that read nothing and are not real bits of a novel
  input gate. They are θ's bits, a constant gate's bits and the padding. A real bit of an input gate outside θ is novel
  information, which a block that uses it is handed.

Bit `(g, i)` is numbered `i + W g` (`finProdFinEquiv`), so the bits a bit reads, of gates of smaller index, have smaller
numbers. `wordBits d S` is the set of the real bits of the gates of `S`, a set of whole words.
-/

namespace Nci.Cover

open Finset

namespace BitView

variable (C : Verity.Circuit)

/-- The widest gate's width: every gate's bits fit in `W` slots. -/
def W : ℕ := univ.sup C.w

/-- Bit `i` of gate `g`, real when `i < w g`. -/
abbrev Bit : Type := Fin C.N × Fin (W C)

/-- The real bits of the gates of `S`. -/
def bits (S : Finset (Fin C.N)) : Finset (Bit C) :=
  (S ×ˢ univ).filter fun x => (x.2 : ℕ) < C.w x.1

/-- The bits that bit `x` reads: every real bit of every gate its gate reads when `x` is real, none when it pads. -/
def bitPred (x : Bit C) : Finset (Bit C) :=
  if (x.2 : ℕ) < C.w x.1 then bits C (C.reads x.1) else ∅

/-- The numbering of bits: `(g, i)` is `i + W g`. -/
def enc : Bit C ≃ Fin (C.N * W C) := finProdFinEquiv

/-- The free bits for the baseline `θ`: those that read nothing and are not real bits of an input gate outside `θ`. -/
def free (θ : Finset (Fin C.N)) : Finset (Bit C) :=
  univ.filter fun x => bitPred C x = ∅ ∧ (x.1 ∈ θ ∨ C.isInput x.1 = false ∨ C.w x.1 ≤ (x.2 : ℕ))

end BitView

open BitView in
/-- **A declaration's circuit, bit by bit**: the bits of `d`'s gates, each reading the bits of the gates its gate reads,
with the free bits registered. -/
def Circuit.ofDecl (d : Verity.Registered) : Circuit where
  N := d.D.N * W d.D
  pred y := (bitPred d.D ((enc d.D).symm y)).map (enc d.D).toEmbedding
  pred_lt y y' hy := by
    obtain ⟨x', hx', rfl⟩ := mem_map.1 hy
    obtain ⟨x, rfl⟩ := (enc d.D).surjective y
    rw [Equiv.symm_apply_apply] at hx'
    unfold bitPred at hx'
    split_ifs at hx' with hr
    · have hlt : x'.1 < x.1 := d.D.reads_lt _ _ (mem_product.1 (mem_filter.1 hx').1).1
      have h1 : (x'.2 : ℕ) < W d.D := x'.2.isLt
      have h2 : (x'.1 : ℕ) + 1 ≤ x.1 := hlt
      show x'.2.val + W d.D * x'.1.val < x.2.val + W d.D * x.1.val
      calc x'.2.val + W d.D * x'.1.val < W d.D + W d.D * x'.1.val := Nat.add_lt_add_right h1 _
        _ = W d.D * (x'.1.val + 1) := by rw [Nat.mul_succ, Nat.add_comm]
        _ ≤ W d.D * x.1.val := Nat.mul_le_mul_left _ h2
        _ ≤ x.2.val + W d.D * x.1.val := Nat.le_add_left _ _
    · exact absurd hx' (notMem_empty _)
  θ := (free d.D d.θ).map (enc d.D).toEmbedding
  θ_input y hy := by
    obtain ⟨x, hx, rfl⟩ := mem_map.1 hy
    simp [(mem_filter.1 hx).2.1]

open BitView in
/-- **The bits of a set `S` of `d`'s gates** in its bit view: the real bits of the gates of `S`. -/
def wordBits (d : Verity.Registered) (S : Finset (Fin d.D.N)) : Finset (Fin (Circuit.ofDecl d).N) :=
  (bits d.D S).map (enc d.D).toEmbedding

end Nci.Cover
