import Proofs.Flock.Soundness.Discharge.Zero.Delta

/-!
# A typed flat class's inputs that read the zero carry 0 (`hZero`, e2e)

`setupH_flatTableClass` (`ExecFlatCopies.lean`) is the `TableClass` of a typed flat class's statement that `Stmt.setupH`
accepted: each unit slot input's copy position is the column Δ copies into it (`flatSrcAt`, or the slot's forced-zero row
when that column lies outside the block), and its forced-zero rows are each unit slot's row past its useful rows.

- `flatCopyPos_mem_zeros`: at an input that reads the zero (`ZeroAt`), the copy position is a forced-zero row of the
  class. This is `hzr` of `UProg.zeroCols_of_classes` at that class, for every slot input that reads the zero.
- `flat_zero_block`: so in every block of a witness satisfying the statement, that input carries 0.
-/

open Flock

namespace FlockSoundness.Discharge.Zero

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
  (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
  {k : Typed.Class} {net : Net} (hr : Typed.read tags cf tables = .ok (k, .inl net))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

/-- **An input that reads the zero copies a forced-zero row of the flat class**: at unit slot `g`, input `i` of the
unit's composed rows, at slot offset `inCols[i]`, copies the slot's own forced-zero row (a zero leaf) or the first slot's
(a cut bit past 16). -/
theorem flatCopyPos_mem_zeros (g : Fin st.c.unitRange.count) (i : ℕ)
    (hz : ZeroAt st.c g ((k.done.getD k.unit default).inCols.getD i 0)) :
    (setupH_flatTableClass ht h hr regs).copyPos g i ∈ (setupH_flatTableClass ht h hr regs).zeros := by
  change flatCopyPos ht h hr regs g i ∈ Finset.univ.image (flatZeroPos ht h hr regs)
  unfold flatCopyPos
  split
  · rcases flatSrcAt_zero hz with he | he
    · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, Fin.ext he.symm⟩
    · exact Finset.mem_image.2 ⟨⟨0, Nat.lt_of_le_of_lt (Nat.zero_le _) g.isLt⟩, Finset.mem_univ _, Fin.ext he.symm⟩
  · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, rfl⟩

/-- **An input that reads the zero carries 0**: at unit slot `g` of the flat class, an input column of the unit's
composed rows whose slot offset reads the zero (`ZeroAt`) carries 0 in every block of a witness satisfying the
statement. -/
theorem flat_zero_block (g : Fin st.c.unitRange.count) (c : Fin (setupH_flatTableClass ht h hr regs).rows.w)
    (hc : c.val < (setupH_flatTableClass ht h hr regs).rows.nIn)
    (hz : ZeroAt st.c g ((k.done.getD k.unit default).inCols.getD c.val 0))
    {z : Model.Witness ((setupH_flatLayout ht h hr).2.2.model regs).m}
    (hsat : ((setupH_flatLayout ht h hr).2.2.model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).nbl)) :
    ((setupH_flatLayout ht h hr).2.2.model regs).block z o ((setupH_flatTableClass ht h hr regs).col g c) = 0 := by
  rw [(setupH_flatTableClass ht h hr regs).copy g c hc z hsat o]
  exact (setupH_flatTableClass ht h hr regs).zero _ (flatCopyPos_mem_zeros ht h hr regs g c.val hz) z hsat o

end FlockSoundness.Discharge.Zero
