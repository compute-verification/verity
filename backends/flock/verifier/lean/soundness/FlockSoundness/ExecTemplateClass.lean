import FlockSoundness.ExecTemplateSetup
import FlockSoundness.Types.ProgramPlaces

/-!
# A typed template's table class, from `Stmt.setupH` (1e for templates)

`TableClass` (`Types/ProgramPlaces.lean`) is what a table's accepted statement gives the end-to-end theorem's places:
- the verifier's derivation of its unit class;
- the class's composed rows, placed at each slot;
- each slot input's copy position;
- the forced-zero rows.

`setupH_templateTableClass` builds it, with no hypothesis, for a typed template (`tags.typed`,
`Typed.read … = .ok (k, .inr t)`). It uses what `ExecTemplateSetup.lean` proves of the accepted statement:
- **The slots are the VUs** (`st.c.g`). Slot `g` holds the unit's composed rows at `Typed.col st.c tb g`
  (`TemplateLayout.placement`).
- **Input `w` of VU `g` copies its message bit** `Typed.copySrc st.c g w` (`setupH_inputCopy`). The composed rows' first
  columns are the unit's input columns in order (`order_sound`, `order_getD_inCols`), so input column `w` of the class's
  rows is at the position that copy row reads.
- **There are no forced-zero rows.** Every input copies a message bit inside the block (`setupH_copySrc_lt`).
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

section tableClass

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = true)
  (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {t : Typed.Template}
  (hr : Typed.read tags cf tables = .ok (k, .inr t)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

/-- **Input `w` of VU `g`'s copy position**: its message bit, `Typed.copySrc st.c g w`, for an input of the unit (the pin
past them, which no table class reads). -/
def templateCopyPos (g : Fin st.c.g) (w : ℕ) : Fin (2 ^ ((setupH_templateLayout ht h hr).2.2.model regs).kLog) :=
  if hw : w < (k.done.getD k.unit default).inCols.size then ⟨Typed.copySrc st.c g w, setupH_copySrc_lt ht h hr g.isLt hw⟩
  else ((setupH_templateLayout ht h hr).2.2.model regs).pin

/-- **A typed template's table class**, from the statement `Stmt.setupH` accepted: its unit class `k`, the VUs as slots,
each at `Typed.col st.c tb g`, each input copying its message bit, and no forced-zero rows. -/
def setupH_templateTableClass : TableClass ((setupH_templateLayout ht h hr).2.2.model regs) where
  types := Typed.typeIn k.types
  ls := k.ls
  info := Typed.infoIn k.held
  words := Typed.wordsIn k.held
  unit := k.unit
  done := k.done
  hd := k.hd
  slots := st.c.g
  col g := composePlace _ _ (ofBlock_ordered k.hd) (Typed.col st.c (HmRow.blockOf t) g)
    (((setupH_templateLayout ht h hr).2.2.blockFacts k.hd (unit_lt k.hd) (unitShape_of_class k) regs g.isLt
      le_rfl).realizes k.hd le_rfl).lt
  placed g := (setupH_templateLayout ht h hr).2.2.placement k.hd (unit_lt k.hd) (unitShape_of_class k) regs g.isLt
  copyPos := templateCopyPos ht h hr regs
  copy g c hc z hz o := by
    have hw : c.val < (k.done.getD k.unit default).inCols.size := hc
    obtain ⟨-, -, htake, -, -⟩ := order_sound k.hd 0
    have hlen : (k.done.getD k.unit default).inCols.size ≤ (order k.done (k.done.getD k.unit default)).length := by
      have := congrArg List.length htake
      simp only [List.length_take, Array.length_toList] at this
      omega
    have hcol : composePlace _ _ (ofBlock_ordered k.hd) (Typed.col st.c (HmRow.blockOf t) g)
        (((setupH_templateLayout ht h hr).2.2.blockFacts k.hd (unit_lt k.hd) (unitShape_of_class k) regs g.isLt
          le_rfl).realizes k.hd le_rfl).lt c =
        ⟨Typed.col st.c (HmRow.blockOf t) g (k.done.getD k.unit default).inCols[c.val],
          (setupH_templateLayout ht h hr).2.2.col_lt k.hd g.isLt
            (TemplateLayout.inCols_row k.hd (unit_lt k.hd) (unitShape_of_class k) (Array.getElem_mem hw)).1⟩ := by
      apply Fin.ext
      rw [composePlace_val]
      have hlt : c.val < (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default)).order.length :=
        lt_of_lt_of_le hw hlen
      rw [if_pos hlt]
      show Typed.col st.c (HmRow.blockOf t) g ((order k.done (k.done.getD k.unit default)).getD c.val 0) = _
      rw [order_getD_inCols htake hw]
      have hget : (k.done.getD k.unit default).inCols.getD c.val 0 = (k.done.getD k.unit default).inCols[c.val] := by
        generalize k.done.getD k.unit default = u at hw ⊢
        intro hw
        simp [Array.getD, hw]
      rw [hget]
    have hpos : templateCopyPos ht h hr regs g c.val = ⟨Typed.copySrc st.c g c.val, setupH_copySrc_lt ht h hr g.isLt hw⟩ := by
      unfold templateCopyPos
      exact dite_eq_left_of_eq_true (eq_true hw)
    show ((setupH_templateLayout ht h hr).2.2.model regs).block z o (composePlace _ _ (ofBlock_ordered k.hd)
        (Typed.col st.c (HmRow.blockOf t) g)
        (((setupH_templateLayout ht h hr).2.2.blockFacts k.hd (unit_lt k.hd) (unitShape_of_class k) regs g.isLt
          le_rfl).realizes k.hd le_rfl).lt c) =
      ((setupH_templateLayout ht h hr).2.2.model regs).block z o (templateCopyPos ht h hr regs g c.val)
    rw [hcol, hpos]
    exact (setupH_inputCopy ht h hr regs g.isLt hw).block_eq hz o
  zeros := ∅
  zero q hq := absurd hq (Finset.notMem_empty q)

end tableClass

end FlockSoundness
