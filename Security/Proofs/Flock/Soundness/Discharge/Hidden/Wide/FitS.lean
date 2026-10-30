import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegFit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegTgt

/-!
# The rows the hidden-output headlines cover: fit rows and `sha512/row-seg/v1` rows, each under its own prefix

`RowsFit` covers rows that fit (`FitRow`), all of one kind (`sha512/row/v1` or `/row/v2`), so one prefix (`rowPre`) at
every row. `Any.RowsFitS` covers each row unsegmented, its padding perhaps starting in its last compression's block
(`Any.PadRow`), or a `sha512/row-seg/v1` row as `HmRow.segShape` cuts it (`Any.SegRow`), of any mix, the output rows
fitting (`Any.OutRowsS`). A row's `hm96` slot hashes `Any.rowMsg` of its message bytes: its own prefix (`rowPre` at its
own `v2`) and the bytes at an unsegmented row, `RowSeg.enc` of them at a row-seg row
(`Any.row_bc_tgtS`, from `Any.row_bc_tgt` and `Any.seg_bc_tgt`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag
open SegShape

namespace Any

/-- **A `sha512/row-seg/v1` row the hidden-output statement's proof covers**: segmented as `HmRow.segShape` cuts its
bits, `16 · words` bits, its chunks its segments' compressions and then its outer message's (`outerComps`), each
segment's bytes ending in its last compression's block (`SegFits`). -/
def SegRow (p : Port) : Prop :=
  p.segs = some (HmRow.segShape p.bits).1 ∧ p.bits = 16 * p.words ∧
    p.chunks = firstOf (HmRow.segShape p.bits).1 (Binding.RowSeg.count p.bits) + outerComps p.bits ∧ SegFits p.bits

/-- **The rows the hidden-output headlines cover**: one unsegmented output row per output port after the
input rows (`OutRowsS`), and every row an unsegmented row (`PadRow`) or a `sha512/row-seg/v1` row (`SegRow`). -/
def RowsFitS (c : Circuit) : Prop := OutRowsS c ∧ ∀ p ∈ c.ports.toList, PadRow p ∨ SegRow p

/-- **The bytes a row's `hm96` slot hashes**, `bytes` its message bytes: `RowSeg.enc` of them at a segmented row, else
its own prefix (`rowPre`) and them. -/
def rowMsg (p : Port) (bytes : List UInt8) : List UInt8 :=
  if p.segs.isSome then Binding.RowSeg.enc p.bits bytes else (rowPre p.v2 p.words).data.toList ++ bytes

namespace RowsFitS

variable {c : Circuit} (h : RowsFitS c)
include h

theorem kind {p : ℕ} (hp : p < c.ports.size) : PadRow c.ports[p]! ∨ SegRow c.ports[p]! :=
  h.2 _ (by rw [getElem!_pos c.ports p hp]; exact Array.getElem_mem_toList hp)

theorem bits16 : ∀ p ∈ c.ports.toList, p.bits = 16 * p.words := fun p hp => by
  rcases h.2 p hp with h1 | h1
  · exact h1.2.1
  · exact h1.2.1

end RowsFitS

/-- Rows that fit, of one kind, give `RowsFitS`. -/
theorem RowsFit.rowsFitS {c : Circuit} (h : RowsFit c) : RowsFitS c :=
  ⟨Any.OutRowsF.outRowsS h.1, fun p hp => Or.inl (padRow_of_fit (h.1.1 p hp))⟩

/-- **At an unsegmented row, `rowMsg` is its own prefix and the bytes.** -/
theorem rowMsg_fit {p : Port} (h : PadRow p) (bytes : List UInt8) :
    rowMsg p bytes = (HmRow.prefixOf p).data.toList ++ bytes := by
  obtain ⟨hs, hb, -, -⟩ := h
  unfold rowMsg HmRow.prefixOf rowPre
  rw [hs, hb]
  rfl

/-- **At a row-seg row, `rowMsg` is `RowSeg.enc`.** -/
theorem rowMsg_seg {p : Port} (h : SegRow p) (bytes : List UInt8) :
    rowMsg p bytes = Binding.RowSeg.enc p.bits bytes := by
  unfold rowMsg
  rw [h.1]
  rfl

section Bc

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **A row's `b ‖ c` in a satisfying witness of the untyped statement**, unsegmented or row-seg: the commit string of
`rowMsg` of its `2 · words` message bytes, under the 192 bytes at `saltCol` (`Any.row_bc_tgt`, `Any.seg_bc_tgt`). -/
theorem row_bc_tgtS (hda : st.da = (HmRow.delta st.c st.pin).1) (hupv : 0 < st.c.upv)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) (UnitIn st.c) z) (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hk : PadRow st.c.ports[p]! ∨ SegRow st.c.ports[p]!) :
    bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) = Binding.commitString
      (rowMsg st.c.ports[p]! (bytesOf (blockBit (L.model regs) z o) (2 * st.c.ports[p]!.words) (Typed.msgCol st.c g p)))
      (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g p)) := by
  rcases hk with hf | hs
  · rw [rowMsg_fit hf]
    exact Any.row_bc_tgt L F hda hupv hz o hg hp hf.2.1 hf.2.2.1 hf.2.2.2 hf.1
  · have hl : (st.c.ports[p]!.bits + 7) / 8 = 2 * st.c.ports[p]!.words := by rw [hs.2.1]; omega
    have := Any.seg_bc_tgt L F hda hupv hz o hg hp (l := st.c.ports[p]!.bits) hs.1 rfl hs.2.2.1 hs.2.2.2
    rw [hl] at this
    rw [rowMsg_seg hs]
    exact this

end Bc

end Any

end FlockSoundness.Discharge.Hidden
