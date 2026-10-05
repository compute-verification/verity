/-
Δ's copies at the rows' inputs, in a satisfying witness. Each `cv`, chain and `pad` input bit of a row's slots is a self
row of its net (`HmNets.layout`), so its block row is `[x]` plus Δ's pairs there. Δ (`delta_rows`) writes there exactly
one copy `(x, x), (x, src)`: the rows' slots are distinct (`at_inj`, `slot_inj`) and the `sha512x3` and `hm96` nets are
two (`sha_ne_hm`). So the block row at `x` is `[src]`, and the block holds the copy (`inBit_copy`): the first `cv` is the
row prefix's midstate, each later `cv` the previous output, the `hm96` slot's `cv` the last output and its `pad` the
padding block (`row_copies_of_sat`), the constant bits read from the pin and the forced-zero row.
-/
import FlockSoundness.Discharge.Hm.Copies
import FlockSoundness.Protocol.Discharge.Hm.Copy

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

/-! ## The rows' nets' inputs -/

theorem wireCopies_eq {c : Circuit} (hw : c.wires = HmRow.rowWires c.ports c.shaNet c.hmNet)
    (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks)
    (hs : ∀ q < c.ports.size, c.ports[q]!.segs = none) :
    wireCopies c = (List.range c.g).flatMap fun g =>
      (List.range (HmRow.rowSlots c.ports).2.size).flatMap fun p => wireF c g p := by
  unfold wireCopies wireF
  rw [hw, rowWires_eq _ _ _ hpos hs]
  simp only [List.flatMap_assoc]

theorem RowRanges.ofT {st : Stmt} {tb : TypedBlock} {words : String → Option (Array ℕ)} {done : Array D} {u : D}
    {tags : Tags} {t : Typed.Template} (T : TemplateLayout st tb words done u) (C : CheckFactsT st.c tb)
    (P : ParseFactsT tags t st.c) : RowRanges st.c :=
  ⟨T.disj, T.bound, T.slotLog, T.count, T.one_range, T.fit, T.shaNet.1, T.hmNet.1, C.perVu_sha, C.perVu_hm, P.rowNets,
    T.kLog⟩

namespace RowRanges

variable {c : Circuit} (R : RowRanges c)
include R

end RowRanges

section Loc

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

/-- **An input bit of a row net's slot holds its one copy**, in a typed template's statement. At a self row `x` of a
`sha512x3` or `hm96` slot below the net's input rows, where Δ's copies (`rowCopies`, `wireCopies`) write exactly one
`(x, src)`, every block of a satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copy {tags : Tags} {t : Typed.Template} (hd : deriveChecked types ls info words unit = .ok done)
    (hul : unit < ls.size) (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    (P : ParseFactsT tags t st.c) (C : CheckFactsT st.c tb)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) (hdb : st.db = (HmRow.delta st.c st.pin).2)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopies st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) = [(x, src)]) :
    blockBit (T.model regs) z o x = blockBit (T.model regs) z o src := by
  have R := RowRanges.ofT T C P
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, -⟩ := R.range hnn
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  have hy : ∀ g' < st.c.g, ∀ u, InUnit done 𝒖 u → NotConst 𝒖 u → Typed.col st.c tb g' u ≠ x := by
    intro g' hg' u hu _ hux
    rcases T.at_inUnit hd hg' hu with ⟨-, hat'⟩ | ⟨i, hi, hiu, -, -, -, hat'⟩
    · obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range T.unitNet).1 (T.range T.unitNet).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact T.shaNet.2 h1.symm
      · exact T.hmNet.2 h1.symm
    · obtain ⟨-, hpn, -, hps, hph, -⟩ := T.part i hi hiu
      obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range hpn).1 (T.range hpn).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact hps h1
      · exact hph h1
  have hC : ∀ j, (constPairs st.c st.pin).count (x, j) = 0 := fun j => by
    rw [constPairs_count T.disj T.slotLog T.fit hm hk hat j, ite_eq_right (fun h => by have := h.1; omega)]
  have hF := flatPairs_count hC (fun j => by rw [leafPairs_nil P.leavesIn P.upv]; rfl)
    (fun j => by rw [cutPairs_nil P.leafCuts]; rfl) hf
  have hδ : ∀ j, deltaEntry st.da x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db x j = (((if j = x then 1 else 0) + (if j = src then 1 else 0) : ℕ) : ZMod 2) := by
    intro j
    obtain ⟨h1, h2⟩ := delta_rows st.c st.pin T.typed hv1 P.portWords P.v1Segs
    rw [deltaEntry_eq_count, deltaEntry_eq_count, hda, hdb, h1, h2, List.count_append, List.count_append, hF,
      T.tail_zero hd hul U hy true, T.tail_zero hd hul U hy false, Nat.add_zero]
    exact ⟨rfl, rfl⟩
  have hx : x < 2 ^ (T.model regs).kLog := hat.lt T.bound hm
  exact block_of_rows hz o hx fun j => inBit_rows R hn hat hl hself hx hδ j

end Loc

end FlockSoundness.Discharge.Hm
