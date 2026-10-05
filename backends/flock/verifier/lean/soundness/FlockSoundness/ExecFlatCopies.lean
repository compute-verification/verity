import FlockSoundness.ExecFlatClass
import FlockSoundness.Types.ProgramPlaces
import FlockSoundness.Protocol.ExecFlatCopies

/-!
# A typed flat class's input copies and forced-zero rows (1e for flat classes)

A flat statement's Δ (`HmRow.delta`, no typed template) is its slot constants' pairs, the rows' constant bits, then each
unit copy's leaf bits, cut bits and wired bits (`delta_flat`). `HmRow.check` wires each of a unit's input ports exactly
once (`CheckFacts.wired`, leaves by `leaves_in`), and #434's `Typed.flatNet` makes the unit's inputs its net's input port
bits in order, so the ports' bits are distinct columns. At each of a unit copy's input port bits, Δ then has one copy
pair, over the net's self row there:
- `Layout.delta_port`: the count of Δ's pairs at a port bit is its own and its source's (`flatSrc`);
- `Layout.placed_port`: the block rows there read the source, so they copy it, or are zero when it lies outside the
  block;
- `Layout.placed_zero`: each unit slot's row past its useful rows is a forced-zero row;
- `setupH_flatTableClass`: the flat class's `TableClass`, from the statement `Stmt.setupH` accepted.
-/

open Flock FlockLevel3

namespace FlockSoundness

open Compose Flock.DeriveAll Types.Dag

/-! ## Δ, pair by pair -/

namespace Flat

end Flat


/-! ## A net's input port bits -/

/-! ## Positions in a unit's range -/

namespace Layout

variable {st : Stmt} (L : Layout st)
include L

end Layout

/-! ## A flat unit's net -/

/-- A flat unit's net holds its rows as they are: its row `c` is the unit's, empty past the unit's rows. -/
theorem netOfD_row {d : D} {name : String} {n : Net} (h : Typed.netOfD d name = .ok n) (hlog : n.unitLog ≤ 32) (c : ℕ) :
    n.a.row c = (if c < d.size then (d.rows.getD c ([], [])).1.toArray else #[]) ∧
      n.b.row c = if c < d.size then (d.rows.getD c ([], [])).2.toArray else #[] := by
  obtain ⟨-, ha, hb, -⟩ := Typed.netOfD_ok h
  have hw : 2 ^ n.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  rw [ofRows_row ha hw, ofRows_row hb hw]
  unfold Typed.rowsA Typed.rowsB
  by_cases hc : c < d.size
  · simp [Array.getD_eq_getD_getElem?, hc]
  · simp [Array.getD_eq_getD_getElem?, hc]

/-! ## The accepted flat-class statement -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **What a flat class's accepted statement holds beside its `Layout`**: its circuit's parse and check, Δ pair by pair,
and a unit that is neither of the rows' circuits (it is `parseTyped`'s first net, named `unit`). -/
theorem setupH_flatFacts (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) :
    ParseFacts tags st.c ∧ CheckFacts st.c ∧ FlatDelta st.c st.pin st.da ∧ st.c.unitNet ≠ st.c.shaNet ∧
      st.c.unitNet ≠ st.c.hmNet := by
  obtain ⟨hp, -, hda, -, -, -⟩ := setupH_spec_typed ht h
  obtain ⟨k', r, hr', hcase⟩ := parseTyped_spec hp
  rw [hr] at hr'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hr')
  rcases hcase with ⟨t, c', hrt, -, -⟩ | ⟨net', c', hrn, hpc, hc⟩
  · cases hrt
  · cases hrn
    obtain ⟨name, hnet, -⟩ := Typed.read_flat hr
    obtain ⟨P, ⟨more, hmore⟩, hunit, hsn, hhn⟩ := parse_facts_pre (fun x hx => by
      simp at hx
      subst hx
      exact netOfD_built hnet) hpc
    have P' : ParseFacts tags st.c := by
      rw [hc]
      exact ⟨P.nets, P.portWords, P.v1Segs, P.rowWires, P.unitNet, P.shaNet, P.hmNet, P.kinds, P.check,
        (by rw [← P.rowNets]; exact HmNets.check_congr rfl rfl rfl), P.typed⟩
    obtain ⟨ls, hcheck⟩ := P'.check
    have h0 : c'.nets[0]? = some ("unit", net) := by
      rw [← Array.getElem?_toList, hmore]
      simp
    have hu0 : c'.unitNet = 0 := by
      obtain ⟨hlt, -, hmin⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
      by_contra hne
      have h1 := hmin 0 (Nat.pos_of_ne_zero hne)
      rw [Array.getElem?_eq_getElem (by omega)] at h0
      simp [Option.some.inj h0] at h1
    have hname0 : (c'.nets.getD 0 default).1 = "unit" := by rw [getD_of_getElem? h0]
    refine ⟨P', check_facts P'.typed hcheck, by rw [hda]; exact delta_flat st.c st.pin P'.typed, ?_, ?_⟩
    · rw [hc]
      show c'.unitNet ≠ c'.shaNet
      intro he
      rw [← he, hu0, hname0] at hsn
      exact absurd hsn (by unfold HmRow.SHA_NET; decide)
    · rw [hc]
      show c'.unitNet ≠ c'.hmNet
      intro he
      rw [← he, hu0, hname0] at hhn
      exact absurd hhn (by unfold HmRow.HM_NET; decide)

/-- **A flat class's input copies, at each unit slot**: input `w`'s block rows read the column Δ copies into it
(`flatSrcAt`). The unit's inputs are its net's input port bits in order (`Typed.flatNet`, #434). -/
theorem setupH_flatInput (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) {s₀ w : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hw : w < (k.done.getD k.unit default).inCols.size) (r j : Fin (2 ^ st.c.kLog))
    (hr' : r.val = st.c.unitRange.slot s₀ + (k.done.getD k.unit default).inCols[w]) :
    placedA st r j = (if j.val = flatSrcAt st.c s₀ (k.done.getD k.unit default).inCols[w] then 1 else 0) ∧
      placedB st r j = if j.val = flatSrcAt st.c s₀ (k.done.getD k.unit default).inCols[w] then 1 else 0 := by
  obtain ⟨-, hunit, L⟩ := setupH_flatLayout ht h hr
  obtain ⟨P, C, hΔ, hsha, hhm⟩ := setupH_flatFacts ht h hr
  obtain ⟨-, name, hnet, -, hin⟩ := read_flat' hr
  replace hin : (k.done.getD k.unit default).inCols.toList = portBits net.inGroups := hin
  have hlog : net.unitLog ≤ 32 := hunit ▸ L.unitLog_le hs₀
  obtain ⟨hnd0, -, htake, -, -⟩ := order_sound k.hd 0
  have hndin : (k.done.getD k.unit default).inCols.toList.Nodup := by
    rw [← htake]; exact hnd0.sublist (List.take_sublist _ _)
  generalize hu : k.done.getD k.unit default = u at hin hw hr' hndin htake hnd0 ⊢
  have hnd : (portBits st.c.unit.inGroups).Nodup := by rw [hunit, ← hin]; exact hndin
  -- input `w` is a port bit
  have hmem : u.inCols[w] ∈ portBits net.inGroups := by rw [← hin]; exact Array.getElem_mem_toList hw
  obtain ⟨q, hq, hqr⟩ := List.mem_flatMap.1 hmem
  obtain ⟨p, hp, rfl⟩ := List.getElem_of_mem hq
  rw [List.mem_range'_1] at hqr
  have hp' : p < (portCols st.c.unit.inGroups).size := by rw [hunit]; simpa using hp
  have hq' : (portCols st.c.unit.inGroups)[p]! = (portCols net.inGroups).toList[p] := by
    rw [hunit, getElem!_pos _ _ (by simpa using hp), Array.getElem_toList]
  obtain ⟨t, ht', hoff⟩ : ∃ t, t < (portCols st.c.unit.inGroups)[p]!.2 ∧
      u.inCols[w] = (portCols st.c.unit.inGroups)[p]!.1 + t :=
    ⟨u.inCols[w] - (portCols st.c.unit.inGroups)[p]!.1, by rw [hq']; omega, by rw [hq']; omega⟩
  have hsrc : flatSrcAt st.c s₀ u.inCols[w] = Flat.flatSrc st.c s₀ p t := by
    unfold flatSrcAt
    rw [hoff, portOf_eq hnd hp' ht']
    congr 1
    omega
  rw [hsrc]
  -- the net's self row there
  have hcr : u.rows[u.inCols[w]]? = some ([u.inCols[w]], [u.inCols[w]]) := by
    have := (unit_inputs k.hd u.inCols[w] (by rw [hu]; exact Array.getElem_mem hw)).1
    rwa [hu] at this
  have hcs : u.inCols[w] < u.size := by
    have hmo : u.inCols[w] ∈ order k.done u := by
      rw [← List.take_append_drop u.inCols.size (order k.done u), htake]
      exact List.mem_append_left _ (Array.getElem_mem_toList hw)
    obtain ⟨hflat, -⟩ := read_flat' hr
    rw [hu] at hflat
    simp only [Typed.isFlat, Bool.and_eq_true] at hflat
    rcases (hu ▸ order_cols k.hd) u.inCols[w] hmo with h1 | ⟨pp, hpp, -⟩
    · exact h1
    · rw [Array.isEmpty_iff.1 hflat.1.1] at hpp; simp at hpp
  rw [hu] at hnet
  obtain ⟨hra, hrb⟩ := netOfD_row hnet hlog u.inCols[w]
  have hgd : u.rows.getD u.inCols[w] ([], []) = ([u.inCols[w]], [u.inCols[w]]) := getD_of_getElem? hcr
  have hself : ∀ j', (([u.inCols[w]].toArray : Array ℕ).toList.count j' : ZMod 2) = if j' = u.inCols[w] then 1 else 0 := by
    intro j'
    by_cases hj : j' = u.inCols[w]
    · subst hj; simp
    · simp [hj, Ne.symm hj]
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  refine L.placed_port P C hΔ hsha hhm hnd hs₀ hp' ht' (fun j' => ?_) (fun j' => ?_) r j (by rw [hr', hoff])
  · rw [← hoff, sparseEntry_built_a L.unit_built, hunit, hra, ite_eq_left hcs, hgd]
    exact hself j'
  · rw [← hoff, sparseEntry_built_b L.unit_built, hlk, hunit, hrb, ite_eq_left hcs, hgd]
    show _ + 0 = _
    rw [add_zero]
    exact hself j'

/-- **A flat class's forced-zero rows**: each unit slot's row past the unit's useful rows reads nothing. -/
theorem setupH_flatZero (ht : tags.typed = true)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
    (hr : Typed.read tags cf tables = .ok (k, .inl net)) {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count)
    (r j : Fin (2 ^ st.c.kLog)) (hr' : r.val = st.c.unitRange.slot s₀ + st.c.unit.useful) :
    placedA st r j = 0 ∧ placedB st r j = 0 := by
  obtain ⟨-, hunit, L⟩ := setupH_flatLayout ht h hr
  obtain ⟨-, C, -⟩ := setupH_flatFacts ht h hr
  obtain ⟨-, name, hnet, -⟩ := read_flat' hr
  have hlog : net.unitLog ≤ 32 := hunit ▸ L.unitLog_le hs₀
  obtain ⟨-, -, -, huse, hcst, -⟩ := Typed.netOfD_ok hnet
  have hcu := (ofRows_groups (show Net.ofRows _ _ _ _ _ _ _ = .ok net from hnet)).1
  obtain ⟨hra, hrb⟩ := netOfD_row hnet hlog net.useful
  have hpast : ¬ net.useful < (k.done.getD k.unit default).size := by omega
  rw [ite_eq_right hpast] at hra hrb
  have hlk : st.c.unit.lookup = none := L.unit_lookup
  refine L.placed_zero hs₀ C.unit_zero (by rw [hunit]; omega) (fun j' => ?_) (fun j' => ?_) r j hr'
  · rw [sparseEntry_built_a L.unit_built, hunit, hra]; rfl
  · rw [sparseEntry_built_b L.unit_built, hlk, hunit, hrb]; rfl


/-! ## The flat class's table class -/

section tableClass

variable (ht : tags.typed = true) (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
  {k : Typed.Class} {net : Net} (hr : Typed.read tags cf tables = .ok (k, .inl net))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

/-- Unit slot `g`'s forced-zero row, past the unit's useful rows. -/
def flatZeroPos (g : Fin st.c.unitRange.count) : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).kLog) :=
  ⟨st.c.unitRange.slot g + st.c.unit.useful,
    (setupH_flatLayout ht h hr).2.2.unit_slot_lt g.isLt (setupH_flatFacts ht h hr).2.1.unit_zero⟩

/-- Unit slot `g`'s input `i`'s copy position: the column Δ copies into it (`flatSrcAt`), or the slot's forced-zero row
when that column lies outside the block, where the input's rows are zero. -/
def flatCopyPos (g : Fin st.c.unitRange.count) (i : ℕ) :
    Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).kLog) :=
  if hlt : flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) < 2 ^ st.c.kLog then ⟨_, hlt⟩
  else flatZeroPos ht h hr regs g

/-- The rows at a unit slot's forced-zero row carry 0. -/
theorem flatZero_block (g : Fin st.c.unitRange.count) {z : Model.Witness ((setupH_flatLayout ht h hr).2.2.model regs).m}
    (hz : ((setupH_flatLayout ht h hr).2.2.model regs).Satisfies z) (o : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).nbl)) :
    ((setupH_flatLayout ht h hr).2.2.model regs).block z o (flatZeroPos ht h hr regs g) = 0 := by
  have hZ : ZeroRow ((setupH_flatLayout ht h hr).2.2.model regs) (flatZeroPos ht h hr regs g) :=
    ⟨fun j => (setupH_flatZero ht h hr g.isLt _ j rfl).1, fun j => (setupH_flatZero ht h hr g.isLt _ j rfl).2⟩
  exact hZ.block_eq hz o

/-- **A flat class's input copies**: at unit slot `g`, each input of the unit's composed rows carries what its copy
position carries, in every block of a satisfying witness. -/
theorem setupH_flatCopy (g : Fin st.c.unitRange.count)
    (c : Fin (Rows.compose (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default)) (ofBlock_ordered k.hd)).w)
    (hc : c.val < (Rows.compose (ofBlock (Typed.wordsIn k.held) k.done (k.done.getD k.unit default))
      (ofBlock_ordered k.hd)).nIn)
    {z : Model.Witness ((setupH_flatLayout ht h hr).2.2.model regs).m}
    (hz : ((setupH_flatLayout ht h hr).2.2.model regs).Satisfies z)
    (o : Fin (2 ^ ((setupH_flatLayout ht h hr).2.2.model regs).nbl)) :
    ((setupH_flatLayout ht h hr).2.2.model regs).block z o
        (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
          (setupH_flatRealizes ht h hr regs g.isLt).lt c) =
      ((setupH_flatLayout ht h hr).2.2.model regs).block z o (flatCopyPos ht h hr regs g c.val) := by
  obtain ⟨-, -, htake, -, -⟩ := order_sound k.hd 0
  have hw : c.val < (k.done.getD k.unit default).inCols.size := hc
  have hlen : (k.done.getD k.unit default).inCols.size ≤ (order k.done (k.done.getD k.unit default)).length := by
    have := congrArg List.length htake
    simp only [List.length_take, Array.length_toList] at this
    omega
  have hcol : (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
      (setupH_flatRealizes ht h hr regs g.isLt).lt c).val =
      st.c.unitRange.slot g + (k.done.getD k.unit default).inCols[c.val] := by
    unfold composePlace
    rw [dite_eq_left (show c.val < (order k.done (k.done.getD k.unit default)).length by omega)]
    show st.c.unitRange.slot g + (order k.done (k.done.getD k.unit default)).getD c.val 0 = _
    rw [order_getD_inCols htake hw, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hw, Option.getD_some]
  have hrow := fun j => setupH_flatInput ht h hr g.isLt hw _ j hcol
  have hgd : (k.done.getD k.unit default).inCols.getD c.val 0 = (k.done.getD k.unit default).inCols[c.val] := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hw, Option.getD_some]
  unfold flatCopyPos
  rw [hgd]
  split
  · rename_i hlt
    have hC : CopyRow ((setupH_flatLayout ht h hr).2.2.model regs) _ ⟨_, hlt⟩ :=
      ⟨fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).1, fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).2⟩
    exact hC.block_eq hz o
  · rename_i hlt
    have hZ : ZeroRow ((setupH_flatLayout ht h hr).2.2.model regs)
        (composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
          (setupH_flatRealizes ht h hr regs g.isLt).lt c) := by
      refine ⟨fun j => ?_, fun j => ?_⟩
      · rw [show ((setupH_flatLayout ht h hr).2.2.model regs).A₀ = placedA st from rfl, (hrow j).1,
          ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
      · rw [show ((setupH_flatLayout ht h hr).2.2.model regs).B₀ = placedB st from rfl, (hrow j).2,
          ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
    rw [hZ.block_eq hz o, flatZero_block ht h hr regs g hz o]

/-- **A flat class's table class** (the soundness lane's `TableClass`), from the statement `Stmt.setupH` accepted: its
unit slots, the unit's composed rows placed at each (`setupH_flatPlacement`), each input's copy position
(`setupH_flatCopy`) and the slots' forced-zero rows (`setupH_flatZero`). -/
def setupH_flatTableClass : TableClass ((setupH_flatLayout ht h hr).2.2.model regs) where
  types := Typed.typeIn k.types
  ls := k.ls
  info := Typed.infoIn k.held
  words := Typed.wordsIn k.held
  unit := k.unit
  done := k.done
  hd := k.hd
  slots := st.c.unitRange.count
  col g := composePlace _ _ (ofBlock_ordered k.hd) (fun x => st.c.unitRange.slot g + x)
    (setupH_flatRealizes ht h hr regs g.isLt).lt
  placed g := setupH_flatPlacement ht h hr regs g.isLt
  copyPos := flatCopyPos ht h hr regs
  copy g c hc _ hz o := setupH_flatCopy ht h hr regs g c hc hz o
  zeros := Finset.univ.image (flatZeroPos ht h hr regs)
  zero q hq _ hz o := by
    obtain ⟨g, -, rfl⟩ := Finset.mem_image.1 hq
    exact flatZero_block ht h hr regs g hz o

end tableClass

end FlockSoundness
