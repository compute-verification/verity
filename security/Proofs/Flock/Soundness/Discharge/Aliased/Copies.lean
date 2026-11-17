module

public import Proofs.Flock.Soundness.Discharge.Aliased.Rows
public import Proofs.Flock.Soundness.Discharge.Hm.SatEx
public import Proofs.Flock.Soundness.ExecFlatCopies
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.ExecLookup

/-! `Layout.Aliased` from Δ's copy positions: an instance that wires a unit's stacked inputs as Δ copies them (`Copies`) puts one value on each gate. -/

@[expose] public section

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

structure UnitFacts (tags : Tags) (st : Stmt) : Prop where
  parse : ParseFactsU tags (unitPre st.c) st.c
  nets : NetsFit st.c
  nodup : ∀ i < st.c.nets.size, (portBits (st.c.nets.getD i default).2.inGroups).Nodup
  check : CheckFacts st.c
  delta : FlatDelta st.c st.pin st.da
  sha : st.c.unitNet ≠ st.c.shaNet
  hm : st.c.unitNet ≠ st.c.hmNet
  rowsA : st.c.unit.a.rows ≤ st.c.unit.useful
  rowsB : st.c.unit.b.rows ≤ st.c.unit.useful
  useful : st.c.unit.useful = st.c.unit.constPos + 1

theorem UnitFacts.ofFlat {tags : Tags} {st : Stmt} (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hΔ : FlatDelta st.c st.pin st.da) (hs : st.c.unitNet ≠ st.c.shaNet) (hh : st.c.unitNet ≠ st.c.hmNet)
    (ha : st.c.unit.a.rows ≤ st.c.unit.useful) (hb : st.c.unit.b.rows ≤ st.c.unit.useful) : UnitFacts tags st :=
  ⟨P.toU _, nets_ok P, fun i hi => by
    rcases P.nets _ (getD_mem_toList hi) with hb | ⟨_, _, _, _, _, hb⟩
    · exact portBits_nodup hb
    · rw [(build_facts hb).2.1, portBits_eq]
      simpa using List.nodup_range', C, hΔ, hs, hh, ha, hb, (built_spec (unit_built P C)).2.1⟩

theorem UnitFacts.flat {tags : Tags} {st : Stmt} (F : UnitFacts tags st) (ht : st.c.tmpl = none) :
    ParseFacts tags st.c :=
  F.parse.facts (fun x hx => by rw [unitPre_flat ht] at hx; simp at hx) ht

/-- `UnitFacts` with nets `pre` first, of any kind: a templated file's unit net (`Tmpl.unitNet`) is not `Built`. -/
structure UnitFactsPre (tags : Tags) (pre : Array (String × Net)) (st : Stmt) : Prop where
  parse : ParseFactsPre tags pre st.c
  check : CheckFacts st.c
  delta : FlatDelta st.c st.pin st.da
  sha : st.c.unitNet ≠ st.c.shaNet
  hm : st.c.unitNet ≠ st.c.hmNet
  rowsA : st.c.unit.a.rows ≤ st.c.unit.useful
  rowsB : st.c.unit.b.rows ≤ st.c.unit.useful

theorem UnitFactsPre.facts {tags : Tags} {pre : Array (String × Net)} {st : Stmt} (F : UnitFactsPre tags pre st)
    (hpre : ∀ x ∈ pre.toList, Built x.2) : UnitFacts tags st :=
  .ofFlat (F.parse.facts hpre) F.check F.delta F.sha F.hm F.rowsA F.rowsB

theorem unit_self {st : Stmt} (L : Layout st) (hlog : st.c.unit.unitLog ≤ 32) {x : ℕ}
    (hx : x ∈ portBits st.c.unit.inGroups) (j : ℕ) :
    sparseEntry st.c.unit.a x j = (if j = x then 1 else 0) ∧ sparseEntry st.c.unit.b x j = if j = x then 1 else 0 := by
  have A := L.unit_at
  obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hx
  have hin : x < st.c.unit.inWords * WORD := hpt ▸ L.port_lt hp ht
  have hc : x < (lunit st.c).constPos := by have := A.in_le; rw [A.in_eq] at this; omega
  have hpos := A.inputs x hin
  have hx' : x ∈ portBits (lunit st.c).inGroups := by rw [A.groups]; exact hx
  obtain ⟨ra, rb⟩ := port_row L.unit_built (le_trans A.log_le hlog) hx'
  have hself : (((((#[x] : Array ℕ)).toList.map (upos st.c)).count j : ℕ) : ZMod 2) = if j = x then 1 else 0 := by
    by_cases hj : j = x
    · subst hj; simp [hpos]
    · simp [hpos, hj, Ne.symm hj]
  constructor
  · have e := A.row_a x hc hlog j
    rw [hpos, ra] at e
    exact e.trans hself
  · have e := A.row_b x hc hlog j
    rw [hpos, rb] at e
    exact e.trans hself

theorem unit_nodup {st : Stmt} (L : Layout st) : (portBits st.c.unit.inGroups).Nodup := by
  rw [← L.unit_at.groups]; exact portBits_nodup L.unit_built

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : UnitFacts tags st)

/-- Unit slot `s₀`'s forced-zero row, past the unit's useful rows. -/
def zeroPos {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) : Fin (2 ^ (L.model regs).kLog) :=
  ⟨st.c.unitRange.slot s₀ + st.c.unit.useful, L.unit_slot_lt hs₀ F.check.unit_zero⟩

theorem zeroPos_block_ex {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) {D : ℕ → Prop} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) D z) (hD : ¬ D (zeroPos L regs F hs₀).val) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (zeroPos L regs F hs₀) = 0 := by
  have huse := F.useful
  have hA : ∀ j', sparseEntry st.c.unit.a st.c.unit.useful j' = 0 := fun j' => by
    unfold sparseEntry
    rw [ite_eq_right (by have := F.rowsA; omega)]
  have hB : ∀ j', sparseEntry st.c.unit.b st.c.unit.useful j' + lookupEntry st.c.unit.lookup st.c.unit.useful j' = 0 :=
    fun j' => by
      rw [L.unit_lookup]
      unfold sparseEntry
      rw [ite_eq_right (by have := F.rowsB; omega)]
      rfl
  have hZ : ZeroRow (L.model regs) (zeroPos L regs F hs₀) :=
    ⟨fun j => (L.placed_zero hs₀ F.check.unit_zero huse hA hB _ j rfl).1,
      fun j => (L.placed_zero hs₀ F.check.unit_zero huse hA hB _ j rfl).2⟩
  exact hZ.block_eq_ex hz hD o

theorem zeroPos_block {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (zeroPos L regs F hs₀) = 0 :=
  zeroPos_block_ex L regs F hs₀ (hz.ex fun _ => False) not_false o

/-- Unit slot `s₀`'s input `off`'s copy position: at a port bit, the column Δ copies into it (`flatSrcAt`), or the slot's
forced-zero row when that column lies outside the block; at an input padding row, the row itself. -/
def copyAt {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (off : ℕ) : Fin (2 ^ (L.model regs).kLog) :=
  if off ∈ portBits st.c.unit.inGroups then
    if hlt : flatSrcAt st.c s₀ off < 2 ^ st.c.kLog then ⟨_, hlt⟩ else zeroPos L regs F hs₀
  else if hlt : st.c.unitRange.slot s₀ + off < 2 ^ st.c.kLog then ⟨_, hlt⟩ else zeroPos L regs F hs₀

theorem copyAt_block {s₀ off : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hoff : off < st.c.unit.inWords * WORD)
    (r : Fin (2 ^ (L.model regs).kLog)) (hr : r.val = st.c.unitRange.slot s₀ + off) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o r = (L.model regs).block z o (copyAt L regs F hs₀ off) := by
  unfold copyAt
  by_cases hm : off ∈ portBits st.c.unit.inGroups
  · rw [ite_eq_left hm]
    have hnd := unit_nodup L
    obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
    have hsrc : flatSrcAt st.c s₀ off = Flat.flatSrc st.c s₀ p t := by
      unfold flatSrcAt
      rw [hpt, portOf_eq hnd hp ht]
      congr 1
      omega
    have hself := unit_self L (L.unitLog_le hs₀) hm
    rw [hpt] at hself
    have hrow := fun j => L.placed_portT F.parse (fun n => (F.nets n).1) F.check F.delta F.sha F.hm hnd hs₀ hp ht (fun j' => (hself j').1)
      (fun j' => by rw [L.unit_lookup]; exact (add_zero _).trans (hself j').2) r j (by rw [hr, hpt])
    rw [hsrc]
    split
    · rename_i hlt
      have hC : CopyRow (L.model regs) r ⟨_, hlt⟩ :=
        ⟨fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).1, fun j => by rw [← ite_fin_val j hlt]; exact (hrow j).2⟩
      exact hC.block_eq hz o
    · rename_i hlt
      have hZ : ZeroRow (L.model regs) r := by
        refine ⟨fun j => ?_, fun j => ?_⟩
        · rw [show (L.model regs).A₀ = placedA st from rfl, (hrow j).1,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
        · rw [show (L.model regs).B₀ = placedB st from rfl, (hrow j).2,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
      rw [hZ.block_eq hz o, zeroPos_block L regs F hs₀ hz o]
  · rw [ite_eq_right hm]
    have hlt : st.c.unitRange.slot s₀ + off < 2 ^ st.c.kLog :=
      L.unit_slot_lt hs₀ (lt_of_lt_of_le hoff (L.fit st.c.unitNet).2)
    rw [dite_eq_left hlt]
    exact congrArg _ (Fin.ext hr)

include L in
theorem stack_copy_lt {g x : ℕ} (hg : g < st.c.g) (hx : x < st.c.upv * (st.c.unit.inWords * WORD)) :
    g * st.c.upv + x / (st.c.unit.inWords * WORD) < st.c.unitRange.count := by
  have hn : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
  exact L.slot_lt hg ((Nat.div_lt_iff_lt_mul hn).2 hx)

/-- **Template instance `g`'s stacked unit input `x`'s copy position**: input `x % (inWords·WORD)` of unit slot
`g·upv + x / (inWords·WORD)` (`copyAt`). -/
def copyPos {g : ℕ} (hg : g < st.c.g) (x : ℕ) : Fin (2 ^ (L.model regs).kLog) :=
  if hx : x < st.c.upv * (st.c.unit.inWords * WORD) then
    copyAt L regs F (stack_copy_lt L hg hx) (x % (st.c.unit.inWords * WORD))
  else ⟨0, Nat.two_pow_pos _⟩

theorem copyPos_block {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (L.col regs hg x) = (L.model regs).block z o (copyPos L regs F hg x.val) := by
  have hn := L.unitRows_nIn
  have hx' : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by rw [← hn]; exact hx'
  unfold copyPos
  rw [dite_eq_left hx'']
  refine copyAt_block L regs F _ (Nat.mod_lt _ (by
    exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx''; omega)) _ ?_ hz o
  rw [L.col_val_in regs hg x hx', hn]

def IsZeroPos (q : Fin (2 ^ (L.model regs).kLog)) : Prop :=
  ∃ s₀, ∃ hs₀ : s₀ < st.c.unitRange.count, q = zeroPos L regs F hs₀

/-- Inputs an instance puts on one gate copy one position, or each copies a forced-zero row; the program's construction supplies it. -/
def Copies {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g) : Prop :=
  ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
    c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
      copyPos L regs F hg c.val = copyPos L regs F hg c'.val ∨
        (IsZeroPos L regs F (copyPos L regs F hg c.val) ∧ IsZeroPos L regs F (copyPos L regs F hg c'.val))

theorem aliased_of_copies {C : Verity.Boolean.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies L regs F inst hg) (o : Fin (2 ^ (L.model regs).nbl)) : L.Aliased regs inst o hg := by
  intro c c' hw hc hc' z hz
  rw [copyPos_block L regs F hg c hc hz o, copyPos_block L regs F hg c' hc' hz o]
  rcases hcp c c' hw hc hc' with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_block L regs F hs hz o, zeroPos_block L regs F hs' hz o]

end copies

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

theorem setupH_unitFacts (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) : UnitFacts tags st := by
  obtain ⟨hp, -, hda, -, -, -⟩ := setupH_spec ht h
  obtain ⟨P, ⟨more, hm, hmore⟩, hunit, hsn, hhn⟩ := parse_facts_nets (pre := #[]) (fun _ hx => by simp at hx) hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  have hname : (st.c.nets.getD st.c.unitNet default).1 = "unit" := by
    obtain ⟨hlt, hp', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    simpa using hp'
  have hmem : st.c.nets.getD st.c.unitNet default ∈ more := by
    have : st.c.nets.getD st.c.unitNet default ∈ st.c.nets.toList := getD_mem_toList P.unitNet
    rw [hm] at this
    simpa using this
  have hrows : st.c.unit.a.rows = st.c.unit.useful ∧ st.c.unit.b.rows = st.c.unit.useful := by
    rcases hmore _ hmem with ⟨lines, hl⟩ | ⟨_, _, _, _, _, hb⟩
    · exact parse_rows hl
    · have h1 := (build_facts hb).1
      have h2 : (st.c.nets.getD st.c.unitNet default).2.lookup.isSome = false := C.unit_lookup
      rw [h1] at h2
      cases h2
  refine .ofFlat P C (by rw [hda]; exact delta_flat st.c st.pin P.typed) (fun he => ?_) (fun he => ?_)
    hrows.1.le hrows.2.le
  · rw [← he, hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  · rw [← he, hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)

theorem setupH_aliased (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Verity.Boolean.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies (setupH_layout ht h) regs (setupH_unitFacts ht h) inst hg) :
    (setupH_layout ht h).Aliased regs inst o hg :=
  aliased_of_copies _ regs _ inst hg hcp o

noncomputable def setupH_unitPlace (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Verity.Boolean.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies (setupH_layout ht h) regs (setupH_unitFacts ht h) inst hg) :
    UnitPlace P ((setupH_layout ht h).model regs) u :=
  unitPlace_of_setupH ht h regs inst o hg (setupH_aliased ht h regs inst o hg hcp)

end FlockSoundness.Discharge.Aliased
