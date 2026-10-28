import Proofs.Flock.Soundness.Discharge.Aliased.Rows
import Proofs.Flock.Soundness.Discharge.Hm.SatEx

/-!
# `Layout.Aliased` from Δ's copy positions (`pp.aliased`)

At a unit slot's input port bit the block rows are the net's self row and Δ's one copy pair (`Layout.placed_port`, from
`HmRow.check`'s exactly-once wiring), so they copy the column Δ copies there (`flatSrcAt`), or are zero when that
column lies outside the block. Each unit slot's row past its useful rows is a forced-zero row (`Layout.placed_zero`):
the net's row there is empty (`parse_rows`), no slot constant is there, and Δ's other pairs are at input bits. So in
every witness satisfying the accepted statement:
- `copyPos_block`: each of template instance `g`'s stacked unit inputs carries what its copy position (`copyPos`)
  carries;
- `zeroPos_block`: each unit slot's forced-zero row (`zeroPos`) carries 0. Δ copies these into zero leaves (the slot's
  own) and into cut bits past a row word (the range's first slot's).

`Layout.Aliased` quantifies over any instance of the unit's rows in any audit circuit, so it is not a fact of the
statement alone: two inputs an instance puts on one gate can copy different sources. `aliased_of_copies` derives it from
`Copies`, which relates the instance's wiring to the copy positions: inputs on one gate copy one position, or each copies
a forced-zero row. `setupH_aliased` and `setupH_unitPlace` instantiate it at the untyped statement `Stmt.setupH`
accepted.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

/-- **What the copy positions read beside the `Layout`**: the circuit's parse and check, Δ pair by pair, a unit that is
neither of the rows' circuits, and a unit net whose rows past its useful rows are empty. -/
structure UnitFacts (tags : Tags) (st : Stmt) : Prop where
  parse : ParseFacts tags st.c
  check : CheckFacts st.c
  delta : FlatDelta st.c st.pin st.da
  sha : st.c.unitNet ≠ st.c.shaNet
  hm : st.c.unitNet ≠ st.c.hmNet
  rowsA : st.c.unit.a.rows ≤ st.c.unit.useful
  rowsB : st.c.unit.b.rows ≤ st.c.unit.useful

/-- **`UnitFacts` with nets `pre` first, of any kind** (a templated file's unit net, `Tmpl.unitNet`, is not `Built`): its
parse is `ParseFactsPre`'s. -/
structure UnitFactsPre (tags : Tags) (pre : Array (String × Net)) (st : Stmt) : Prop where
  parse : ParseFactsPre tags pre st.c
  check : CheckFacts st.c
  delta : FlatDelta st.c st.pin st.da
  sha : st.c.unitNet ≠ st.c.shaNet
  hm : st.c.unitNet ≠ st.c.hmNet
  rowsA : st.c.unit.a.rows ≤ st.c.unit.useful
  rowsB : st.c.unit.b.rows ≤ st.c.unit.useful

/-- `UnitFactsPre` with `pre`'s nets `Built` is `UnitFacts`. -/
theorem UnitFactsPre.facts {tags : Tags} {pre : Array (String × Net)} {st : Stmt} (F : UnitFactsPre tags pre st)
    (hpre : ∀ x ∈ pre.toList, Built x.2) : UnitFacts tags st :=
  ⟨F.parse.facts hpre, F.check, F.delta, F.sha, F.hm, F.rowsA, F.rowsB⟩

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : UnitFacts tags st)

/-- Unit slot `s₀`'s forced-zero row, past the unit's useful rows. -/
def zeroPos {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) : Fin (2 ^ (L.model regs).kLog) :=
  ⟨st.c.unitRange.slot s₀ + st.c.unit.useful, L.unit_slot_lt hs₀ F.check.unit_zero⟩

/-- **A unit slot's forced-zero row carries 0**, in every block of a witness that satisfies it (`SatEx`). -/
theorem zeroPos_block_ex {s₀ : ℕ} (hs₀ : s₀ < st.c.unitRange.count) {D : ℕ → Prop} {z : Model.Witness (L.model regs).m}
    (hz : SatEx (L.model regs) D z) (hD : ¬ D (zeroPos L regs F hs₀).val) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (zeroPos L regs F hs₀) = 0 := by
  have huse := (built_spec (L.built_flat F.parse.tmpl)).2.1
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

/-- **A unit slot's forced-zero row carries 0**, in every block of a satisfying witness. -/
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

/-- **A unit slot's input carries what its copy position carries**, in every block of a satisfying witness. -/
theorem copyAt_block {s₀ off : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hoff : off < st.c.unit.inWords * WORD)
    (r : Fin (2 ^ (L.model regs).kLog)) (hr : r.val = st.c.unitRange.slot s₀ + off) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o r = (L.model regs).block z o (copyAt L regs F hs₀ off) := by
  unfold copyAt
  by_cases hm : off ∈ portBits st.c.unit.inGroups
  · rw [ite_eq_left hm]
    have hnd := portBits_nodup (L.built_flat F.parse.tmpl)
    obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
    have hsrc : flatSrcAt st.c s₀ off = Flat.flatSrc st.c s₀ p t := by
      unfold flatSrcAt
      rw [hpt, portOf_eq hnd hp ht]
      congr 1
      omega
    have hself := port_self (L.built_flat F.parse.tmpl) (L.unitLog_le hs₀) hm
    rw [hpt] at hself
    have hrow := fun j => L.placed_port F.parse F.check F.delta F.sha F.hm hnd hs₀ hp ht (fun j' => (hself j').1)
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

/-- **Each of template instance `g`'s stacked unit inputs carries what its copy position carries**, at its place
(`Layout.col`, which `Layout.Aliased` reads), in every block of a satisfying witness. -/
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

/-- A forced-zero row of one of the unit slots. -/
def IsZeroPos (q : Fin (2 ^ (L.model regs).kLog)) : Prop :=
  ∃ s₀, ∃ hs₀ : s₀ < st.c.unitRange.count, q = zeroPos L regs F hs₀

/-- **An instance wires template instance `g`'s stacked unit inputs as Δ copies them**: inputs it puts on one gate copy
one position, or each copies a forced-zero row. A relation between the instance and the statement, which the program's
construction supplies. -/
def Copies {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g) : Prop :=
  ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
    c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
      copyPos L regs F hg c.val = copyPos L regs F hg c'.val ∨
        (IsZeroPos L regs F (copyPos L regs F hg c.val) ∧ IsZeroPos L regs F (copyPos L regs F hg c'.val))

/-- **The unit's inputs on one gate carry one value** (`Layout.Aliased`), at template instance `g`'s stacked unit slots
in every block, for an instance that wires them as Δ copies them (`Copies`). -/
theorem aliased_of_copies {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies L regs F inst hg) (o : Fin (2 ^ (L.model regs).nbl)) : L.Aliased regs inst o hg := by
  intro c c' hw hc hc' z hz
  rw [copyPos_block L regs F hg c hc hz o, copyPos_block L regs F hg c' hc' hz o]
  rcases hcp c c' hw hc hc' with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_block L regs F hs hz o, zeroPos_block L regs F hs' hz o]

end copies

/-! ## At the untyped statement `Stmt.setupH` accepted -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}

/-- **What the copy positions read, from the untyped statement `Stmt.setupH` accepted**: `HmRow.parse`'s nets and
`HmRow.check`, Δ pair by pair (`delta_flat`), a unit named `unit` (so neither `sha512x3` nor `hm96`), and a unit net
`Net.parse` read (a lookup slot is refused as the unit), whose rows past its useful rows are empty (`parse_rows`). -/
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
  refine ⟨P, C, by rw [hda]; exact delta_flat st.c st.pin P.typed, fun he => ?_, fun he => ?_,
    hrows.1.le, hrows.2.le⟩
  · rw [← he, hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  · rw [← he, hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)

/-- **`pp.aliased` at the untyped statement `Stmt.setupH` accepted** (`unitPlace_of_setupH`'s `hal`): for an instance
that wires template instance `g`'s stacked unit inputs as Δ copies them (`Copies`). -/
theorem setupH_aliased (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies (setupH_layout ht h) regs (setupH_unitFacts ht h) inst hg) :
    (setupH_layout ht h).Aliased regs inst o hg :=
  aliased_of_copies _ regs _ inst hg hcp o

/-- **A drawn unit's place, from the executable's accept step** (`unitPlace_of_setupH` with `setupH_aliased`): template
instance `g`'s stacked unit slots in block `o`, for an instance that wires their inputs as Δ copies them. -/
noncomputable def setupH_unitPlace (ht : tags.typed = false)
    (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {C : Audit.Circuit} {n : ℕ}
    {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st (setupH_layout ht h).unit_built).stack st.c.upv))
    (o : Fin (2 ^ ((setupH_layout ht h).model regs).nbl)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Copies (setupH_layout ht h) regs (setupH_unitFacts ht h) inst hg) :
    UnitPlace P ((setupH_layout ht h).model regs) u :=
  unitPlace_of_setupH ht h regs inst o hg (setupH_aliased ht h regs inst o hg hcp)

end FlockSoundness.Discharge.Aliased
