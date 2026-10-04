import FlockSoundness.Discharge.ZkPubIn.Spec
import FlockSoundness.Discharge.HiddenExec.Copies
import FlockSoundness.Discharge.HiddenExec.Ret

set_option Elab.async false

/-!
# A statement with the verifier's public inputs and its flat twin

`HiddenExec/Flat.lean` and `HiddenExec/Copies.lean` with `Stmt.setupHidden … (some own)`, whose Δ is `HmIn.delta`:
`HmOut.delta`'s without the pairs whose first column is a public leaf's bit (`HmIn.publicCol`). A public leaf is a
zero leaf of `leaves_in`, so its flat twin (`Hidden.flatTwin`, `HmRow.delta`'s Δ) copies the forced zero into it, and
the statement does not: the two read different rows at a public leaf's bits, and the same rows at every other unit
slot's row (`UnitRowsFlatP`, `rowsFlatP`).

* `publicCol_spec`: a public column is bit `t` of leaf port `j` of a unit slot, a public leaf of its unit;
* `publicCol_not_const`, `layoutP`: no pair of Δ's first loop is at a public column, so the statement has the layout
  the placement reads;
* `publicCol_slot_none`: the row of a VU-stacked input that is no public leaf's bit (`pubLeaf`) is no public column;
* `copyAt_blockHP`, `copyPos_blockHP`, `aliased_of_copiesHP`: `copyAt_blockH`, `copyPos_blockH` and
  `aliased_of_copiesH` at those inputs, for an instance whose public leaf bits are each alone on their gate.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 HiddenExec

/-- **The statement reads its flat twin's rows at every unit slot's row that is no public leaf's bit**, in `A` and in
`B`, at the public input ports `HmIn.parse` reads. -/
def UnitRowsFlatP (st : Stmt) : Prop :=
  ∀ pi, HmIn.parse st.c = .ok pi → ∀ s₀ < st.c.unitRange.count, ∀ b < 2 ^ st.c.unit.unitLog,
    ∀ r j : Fin (2 ^ st.c.kLog), r.val = st.c.unitRange.slot s₀ + b → HmIn.publicCol st.c pi r.val = false →
      placedA st r j = placedA (Hidden.flatTwin st) r j ∧ placedB st r j = placedB (Hidden.flatTwin st) r j

/-! ## Public columns -/

section publicCol

variable {st : Stmt} (L : Layout st) (C : CheckFacts st.c) {pi : HmIn.Ports} (lp : LeavesPublic st.c pi)

private theorem getD_size_of_nonneg {a : Array ℤ} {j : ℕ} (h : 0 ≤ a.getD j (-1)) : j < a.size := by
  by_contra hj
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none (by omega), Option.getD_none] at h
  omega

private theorem slot_split (r : Range) {x : ℕ} (h : r.start ≤ x) :
    x = r.slot ((x - r.start) / 2 ^ r.slotLog) + (x - r.start) % 2 ^ r.slotLog := by
  have hd := Nat.div_add_mod (x - r.start) (2 ^ r.slotLog)
  have : r.slot ((x - r.start) / 2 ^ r.slotLog) = r.start + 2 ^ r.slotLog * ((x - r.start) / 2 ^ r.slotLog) := by
    simp only [Range.slot, Range.start]; ring
  omega

include L C lp in
/-- **A public column is bit `t` of a public leaf**: unit slot `k`'s leaf port `j` (16 bits wide), at column
`portCols[j].1 + t` of the slot, with `j` a public leaf of unit `k % upv`. -/
theorem publicCol_spec {x : ℕ} (h : HmIn.publicCol st.c pi x = true) :
    ∃ k < st.c.unitRange.count, ∃ j < (st.c.unit.inGroups.getD 0 default).bits.size, ∃ t < 16,
      j < (portCols st.c.unit.inGroups).size ∧ (portCols st.c.unit.inGroups)[j]!.2 = 16 ∧
      x = st.c.unitRange.slot k + ((portCols st.c.unit.inGroups)[j]!.1 + t) ∧
      0 ≤ (pi.leaves.getD (k % st.c.upv) #[]).getD j (-1) := by
  unfold HmIn.publicCol at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨h1, h2⟩, h3⟩, h4⟩ := h
  have hx := slot_split st.c.unitRange h1
  set k := (x - st.c.unitRange.start) / 2 ^ st.c.unitRange.slotLog with hk
  set a := (x - st.c.unitRange.start) % 2 ^ st.c.unitRange.slotLog with ha
  have hd16 := Nat.div_add_mod (a - (st.c.unit.inGroups.getD 0 default).col * WORD) 16
  set j := (a - (st.c.unit.inGroups.getD 0 default).col * WORD) / 16 with hj
  have hupv : 0 < st.c.upv := by
    rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
    · rw [L.unit_count, h0, Nat.mul_zero] at h2; exact absurd h2 (Nat.not_lt_zero _)
    · exact h0
  have hu : k % st.c.upv < st.c.upv := Nat.mod_lt _ hupv
  have hjl := getD_size_of_nonneg h4
  rw [(lp.maps _ hu).1] at hjl
  have hlv : (st.c.leavesIn.getD (k % st.c.upv) #[]) = st.c.leavesIn[k % st.c.upv]! := by
    rw [getElem!_pos _ _ (by rw [C.leaves.1]; exact hu), Array.getD_eq_getD_getElem?,
      Array.getElem?_eq_getElem (by rw [C.leaves.1]; exact hu), Option.getD_some]
  rw [hlv, C.leaves.2 _ hu] at hjl
  obtain ⟨hp, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjl
  have h16 := portCols_16_get C.unit_groups.2 hjl
  have hb16 : (st.c.unit.inGroups.getD 0 default).bits[j]! = 16 :=
    C.unit_groups.2 _ (by rw [getElem!_pos (st.c.unit.inGroups.getD 0 default).bits j hjl]; exact Array.getElem_mem_toList hjl)
  have hsz : (portCols #[(st.c.unit.inGroups.getD 0 default)]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
    simpa using congrArg List.length (portCols_single (st.c.unit.inGroups.getD 0 default))
  have hjs : j < (portCols #[(st.c.unit.inGroups.getD 0 default)]).size := by rw [hsz]; exact hjl
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_eq_getElem hjs, Option.map_some,
    Option.getD_some] at hcol
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hjs, Option.getD_some] at h16
  refine ⟨k, h2, j, hjl, (a - (st.c.unit.inGroups.getD 0 default).col * WORD) % 16, Nat.mod_lt _ (by norm_num), hp, hw.trans hb16, ?_, h4⟩
  rw [← hcol, h16]
  omega

include L C lp in
/-- **No pair of Δ's first loop is at a public column**: a public column is a unit slot's input bit, below the unit's
constant (`Layout.constPairs_not_below`). -/
theorem publicCol_not_const {x : ℕ} (h : HmIn.publicCol st.c pi x = true) (j' : ℕ) :
    (x, j') ∉ constPairs st.c st.pin := by
  obtain ⟨k, hk, j, -, t, ht, hp, hw, rfl, -⟩ := publicCol_spec L C lp h
  exact L.constPairs_not_below L.unit_mem L.unit_kind hk
    (lt_of_lt_of_le (L.port_lt hp (by rw [hw]; exact ht)) L.unit_in) j'

include L C lp in
/-- **The row of a VU-stacked input that is no public leaf's bit is no public column**: input `x` of VU `g` sits at
column `x % W` of unit slot `g · upv + x / W` (`W = inWords · WORD`); were that a public column, it would be bit `t`
of a public leaf port `j`, and `pubLeaf` would read it. -/
theorem publicCol_slot_none (g : ℕ) {x : ℕ} (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hpub : pubLeaf st.c pi x = none) :
    HmIn.publicCol st.c pi
      (st.c.unitRange.slot (g * st.c.upv + x / (st.c.unit.inWords * WORD)) + x % (st.c.unit.inWords * WORD)) =
        false := by
  cases hc : HmIn.publicCol st.c pi
      (st.c.unitRange.slot (g * st.c.upv + x / (st.c.unit.inWords * WORD)) + x % (st.c.unit.inWords * WORD)) with
  | false => rfl
  | true =>
    exfalso
    obtain ⟨k, -, j, hjb, t, ht, hp, hw, he, hq⟩ := publicCol_spec L C lp hc
    have hW : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
    have hxW : x / (st.c.unit.inWords * WORD) < st.c.upv := (Nat.div_lt_iff_lt_mul hW).2 hx
    have hoff : x % (st.c.unit.inWords * WORD) < 2 ^ st.c.unitRange.slotLog := by
      rw [L.unit_slotLog]
      exact lt_of_lt_of_le (Nat.mod_lt _ hW) (L.fit st.c.unitNet).2
    obtain ⟨hk, hoe⟩ := slot_add_inj hoff (L.port_lt_slot hp (by rw [hw]; exact ht)) he
    have hm : x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups := by
      rw [hoe]
      exact List.mem_flatMap.2 ⟨(portCols st.c.unit.inGroups)[j]!, by
        rw [getElem!_pos (portCols st.c.unit.inGroups) j hp]; exact Array.getElem_mem_toList hp,
        List.mem_range'_1.2 ⟨by omega, by omega⟩⟩
    have hpo : portOf (portCols st.c.unit.inGroups) (x % (st.c.unit.inWords * WORD)) = j := by
      rw [hoe]; exact portOf_eq (Aliased.portBits_nodup L.unit_built) hp (by rw [hw]; exact ht)
    have hku : k % st.c.upv = x / (st.c.unit.inWords * WORD) := by
      rw [← hk, Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hxW]
    rw [hku] at hq
    unfold pubLeaf at hpub
    simp only [hm, hpo, hjb, and_self, ↓reduceIte, hq] at hpub
    cases hpub

end publicCol

/-! ## The flat twin, with the public inputs -/

/-- Filtering pairs by a predicate that holds at every pair of row `r` keeps their entries at row `r`. -/
theorem deltaEntry_filter {a : Array (ℕ × ℕ)} {p : ℕ × ℕ → Bool} {r : ℕ} (hp : ∀ j', p (r, j') = true) (j : ℕ) :
    deltaEntry (a.filter p) r j = deltaEntry a r j := by
  unfold deltaEntry
  rw [Array.toList_filter, List.filter_filter]
  congr 2
  refine List.filter_congr fun x _ => ?_
  by_cases hx : x = (r, j)
  · subst hx; simp [hp]
  · simp [hx]

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt}

/-- **The flat twin's layout** (`layout_of`), with the public inputs. -/
theorem flat_layoutP (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    Layout (Hidden.flatTwin st) :=
  have s := (setupHidden_specP h).choose_spec
  layout_of (st := Hidden.flatTwin st) (parse_facts (Hidden.parse_untyped ht s.parse).1) s.pin rfl rfl s.m s.kLog

/-- **The flat twin's unit facts** (`Hidden.unitFacts_of_parse`), with the public inputs. -/
theorem flat_unitFactsP (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) :=
  Hidden.unitFacts_of_parse (st := Hidden.flatTwin st)
    (Hidden.parse_untyped ht (setupHidden_specP h).choose_spec.parse).1 rfl

/-- Δ with the public inputs: `HmRow.delta`'s, then `outPairs`, without the pairs at a public column. -/
theorem delta_filterP (ht : tags.typed = false) {pi : HmIn.Ports}
    (s : SpecP tags cf pf coins tables draw own pi st) :
    st.da = ((HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray).filter
        (fun x => !HmIn.publicCol st.c pi x.1) ∧
      st.db = ((HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray).filter
        (fun x => !HmIn.publicCol st.c pi x.1) := by
  have hP := parse_facts (Hidden.parse_untyped ht s.parse).1
  have hd := Hidden.hidden_delta' st.c st.pin
  refine ⟨?_, ?_⟩
  · rw [s.da]
    unfold HmIn.delta
    simp only [s.ports, Bool.false_eq_true, ↓reduceIte, hd]
  · rw [s.db]
    unfold HmIn.delta
    simp only [s.ports, Bool.false_eq_true, ↓reduceIte, hd, delta_snd _ _ hP.typed]

/-- **The statement has the layout the placement reads** (`Hidden.hidden_layout'` with the public inputs): Δ's first
loop's pairs are at no public column (`publicCol_not_const`), so `HmIn.delta`'s filter keeps them, and every pair it
keeps past them is an input bit of a net slot. -/
theorem layoutP (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st) : Layout st := by
  obtain ⟨pi, s⟩ := setupHidden_specP h
  obtain ⟨hp, hc⟩ := Hidden.parse_untyped ht s.parse
  have P := parse_facts hp
  obtain ⟨ls, hcheck⟩ := P.check
  have C := check_facts P.typed hcheck
  have L := flat_layoutP ht h
  obtain ⟨hda, hdb⟩ := delta_filterP ht s
  obtain ⟨rest, h1, h2⟩ := L.delta
  exact {
    disj := L.disj
    bound := L.bound
    slotLog := L.slotLog
    fit := L.fit
    unit_mem := L.unit_mem
    unit_kind := L.unit_kind
    unit_count := L.unit_count
    unit_built := L.unit_built
    unit_lookup := L.unit_lookup
    unit_in := L.unit_in
    unit_const := L.unit_const
    pin := L.pin
    delta_b := by rw [hda, hdb]
    delta := by
      refine ⟨(rest ++ Hidden.outPairs st.c).filter (fun x => !HmIn.publicCol st.c pi x.1), ?_, fun x hx => ?_⟩
      · have h1' : (HmRow.delta st.c st.pin).1.toList = constPairs st.c st.pin ++ rest := h1
        rw [hda, Array.toList_filter, Array.toList_append, h1', List.toList_toArray, List.append_assoc,
          List.filter_append]
        congr 1
        refine List.filter_eq_self.2 fun x hx => ?_
        cases hx' : HmIn.publicCol st.c pi x.1 with
        | false => rfl
        | true => exact absurd hx (publicCol_not_const L C s.leavesPublic hx' x.2)
      · obtain ⟨hx, -⟩ := List.mem_filter.1 hx
        rcases List.mem_append.1 hx with hx | hx
        · exact h2 x hx
        · exact Hidden.outPairs_inBit hp hc x hx
    kLog := L.kLog
    kLog6 := L.kLog6
    m := L.m }

/-- **The statement reads its flat twin's rows at every unit slot's row that is no public column**
(`unitRowsFlat` with the public inputs), for `sha512/row/v1` rows: there `HmIn.delta`'s filter keeps every pair, and
no output pair is at a unit slot's row. -/
theorem rowsFlatP (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program (some own) = .ok st)
    (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false) : UnitRowsFlatP st := by
  obtain ⟨pi₀, s⟩ := setupHidden_specP h
  obtain ⟨hp, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layoutP ht h
  have F := flat_unitFactsP ht h
  intro pi hpi s₀ hs₀ b hb r j hr hpc
  have hpi' : pi₀ = pi := Except.ok.inj (s.inParse.symm.trans hpi)
  subst hpi'
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F.parse F.check F.sha hv1 (checkRows_le hcr) hq hs₀ hb
  have hkeep : ∀ j', (fun x : ℕ × ℕ => !HmIn.publicCol st.c pi₀ x.1) (r.val, j') = true := fun j' => by
    simp [hpc]
  obtain ⟨hda, hdb⟩ := delta_filterP ht s
  have hP := parse_facts hp
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, deltaEntry_filter hkeep, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, deltaEntry_filter hkeep, Hidden.deltaEntry_append_out hout, delta_snd _ _ hP.typed]

/-! ## `Layout.Aliased` from the flat twin's copy positions, off the public leaves -/

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (Lf : Layout (Hidden.flatTwin st))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  (hR : UnitRowsFlatP st) {pi : HmIn.Ports} (hpi : HmIn.parse st.c = .ok pi) (lp : LeavesPublic st.c pi)

open FlockSoundness.Discharge.Aliased (portBits_nodup port_self mem_portBits)

include hR hpi in
/-- **A unit slot's input that is no public column carries what its copy position carries** (`copyAt_blockH`). -/
theorem copyAt_blockHP {s₀ off : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hoff : off < st.c.unit.inWords * WORD)
    (hpc : HmIn.publicCol st.c pi (st.c.unitRange.slot s₀ + off) = false)
    (r : Fin (2 ^ (L.model regs).kLog)) (hr : r.val = st.c.unitRange.slot s₀ + off) {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o r = (L.model regs).block z o (Aliased.copyAt Lf regs F hs₀ off) := by
  unfold Aliased.copyAt
  by_cases hm : off ∈ portBits st.c.unit.inGroups
  · rw [ite_eq_left hm]
    have hnd := portBits_nodup L.unit_built
    obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
    have hsrc : flatSrcAt st.c s₀ off = Flat.flatSrc st.c s₀ p t := by
      unfold flatSrcAt
      rw [hpt, portOf_eq hnd hp ht]
      congr 1
      omega
    have hself := port_self L.unit_built (L.unitLog_le hs₀) hm
    rw [hpt] at hself
    have hb : (portCols st.c.unit.inGroups)[p]!.1 + t < 2 ^ st.c.unit.unitLog :=
      lt_of_lt_of_le (L.port_lt hp ht) (L.fit st.c.unitNet).2
    have hrow := fun j => Lf.placed_port F.parse F.check F.delta F.sha F.hm hnd hs₀ hp ht (fun j' => (hself j').1)
      (fun j' => by rw [Lf.unit_lookup]; exact (add_zero _).trans (hself j').2) r j (by rw [hr, hpt])
    have hrow' : ∀ j, placedA st r j = (if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0) ∧
        placedB st r j = if j.val = Flat.flatSrc st.c s₀ p t then 1 else 0 := fun j => by
      obtain ⟨eA, eB⟩ := hR pi hpi s₀ hs₀ _ hb r j (by rw [hr, hpt]) (by rw [hr]; exact hpc)
      rw [eA, eB]
      exact hrow j
    rw [hsrc]
    split
    · rename_i hlt
      have hC : CopyRow (L.model regs) r ⟨_, hlt⟩ :=
        ⟨fun j => by rw [← ite_fin_val j hlt]; exact (hrow' j).1, fun j => by rw [← ite_fin_val j hlt]; exact (hrow' j).2⟩
      exact hC.block_eq hz o
    · rename_i hlt
      have hZ : ZeroRow (L.model regs) r := by
        refine ⟨fun j => ?_, fun j => ?_⟩
        · rw [show (L.model regs).A₀ = placedA st from rfl, (hrow' j).1,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
        · rw [show (L.model regs).B₀ = placedB st from rfl, (hrow' j).2,
            ite_eq_right (by have := j.isLt; intro he; exact hlt (he ▸ this))]
      rw [hZ.block_eq hz o, zeroPos_blockH L Lf regs F hs₀ hz o]
  · rw [ite_eq_right hm]
    have hlt : st.c.unitRange.slot s₀ + off < 2 ^ st.c.kLog :=
      L.unit_slot_lt hs₀ (lt_of_lt_of_le hoff (L.fit st.c.unitNet).2)
    rw [dite_eq_left hlt]
    exact congrArg _ (Fin.ext hr)

include hR hpi lp in
/-- **Each of VU `g`'s stacked unit inputs that is no public leaf's bit carries what its copy position carries**
(`copyPos_blockH`): its row is no public column (`publicCol_slot_none`). -/
theorem copyPos_blockHP {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hpub : pubLeaf st.c pi x.val = none)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x) =
      (L.model regs).block z o (Aliased.copyPos Lf regs F hg x.val) := by
  have hn := L.unitRows_nIn
  have hx' : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by rw [← hn]; exact hx'
  unfold Aliased.copyPos
  rw [dite_eq_left hx'']
  refine copyAt_blockHP L Lf regs F hR hpi _ (Nat.mod_lt _ (by
    exact Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx''; omega))
    (publicCol_slot_none L F.check lp g hx'' hpub) _ ?_ hz o
  unfold stackPlace
  rw [dite_eq_left hx']
  show st.c.unitRange.slot (g * st.c.upv + x.val / (unitRows st L.unit_built).nIn) +
    x.val % (unitRows st L.unit_built).nIn = _
  rw [hn]

include hR hpi lp in
/-- **The unit's inputs on one gate carry one value** (`aliased_of_copiesH`), for an instance that wires them as Δ
copies them in the flat twin (`Aliased.Copies`) and puts each public leaf's bit on a gate of its own. -/
theorem aliased_of_copiesHP {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hcp : Aliased.Copies Lf regs F inst hg)
    (hpub : ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
      c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn → c ≠ c' →
        pubLeaf st.c pi c.val = none ∧ pubLeaf st.c pi c'.val = none)
    (o : Fin (2 ^ (L.model regs).nbl)) : L.Aliased regs inst o hg := by
  intro c c' hw hc hc' z hz
  by_cases e : c = c'
  · rw [e]
  obtain ⟨h1, h2⟩ := hpub c c' hw hc hc' e
  rw [copyPos_blockHP L Lf regs F hR hpi lp hg c hc h1 hz o, copyPos_blockHP L Lf regs F hR hpi lp hg c' hc' h2 hz o]
  rcases hcp c c' hw hc hc' with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_blockH L Lf regs F hs hz o, zeroPos_blockH L Lf regs F hs' hz o]

end copies

end FlockSoundness.Discharge.ZkPubIn
