import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
import Proofs.Flock.Soundness.Discharge.Hidden.LayoutP
import Proofs.Flock.Soundness.Discharge.Layout.InCover
import Proofs.Flock.Soundness.Discharge.Aliased.KeysP
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret

/-!
# `Layout.Aliased` at a hidden-output statement with public inputs

`Copies.lean` at a statement `Stmt.setupHidden` accepted at the verifier's public inputs (`Hidden.SpecP`). Its Δ is
`HmOut.delta`'s without the pairs into a public leaf's bits (`Hidden.deltaEntry_inDelta`), so:

* `unitRowsFlatP`: every unit slot's row it keeps (`Hidden.keepCol`) reads the flat twin's rows (`UnitRowsFlatP`), and
  `copyPos_blockHP`: a stacked unit input there carries what its twin's copy position carries (`copyAt_blockR`);
* an input off a public leaf (`Aliased.pubLeaf` negative) is at a kept row (`keep_of_neg`);
* `pub_block`: an input at a public leaf, which no Δ pair reaches, carries bit `t` of its leaf's word in the in regions
  (`Layout.in_pinned`), the same word at every unit of the template instance (`Layout.inVal`);
* `aliased_of_keyP`: an instance that puts inputs on one gate only when they have one key with public inputs
  (`Aliased.srcKeyP`) is aliased.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3
open Hidden (keepCol)
open FlockSoundness.Discharge.Aliased (portBits_nodup mem_portBits pubLeaf srcKeyP)
open FlockSoundness.Discharge.Layout (inAddr inVal inBlockOr inCol in_pinned in_cover)

/-- **The hidden statement at public inputs reads its flat twin's rows at every unit slot's row it keeps**, in `A` and
in `B`. -/
def UnitRowsFlatP (st : Stmt) (pi : HmIn.Ports) : Prop :=
  ∀ s₀ < st.c.unitRange.count, ∀ b < 2 ^ st.c.unit.unitLog, ∀ r j : Fin (2 ^ st.c.kLog),
    r.val = st.c.unitRange.slot s₀ + b → keepCol st.c pi r.val = true →
      placedA st r j = placedA (Hidden.flatTwin st) r j ∧ placedB st r j = placedB (Hidden.flatTwin st) r j

section spec

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw : Option Lean.Json} {own : Option ByteArray} {pi : HmIn.Ports} {st : Stmt}

/-- The flat twin's layout, at public inputs (`flat_layout`). -/
theorem flat_layoutP (ht : tags.typed = false) (s : Hidden.SpecP tags cf pf coins tables draw own pi st) :
    Layout (Hidden.flatTwin st) :=
  layout_of (st := Hidden.flatTwin st) (parse_facts (Hidden.parse_untyped ht s.parse).1) s.pin rfl rfl s.m s.kLog

/-- The flat twin's unit facts, at public inputs (`flat_unitFacts`). -/
theorem flat_unitFactsP (ht : tags.typed = false) (s : Hidden.SpecP tags cf pf coins tables draw own pi st) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) :=
  Hidden.unitFacts_of_parse (st := Hidden.flatTwin st) (Hidden.parse_untyped ht s.parse).1 rfl

/-- **An untyped hidden statement at public inputs reads its flat twin's rows at every unit slot's row it keeps**, at
unsegmented output rows (`unitRowsFlat`, with `Hidden.deltaEntry_kept`). -/
theorem unitRowsFlatP (ht : tags.typed = false) (s : Hidden.SpecP tags cf pf coins tables draw own pi st)
    (hrows : Hidden.Any.OutRowsS st.c) : UnitRowsFlatP st pi := by
  have L := flat_layoutP ht s
  have F := flat_unitFactsP ht s
  intro s₀ hs₀ b hb r j hr hk
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unit L F hrows hq hs₀ hb
  obtain ⟨eA, eB⟩ := Hidden.deltaEntry_kept (pin := st.pin) hk j.val
  have hd1 : (HmOut.delta st.c st.pin).1 = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by
    rw [Hidden.hidden_delta']
  have hd2 : (HmOut.delta st.c st.pin).2 = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by
    rw [Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [s.da, eA, hd1, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [s.db, eB, hd2, Hidden.deltaEntry_append_out hout]

end spec

/-! ## The leaf group's ports -/

/-- A port's bit is a port bit. -/
theorem port_mem {gs : Array Group} {p t : ℕ} (hp : p < (portCols gs).size) (ht : t < (portCols gs)[p]!.2) :
    (portCols gs)[p]!.1 + t ∈ portBits gs := by
  unfold portBits
  refine List.mem_flatMap.2 ⟨(portCols gs)[p]!, ?_, List.mem_range'_1.2 ⟨by omega, by omega⟩⟩
  rw [getElem!_pos (portCols gs) p hp]
  exact Array.getElem_mem_toList hp

theorem map_fst_getD (a : Array (ℕ × ℕ)) (p : ℕ) : (a.map (·.1)).getD p 0 = (a.getD p default).1 := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases a[p]? <;> rfl

/-- **Port `p` of the unit's leaf group** is port `p` of the unit, 16 bits from column `col · WORD + 16 p`. -/
theorem leafPort {c : Circuit} (C : CheckFacts c) {p : ℕ} (hp : p < (c.unit.inGroups.getD 0 default).bits.size) :
    p < (portCols c.unit.inGroups).size ∧
      (portCols c.unit.inGroups)[p]!.1 = (c.unit.inGroups.getD 0 default).col * WORD + 16 * p ∧
      (portCols c.unit.inGroups)[p]!.2 = 16 := by
  obtain ⟨hps, h1, h2⟩ := firstGroup_port C.unit_groups.1 hp
  refine ⟨hps, by rw [← h1, map_fst_getD, portCols_16_get C.unit_groups.2 hp], ?_⟩
  rw [h2]
  refine C.unit_groups.2 _ ?_
  rw [getElem!_pos _ p hp]
  exact Array.getElem_mem_toList hp

/-- **A stacked input at a public leaf** is bit `t` of port `p` of its unit's leaf group, its key bit `t` of its public
word. -/
theorem pubLeaf_at {c : Circuit} {pi : HmIn.Ports} (C : CheckFacts c) (hnd : (portBits c.unit.inGroups).Nodup) {x : ℕ}
    (h : 0 ≤ pubLeaf c pi x) :
    ∃ p < (c.unit.inGroups.getD 0 default).bits.size, ∃ t < 16,
      x % (c.unit.inWords * WORD) = (c.unit.inGroups.getD 0 default).col * WORD + 16 * p + t ∧
      pubLeaf c pi x = (pi.leaves.getD (x / (c.unit.inWords * WORD)) #[]).getD p (-1) ∧
      srcKeyP c pi x = .pub (pubLeaf c pi x).toNat t := by
  have hc : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups ∧
      portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD)) <
        (c.unit.inGroups.getD 0 default).bits.size := by
    by_contra hn
    unfold pubLeaf at h
    rw [ite_eq_right hn] at h
    omega
  obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hc.1
  have hpo : portOf (portCols c.unit.inGroups) (x % (c.unit.inWords * WORD)) = p := by
    rw [hpt]; exact portOf_eq hnd hp ht
  have hpl := hc.2
  rw [hpo] at hpl
  obtain ⟨-, h1, h2⟩ := leafPort C hpl
  rw [h2] at ht
  refine ⟨p, hpl, t, ht, by rw [hpt, h1], ?_, ?_⟩
  · unfold pubLeaf
    rw [ite_eq_left hc, hpo]
  · unfold srcKeyP
    rw [ite_eq_left h, hpo, hpt]
    congr 1
    omega

/-- A public leaf means public input ports (`Hidden.parse_noPorts`). -/
theorem ports_of_pub {c : Circuit} {pi : HmIn.Ports} (hp : HmIn.parse c = .ok pi) {x : ℕ} (h : 0 ≤ pubLeaf c pi x) :
    pi.ports.isEmpty = false := by
  cases he : pi.ports.isEmpty
  · rfl
  · rw [Aliased.pubLeaf_empty (Hidden.parse_noPorts hp he)] at h
    omega

theorem slot_div_mod {g upv d : ℕ} (hd : d < upv) : (g * upv + d) / upv = g ∧ (g * upv + d) % upv = d := by
  have hpos : 0 < upv := by omega
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hd, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hd]

/-- Leaf `jj` of unit slot `k` in the in regions (`Layout.inVal`) depends on the slot only through its template instance
and its public leaf. -/
theorem inVal_congr {c : Circuit} {pi : HmIn.Ports} {pub : Public} {b k k' jj jj' : ℕ}
    (hk : k < c.unitRange.count) (hk' : k' < c.unitRange.count) (hd : k / c.upv = k' / c.upv)
    (hq : (pi.leaves.getD (k % c.upv) #[]).getD jj (-1) = (pi.leaves.getD (k' % c.upv) #[]).getD jj' (-1)) :
    inVal c pi pub b k jj = inVal c pi pub b k' jj' := by
  simp only [inVal, hk, hk', ↓reduceIte, hd, hq]

section pubIn

variable {tags : Tags} {st : Stmt} {pi : HmIn.Ports} (L : Layout st) (Lf : Layout (Hidden.flatTwin st))
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))

theorem place_val {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) :
    (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x).val =
      st.c.unitRange.slot (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) +
        x.val % (st.c.unit.inWords * WORD) := by
  have hn := L.unitRows_nIn
  have hx' : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
  unfold stackPlace
  rw [dite_eq_left hx']
  show st.c.unitRange.slot (g * st.c.upv + x.val / (unitRows st L.unit_built).nIn) +
    x.val % (unitRows st L.unit_built).nIn = _
  rw [hn]

include L in
/-- **A stacked input off a public leaf is at a row `HmIn.delta` keeps**: a dropped row is a public leaf's bit
(`Hidden.publicCol_at`), which is the input's own port bit, so the input is at that public leaf. -/
theorem keep_of_neg (hp : HmIn.parse st.c = .ok pi) (C : CheckFacts st.c) {g x : ℕ} (hg : g < st.c.g)
    (hx : x < st.c.upv * (st.c.unit.inWords * WORD)) (hneg : pubLeaf st.c pi x < 0) :
    keepCol st.c pi (st.c.unitRange.slot (g * st.c.upv + x / (st.c.unit.inWords * WORD)) +
      x % (st.c.unit.inWords * WORD)) = true := by
  unfold keepCol
  cases he : pi.ports.isEmpty
  · rw [Bool.false_or]
    cases hq : HmIn.publicCol st.c pi (st.c.unitRange.slot (g * st.c.upv + x / (st.c.unit.inWords * WORD)) +
        x % (st.c.unit.inWords * WORD))
    · rfl
    · exfalso
      obtain ⟨k, -, jj, hjj, t, ht, hq1, hlt, hl⟩ := Hidden.publicCol_at hp he C L.unit_built hq
      have hpos : 0 < st.c.unit.inWords * WORD :=
        Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
      have hW : st.c.unit.inWords * WORD ≤ 2 ^ st.c.unitRange.slotLog := by
        rw [L.unit_slotLog]; exact (L.fit st.c.unitNet).2
      obtain ⟨e1, e2⟩ := slot_add_inj (lt_of_lt_of_le (Nat.mod_lt x hpos) hW) (lt_of_lt_of_le hlt hW) hq1
      have hd : x / (st.c.unit.inWords * WORD) < st.c.upv := (Nat.div_lt_iff_lt_mul hpos).2 hx
      rw [← e1, (slot_div_mod hd).2] at hl
      obtain ⟨hjs, hj1, hj2⟩ := leafPort C hjj
      have hmem : x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups := by
        rw [e2, ← hj1]; exact port_mem hjs (by rw [hj2]; exact ht)
      have hpo : portOf (portCols st.c.unit.inGroups) (x % (st.c.unit.inWords * WORD)) = jj := by
        rw [e2, ← hj1]; exact portOf_eq (portBits_nodup L.unit_built) hjs (by rw [hj2]; exact ht)
      have hpl : pubLeaf st.c pi x = (pi.leaves.getD (x / (st.c.unit.inWords * WORD)) #[]).getD jj (-1) := by
        unfold pubLeaf
        rw [ite_eq_left ⟨hmem, by rw [hpo]; exact hjj⟩, hpo]
      omega
  · rfl

/-- **A stacked unit input off a public leaf carries what its copy position carries** at public inputs: its row is kept
(`keep_of_neg`), so it reads the flat twin's (`UnitRowsFlatP`, `copyAt_blockR`). -/
theorem copyPos_blockHP (hp : HmIn.parse st.c = .ok pi) (C : CheckFacts st.c) (hR : UnitRowsFlatP st pi) {g : ℕ}
    (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hneg : pubLeaf st.c pi x.val < 0)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x) =
      (L.model regs).block z o (Aliased.copyPos Lf regs F hg x.val) := by
  have hn := L.unitRows_nIn
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← hn]; exact hx
  have hpos : 0 < st.c.unit.inWords * WORD :=
    Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx''; omega
  have hplace := place_val L regs hg x hx
  have hs := Aliased.stack_copy_lt Lf hg hx''
  have hm := Nat.mod_lt x.val hpos
  unfold Aliased.copyPos
  rw [dite_eq_left hx'']
  exact copyAt_blockR L Lf regs F hs hm _ hplace
    (fun j => hR _ hs _ (lt_of_lt_of_le hm (L.fit st.c.unitNet).2) _ j hplace
      (by rw [hplace]; exact keep_of_neg L hp C hg hx'' hneg)) hz o

/-- **A stacked input at a public leaf carries bit `t` of its leaf's word in the in regions** (`Layout.in_cover`,
`Layout.in_pinned`): leaf `p` of its unit slot, at its key's bit `t`. -/
theorem pub_block (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {ri : Array Region} (hRi : HmIn.regions st.c pi st.pub = .ok ri) (hri : ∀ r ∈ ri.toList, r ∈ st.regions.toList)
    (hne : pi.ports.isEmpty = false) (C : CheckFacts st.c) {g : ℕ} (hg : g < st.c.g)
    (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hpub : 0 ≤ pubLeaf st.c pi x.val)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    ∃ p t, srcKeyP st.c pi x.val = .pub (pubLeaf st.c pi x.val).toNat t ∧
      (pi.leaves.getD (x.val / (st.c.unit.inWords * WORD)) #[]).getD p (-1) = pubLeaf st.c pi x.val ∧
      bit ((L.model regs).block z o (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x)) =
        (inVal st.c pi st.pub o.val (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) p).testBit t := by
  have hn := L.unitRows_nIn
  have hx'' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← hn]; exact hx
  have hk := Aliased.stack_copy_lt L hg hx''
  obtain ⟨p, -, t, ht, hxm, hq, hkey⟩ := pubLeaf_at C (portBits_nodup L.unit_built) hpub
  have hplace := place_val L regs hg x hx
  have haddr : inAddr st.c (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) p t =
      (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x).val := by
    have hslot : st.c.unitRange.slot (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) =
        st.c.unitRange.start + 2 ^ st.c.unitRange.slotLog * (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) := by
      unfold Range.slot Range.start; ring
    rw [hplace, hslot, hxm, inAddr, inCol]
    ring
  have hfin : inBlockOr st.c.kLog (inAddr st.c (g * st.c.upv + x.val / (st.c.unit.inWords * WORD)) p t) =
      stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
        (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x := by
    unfold inBlockOr
    rw [dite_eq_left (by rw [haddr]; exact (stackPlace (L.model regs) (unitRows st L.unit_built) st.c.upv
      (fun u => st.c.unitRange.slot (g * st.c.upv + u)) (L.hb hg) x).isLt)]
    exact Fin.ext haddr
  obtain ⟨xr, w, s, hxr, hws, hcol, hlo, hhi⟩ :=
    in_cover st.c pi (u := x.val / (st.c.unit.inWords * WORD)) (jj := p) (by rw [← hq]; exact hpub)
  have hb := in_pinned L regs hregs hRi hri hne hz o hxr hws hcol hlo hhi hk ht
  rw [hfin] at hb
  exact ⟨p, t, hkey, hq.symm, hb⟩

include Lf F in
/-- **An instance that puts inputs on one gate only when they have one key with public inputs is aliased**
(`Layout.Aliased`) in the hidden statement at public inputs: inputs at public leaves with one key carry one bit of one
public word of one template instance (`pub_block`), and the others carry what their copy positions in the flat twin
carry (`copyPos_blockHP`, `Aliased.copyPos_of_key`). -/
theorem aliased_of_keyP (hp : HmIn.parse st.c = .ok pi) (C : CheckFacts st.c) (hR : UnitRowsFlatP st pi)
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {ri : Array Region} (hRi : HmIn.regions st.c pi st.pub = .ok ri) (hri : ∀ r ∈ ri.toList, r ∈ st.regions.toList)
    {Cc : Audit.Circuit} {n : ℕ} {P : Audit.Partition Cc n} {u : Fin n}
    (inst : P.IsRowsUnit u ((unitRows st L.unit_built).stack st.c.upv)) {g : ℕ} (hg : g < st.c.g)
    (hk : ∀ c c', inst.wire c = inst.wire c' → c.val < ((unitRows st L.unit_built).stack st.c.upv).nIn →
      c'.val < ((unitRows st L.unit_built).stack st.c.upv).nIn → srcKeyP st.c pi c.val = srcKeyP st.c pi c'.val)
    (o : Fin (2 ^ (L.model regs).nbl)) : L.Aliased regs inst o hg := by
  intro c c' hw hc hc' z hz
  have hkey := hk c c' hw hc hc'
  have hn := L.unitRows_nIn
  have hx1 : c.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← hn]; exact hc
  have hx2 : c'.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← hn]; exact hc'
  by_cases h1 : 0 ≤ pubLeaf st.c pi c.val
  · have h2 : 0 ≤ pubLeaf st.c pi c'.val := by
      by_contra h2
      obtain ⟨t, ht⟩ := Aliased.srcKeyP_of_nonneg h1
      rw [ht, Aliased.srcKeyP_of_neg (by omega)] at hkey
      cases hkey
    have hne := ports_of_pub hp h1
    obtain ⟨p, t, hk1, hq1, hb1⟩ := pub_block L regs hregs hRi hri hne C hg c hc h1 hz o
    obtain ⟨p', t', hk2, hq2, hb2⟩ := pub_block L regs hregs hRi hri hne C hg c' hc' h2 hz o
    rw [hk1, hk2, Aliased.KeyP.pub.injEq] at hkey
    obtain ⟨hq, rfl⟩ := hkey
    have hq' : pubLeaf st.c pi c.val = pubLeaf st.c pi c'.val := by
      rw [← Int.toNat_of_nonneg h1, ← Int.toNat_of_nonneg h2, hq]
    have hpos : 0 < st.c.unit.inWords * WORD :=
      Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx1; omega
    have e1 := slot_div_mod (g := g) ((Nat.div_lt_iff_lt_mul hpos).2 hx1)
    have e2 := slot_div_mod (g := g) ((Nat.div_lt_iff_lt_mul hpos).2 hx2)
    have hv : inVal st.c pi st.pub o.val (g * st.c.upv + c.val / (st.c.unit.inWords * WORD)) p =
        inVal st.c pi st.pub o.val (g * st.c.upv + c'.val / (st.c.unit.inWords * WORD)) p' :=
      inVal_congr (Aliased.stack_copy_lt L hg hx1) (Aliased.stack_copy_lt L hg hx2) (by rw [e1.1, e2.1])
        (by rw [e1.2, e2.2, hq1, hq2, hq'])
    have bi : ∀ a b : ZMod 2, bit a = bit b → a = b := by decide
    exact bi _ _ (by rw [hb1, hb2, hv])
  · have h2 : pubLeaf st.c pi c'.val < 0 := by
      by_contra h2
      obtain ⟨t, ht⟩ := Aliased.srcKeyP_of_nonneg (show 0 ≤ pubLeaf st.c pi c'.val by omega)
      rw [ht, Aliased.srcKeyP_of_neg (by omega)] at hkey
      cases hkey
    rw [copyPos_blockHP L Lf regs F hp C hR hg c hc (by omega) hz o,
      copyPos_blockHP L Lf regs F hp C hR hg c' hc' h2 hz o]
    rw [Aliased.srcKeyP_of_neg (by omega), Aliased.srcKeyP_of_neg h2, Aliased.KeyP.src.injEq] at hkey
    rcases Aliased.copyPos_of_key Lf regs F hg hx1 hx2 hkey with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
    · rw [he]
    · rw [he, he', zeroPos_blockH L Lf regs F hs hz o, zeroPos_blockH L Lf regs F hs' hz o]

end pubIn

end FlockSoundness.Discharge.HiddenExec
