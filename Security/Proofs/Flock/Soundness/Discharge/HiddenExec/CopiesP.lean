import Proofs.Flock.Soundness.Discharge.HiddenExec.Copies
import Proofs.Flock.Soundness.Discharge.Hidden.LayoutP
import Proofs.Flock.Soundness.Discharge.Layout.InCover
import Proofs.Flock.Soundness.Discharge.Aliased.KeysP

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

/-- **Port `p` of the unit's leaf group** is port `p` of the unit, 16 bits from column `col · WORD + 16 p`. -/
theorem leafPort {c : Circuit} (C : CheckFacts c) {p : ℕ} (hp : p < (c.unit.inGroups.getD 0 default).bits.size) :
    p < (portCols c.unit.inGroups).size ∧
      (portCols c.unit.inGroups)[p]!.1 = (c.unit.inGroups.getD 0 default).col * WORD + 16 * p ∧
      (portCols c.unit.inGroups)[p]!.2 = 16 := by
  obtain ⟨hps, h1, h2⟩ := firstGroup_port C.unit_groups.1 hp
  have e : ((portCols #[c.unit.inGroups.getD 0 default]).map (·.1)).getD p 0 =
      ((portCols #[c.unit.inGroups.getD 0 default]).getD p default).1 := by
    simp only [Array.getD_eq_getD_getElem?, Array.getElem?_map]
    cases (portCols #[c.unit.inGroups.getD 0 default])[p]? <;> rfl
  refine ⟨hps, by rw [← h1, e, portCols_16_get C.unit_groups.2 hp], ?_⟩
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

end FlockSoundness.Discharge.HiddenExec
