import Proofs.Flock.Soundness.Discharge.Hidden.Layout

/-!
# A hidden-output statement's flat twin, and its output rows' message bits

`flatTwin st` is the statement with `HmRow.delta`'s Δ alone, the one `Stmt.setupH` would build on the same circuit; it
has the unit facts `Hm.row_bc_flat` reads (`unitFacts_of_parse`). An output row's message bit (`deltaMsgCol`) is an input
bit of a `sha512x3` slot past the compression's `cv` (`deltaMsgCol_at`), where no copy of `HmRow.delta` is (`msg_filter`),
so its flat Δ count is zero (`flat_count_msg`), and `outPairs` copies into it once (`outPairs_count`): the hidden
statement's row there is a copy of `outSrc` (`msg_copy`), its flat twin's a self row (`flat_self`). So a satisfying
witness of the hidden statement satisfies its flat twin (`sat_flat`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- **The flat twin**: the statement with `HmRow.delta`'s Δ for its own. -/
abbrev flatTwin (st : Stmt) : Stmt :=
  { st with da := (HmRow.delta st.c st.pin).1, db := (HmRow.delta st.c st.pin).2 }

/-- **What the copy positions read**, for a statement whose circuit `HmRow.parse` accepted and whose Δ is
`HmRow.delta`'s. -/
theorem unitFacts_of_parse {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {st : Stmt}
    (hp : HmRow.parse tags cf tables = .ok st.c) (hda : st.da = (HmRow.delta st.c st.pin).1) :
    Aliased.UnitFacts tags st := by
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
    · exact Aliased.parse_rows hl
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

/-! ## An output row's message bits -/

/-- The offset of byte `b`'s bit `t` in a compression's 1024 message bits. -/
def msgOff (b t : ℕ) : ℕ := 64 * (b % 128 / 8) + 56 - 8 * (b % 128 % 8) + t

theorem msgOff_lt {b t : ℕ} (ht : t < 8) : msgOff b t < 1024 := by
  unfold msgOff; omega

theorem deltaMsgCol_eq (c : Circuit) (g p b t : ℕ) (hs : (c.ports.getD p default).segs = none)
    (hp : p < c.ports.size) (hk : b / 128 < c.ports[p]!.chunks) :
    deltaMsgCol c g p b t = shaBase c g p (b / 128) + shaIn c (2 * shaSub c p (b / 128) + 1) + msgOff b t := by
  have hk' : (HmRow.byteAt (c.ports.getD p default) b).1 < ((HmRow.rowSlots c.ports).2[p]!).size := by
    rw [Typed.byteAt_none hs, rowSlots_size hp]
    exact hk
  simp only [deltaMsgCol, hk', ↓reduceIte]
  simp only [Typed.byteAt_none hs, shaBase, shaIn, shaSub, msgOff]
  omega

theorem segs_getD {c : Circuit} (hu : ∀ p ∈ c.ports.toList, p.segs = none) {p : ℕ} (hp : p < c.ports.size) :
    (c.ports.getD p default).segs = none := by
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hp, Option.getD_some]
  exact hu _ (Array.getElem_mem_toList hp)

section Cols

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

/-- **Row `p`'s message bit `t` of byte `b`** is at its compression `b / 128`'s `sha512x3` slot, past the `cv`, below
the net's self rows. -/
theorem deltaMsgCol_at {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hb : b < 128 * c.ports[p]!.chunks)
    (ht : t < 8) :
    AtSlot c c.shaNet (g * c.perVu.getD c.shaNet 0 + ((HmRow.rowSlots c.ports).2[p]!)[b / 128]!.1)
      (1536 * shaSub c p (b / 128) + 512 + msgOff b t) (deltaMsgCol c g p b t) ∧
    1536 * shaSub c p (b / 128) + 512 + msgOff b t < 4608 := by
  have hk : b / 128 < c.ports[p]!.chunks := by omega
  have hs := rowSlot_lt hp hk
  have hmo := msgOff_lt (b := b) ht
  have hsub : shaSub c p (b / 128) < 3 := hs.2
  refine ⟨⟨R.slot_lt_count (Or.inl rfl) hg (by rw [R.perVu_sha]; exact hs.1),
    R.net_off_lt (Or.inl rfl) (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; omega), ?_⟩, by omega⟩
  rw [deltaMsgCol_eq _ _ _ _ _ (segs_getD hu hp) hp hk, (shaIn_eq R.rowNets _ hsub).2, shaBase, Circuit.slotCol]
  omega

theorem deltaMsgCol_lt {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hb : b < 128 * c.ports[p]!.chunks)
    (ht : t < 8) : deltaMsgCol c g p b t < 2 ^ c.kLog :=
  (deltaMsgCol_at R hu hg hp hb ht).1.lt R.bound (R.range R.shaNet).1

/-- A compression's `cv` bit is no row's message bit. -/
theorem cv_ne_msg {g' p' k' t' g p b t : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hk' : k' < c.ports[p']!.chunks) (ht' : t' < 512) (hg : g < c.g) (hp : p < c.ports.size)
    (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    shaBase c g' p' k' + shaIn c (2 * shaSub c p' k') + t' ≠ deltaMsgCol c g p b t := by
  intro h
  have hn := R.range R.shaNet
  obtain ⟨-, -, ho⟩ := (R.shaIn_at hg' hp' hk' ht').inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact (deltaMsgCol_at R hu hg hp hb ht).1)
  have hmo := msgOff_lt (b := b) ht
  omega

/-- An `hm96` slot's input bit is no row's message bit. -/
theorem hm_ne_msg {g' p' off' g p b t : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (ho' : off' < 4096)
    (hg : g < c.g) (hp : p < c.ports.size) (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    c.slotCol c.hmNet g' p' + off' ≠ deltaMsgCol c g p b t := by
  intro h
  have hn := R.range R.shaNet
  have hm := R.range R.hmNet
  exact sha_ne_hm R.rowNets ((deltaMsgCol_at R hu hg hp hb ht).1.inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [← h]; exact R.hmIn_at hg' hp' ho')).1

/-- **Distinct message bits are distinct columns**: template instance, row, byte and bit. -/
theorem deltaMsgCol_inj {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hb' : b' < 128 * c.ports[p']!.chunks) (ht' : t' < 8) (h : deltaMsgCol c g p b t = deltaMsgCol c g' p' b' t') :
    g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  have hn := R.range R.shaNet
  have hk : b / 128 < c.ports[p]!.chunks := by omega
  have hk' : b' / 128 < c.ports[p']!.chunks := by omega
  obtain ⟨-, hs, ho⟩ := (deltaMsgCol_at R hu hg hp hb ht).1.inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact (deltaMsgCol_at R hu hg' hp' hb' ht').1)
  have h1 := rowSlot_lt hp hk
  have h2 := rowSlot_lt hp' hk'
  obtain ⟨rfl, hs⟩ := slot_inj (by rw [R.perVu_sha]; exact h1.1) (by rw [R.perVu_sha]; exact h2.1) hs
  have hmo := msgOff_lt (b := b) ht
  have hu' := msgOff_lt (b := b') ht'
  unfold shaSub at ho
  rw [rowSlots_entry hp hk, rowSlots_entry hp' hk'] at hs ho
  simp only at hs ho
  have : atOf c.ports p + b / 128 = atOf c.ports p' + b' / 128 := by omega
  obtain ⟨rfl, hkk⟩ := at_inj hp hp' hk hk' this
  unfold msgOff at ho
  refine ⟨rfl, rfl, ?_, ?_⟩ <;> omega

/-- **No copy of `HmRow.delta`'s rows or of the rows' wires is at a row's message bit.** -/
theorem msg_filter (hpos : ∀ q < c.ports.size, 0 < c.ports[q]!.chunks) {pin g p b t : ℕ} (hg : g < c.g)
    (hp : p < c.ports.size) (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    (rowCopies c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  have hsz := rowSlots_size' c.ports
  have hw := filter_flatMap2_none _ _ (fun g p => wireF c g p) (deltaMsgCol c g p b t) fun g' hg' p' hp' e he => by
    have hp'' : p' < c.ports.size := hsz ▸ hp'
    rcases wireF_mem he with ⟨k', -, hk2, t', ht', he1⟩ | ⟨t', ht', he1⟩
    · rw [rowSlots_size hp''] at hk2
      have hs := (rowSlot_lt hp'' hk2).2
      rw [chainWidth_eq R.rowNets hs] at ht'
      rw [he1, chainDst_eq R.rowNets hs]
      exact cv_ne_msg R hu hg' hp'' hk2 ht' hg hp hb ht
    · rw [hmWidth_eq R.rowNets] at ht'
      rw [he1]
      show hmWireDst c g' p' + t' ≠ _
      rw [hmWireDst_eq R.rowNets]
      exact hm_ne_msg R hu hg' hp'' (by omega) hg hp hb ht
  have hr := filter_flatMap2_none _ _ (fun g' p' =>
      (List.range 512).map (fun t => (cvCol c g' p' + t, constSrc c pin (HmRow.midstate c.ports[p']!.words) t)) ++
      (List.range 1024).map fun t => (padCol c g' p' + t, constSrc c pin (HmRow.padWords c.ports[p']!.words) t))
    (deltaMsgCol c g p b t) fun g' hg' p' hp' e he => by
      have hp'' : p' < c.ports.size := hsz ▸ hp'
      rcases rowF_mem he with ⟨t', ht', he1⟩ | ⟨t', ht', he1⟩
      · rw [he1]
        exact cv_ne_msg R hu (k' := 0) hg' hp'' (hpos p' hp'') ht' hg hp hb ht
      · rw [he1, padCol_eq R.rowNets, Nat.add_assoc]
        exact hm_ne_msg R hu hg' hp'' (by omega) hg hp hb ht
  rw [List.filter_append, wireCopiesOf_rows _ hpos (fun q hq => hu _ (getElem_mem_toList' hq)), hw, List.append_nil]
  unfold rowCopies
  exact hr

end Cols

/-! ## `HmRow.delta`'s Δ at a message bit: nothing -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`HmRow.delta`'s pairs at a row's message bit**: none. The slot constants, the leaf and cut bits and the wires past
the rows' are elsewhere (`Hm.flatPairs_countL`'s), and so are the rows' copies and wires (`msg_filter`). -/
theorem flat_count_msg (hrw : RowWires st.c) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) (hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks)
    {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8)
    (j : ℕ) : (flatPairs st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  have P := F.parse
  have C := F.check
  have hf := msg_filter R hu hpos (pin := st.pin) hg hp hb ht
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hb ht
  generalize deltaMsgCol st.c g p b t = x at hat hf ⊢
  have hnn := R.shaNet
  obtain ⟨hm, hk, -, -⟩ := R.range hnn
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok (shaNet_ofRows R.rowNets)
  have hoff' : 1536 * shaSub st.c p (b / 128) + 512 + msgOff b t <
      (st.c.nets.getD st.c.shaNet default).2.inWords * Flock.WORD := by
    rw [shaNet_inRows R.rowNets]; omega
  obtain ⟨hum, huk, huc⟩ := unit_range_facts P C
  have hunit : ∀ s₀ b, s₀ < st.c.unitRange.count → b < 2 ^ st.c.unitRange.slotLog →
      st.c.unitRange.slot s₀ + b ≠ x := fun s₀ b hs₀ hb he =>
    atSlot_ne L F hnn P.unitNet F.sha hat (show AtSlot st.c st.c.unitNet s₀ b x from ⟨hs₀, hb, he.symm⟩)
  have hC : ∀ j, (constPairs st.c st.pin).count (x, j) = 0 := fun j => by
    rw [constPairs_count R.disj R.slotLog R.fit hm hk hat j, ite_eq_right (fun h => by have := h.1; omega)]
  have hleaf : ∀ j, (Flat.leafPairs st.c).count (x, j) = 0 := by
    intro j
    refine List.count_eq_zero.2 fun hmem => ?_
    simp only [Flat.leafPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨k, hk', jj, hjj, tt, htt, he⟩ := hmem
    have hx : Flat.leafCol st.c k jj tt = x := by rcases he with he | he <;> exact (Prod.mk.inj he).1.symm
    have hupv : 0 < st.c.upv := by
      rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
      · rw [L.unit_count, h0, Nat.mul_zero] at hk'; omega
      · exact h0
    have hmod := Nat.mod_lt k hupv
    have hlt : k % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact hmod
    have hsz : (st.c.leavesIn.getD (k % st.c.upv) #[]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
        ← getElem!_pos st.c.leavesIn _ hlt]
      exact C.leaves.2 _ hmod
    rw [hsz] at hjj
    obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjj
    have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
      C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hjj)
    have htw : tt < (portCols st.c.unit.inGroups)[jj]!.2 := by rw [hw, h16]; exact htt
    refine hunit k _ hk' (L.port_lt_slot hjs htw) ?_
    rw [← hx]
    unfold Flat.leafCol
    rw [hcol, Nat.add_assoc]
  have hcut : ∀ j, (Flat.cutPairs st.c).count (x, j) = 0 := by
    intro j
    refine List.count_eq_zero.2 fun hmem => ?_
    simp only [Flat.cutPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨g, hg, e, he, tt, htt, hpe⟩ := hmem
    have hx : Flat.cutCol st.c g e + tt = x := by rcases hpe with h | h <;> exact (Prod.mk.inj h).1.symm
    obtain ⟨he1, he2⟩ := C.cuts e he
    have hpv := perVu_unit C hum huk huc (by omega)
    refine hunit (g * st.c.upv + e.1) _ (by rw [huc]; exact Rows.mul_add_lt hg he1) (L.port_lt_slot he2 htt) ?_
    rw [← hx]
    unfold Flat.cutCol Circuit.slotCol Circuit.unitRange
    rw [hpv, Nat.add_assoc]
  have hwire : (wireCopies st.c).filter (fun e => e.1 = x) =
      (wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter (fun e => e.1 = x) := by
    obtain ⟨rest, hws, hrest⟩ := hrw
    unfold wireCopies wireCopiesOf
    rw [hws, List.filter_flatMap, List.filter_flatMap]
    refine List.flatMap_congr fun g hg => ?_
    rw [List.flatMap_append, List.filter_append]
    refine (congrArg (HAppend.hAppend _) (filter_nil_of _ x ?_)).trans (List.append_nil _)
    intro e hmem
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hmem
    obtain ⟨w, hw, tt, htt, rfl⟩ := hmem
    obtain ⟨hd1, hd2, hd3⟩ := C.wires w (by rw [hws]; exact List.mem_append_right _ hw)
    have hne : w.dst.1 ≠ st.c.shaNet := (hrest w hw).1
    obtain ⟨hm', hk', -⟩ := rangeOf_spec (C.one_range _ hd1)
    have hq := getD_mem_toList hd3 (d := default)
    have hle := (nets_ok P w.dst.1).1.2.2 _ hq
    have hfit := (L.fit w.dst.1).2
    intro he
    refine atSlot_ne L F hnn hd1 hne hat (s' := g * st.c.perVu.getD w.dst.1 0 + w.dst.2.1)
      (off' := ((portCols (st.c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1 + tt)
      ⟨by rw [C.count _ hm' _ hk']; exact Rows.mul_add_lt (List.mem_range.1 hg) hd2,
        by rw [L.slotLog _ hm' _ hk']; unfold Flat.wireWidth at htt; omega, ?_⟩
    rw [← he]
    unfold Flat.wireDst Circuit.slotCol
    rw [Nat.add_assoc]
  have hf' : (rowCopies st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) = [] := by
    rw [List.filter_append, hwire, ← List.filter_append]
    exact hf
  simp only [flatPairs, rowPairs_eq, wirePairs_eq, List.count_append, hC, hleaf, hcut, Nat.zero_add, Nat.add_zero]
  rw [← List.count_append, ← List.flatMap_append, count_pair2, hf']
  rfl

/-- **The flat Δ's row at a row's message bit is a self row**, in `A` and in `B`. -/
theorem flat_self (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
    (hda : st.da = (HmRow.delta st.c st.pin).1) {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  have hu : ∀ p ∈ st.c.ports.toList, p.segs = none := fun p hp => F.parse.v1Segs p hp (hv1 p hp).1
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hb ht
  have hfc := flat_count_msg L F F.parse.rowWires hu hpos hg hp hb ht
  have hδ : ∀ j : ℕ, deltaEntry st.da (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) ∧
      deltaEntry st.db (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) := by
    intro j
    have e : deltaEntry st.da (deltaMsgCol st.c g p b t) j =
        (((if j = deltaMsgCol st.c g p b t then 1 else 0) + (if j = deltaMsgCol st.c g p b t then 1 else 0) : ℕ) :
          ZMod 2) := by
      rw [deltaEntry_eq_count, hda, delta_flat_eq _ _ F.parse.typed hv1 F.parse.portWords F.parse.v1Segs, hfc]
      split_ifs <;> rfl
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact inBit_rows R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ j

end Twin

/-! ## `outPairs` at a message bit: one copy -/

/-- `outPairs`' copies: each output row's message bit, from the output bit the circuit computes. -/
def outCopies (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).map fun t => (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)

theorem outPairs_eq {c : Circuit} (h : c.outRowPorts.size = c.outPorts.size) :
    outPairs c = (outCopies c).flatMap pair2 := by
  simp only [outPairs, outCopies, h, List.flatMap_assoc, List.flatMap_map, pair2]

/-- Unsegmented rows, one output row per output port, each after the input rows and of `2 · words = 128 · chunks`
bytes (`out_rows_v1`). -/
def OutRows (c : Circuit) : Prop :=
  (∀ p ∈ c.ports.toList, p.segs = none) ∧ c.outRowPorts = Array.replicate c.outPorts.size 1 ∧
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks

theorem OutRows.size {c : Circuit} (h : OutRows c) : c.outRowPorts.size = c.outPorts.size := by
  rw [h.2.1, Array.size_replicate]

theorem outRow_bound {c : Circuit} (hrows : OutRows c) {j k t : ℕ} (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    HmOut.nIn c + j < c.ports.size ∧ 2 * k + t / 8 < 128 * c.ports[HmOut.nIn c + j]!.chunks ∧ t % 8 < 8 := by
  obtain ⟨hp, hw⟩ := hrows.2.2 j hj
  rw [getElem!_eq_getD']
  exact ⟨hp, by omega, by omega⟩

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRows c)
include R hrows

/-- **`outPairs`' one copy at output row `j`'s message bit for word `k`, bit `t`**: from `outSrc`. -/
theorem outCopies_filter {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    (outCopies c).filter (fun e => e.1 = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
      [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)] := by
  have hdec : ∀ g' < c.g, ∀ j' < c.outPorts.size, ∀ k' < (c.ports.getD (HmOut.nIn c + j') default).words,
      ∀ t' < 16, deltaMsgCol c g' (HmOut.nIn c + j') (2 * k' + t' / 8) (t' % 8) =
        deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) → g' = g ∧ j' = j ∧ k' = k ∧ t' = t := by
    intro g' hg' j' hj' k' hk' t' ht' he
    obtain ⟨hp', hb', ht''⟩ := outRow_bound hrows hj' hk' ht'
    obtain ⟨hp, hb, ht'''⟩ := outRow_bound hrows hj hk ht
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_inj R hrows.1 hg' hp' hb' ht'' hg hp hb ht''' he
    exact ⟨h1, by omega, by omega, by omega⟩
  unfold outCopies
  rw [filter_flatMap2_single _ _ (fun g j => (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap
      fun k => (List.range 16).map fun t => (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t))
      _ g j hg hj fun g' hg' j' hj' hne e he => ?_]
  · rw [filter_flatMap_single _ List.nodup_range _ _ k (List.mem_range.2 hk) fun k' hk' hne => ?_]
    · rw [List.map_eq_flatMap, filter_flatMap_single _ List.nodup_range _ _ t (List.mem_range.2 ht)
        fun t' ht' hne => ?_]
      · simp
      · refine List.filter_eq_nil_iff.2 fun e he => ?_
        simp only [List.mem_cons, List.not_mem_nil, or_false] at he
        subst he
        simp only [decide_eq_true_eq]
        exact fun h => hne (hdec g hg j hj k hk t' (List.mem_range.1 ht') h).2.2.2
    · refine List.filter_eq_nil_iff.2 fun e he => ?_
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t', ht', rfl⟩ := he
      simp only [decide_eq_true_eq]
      exact fun h => hne (hdec g hg j hj k' (List.mem_range.1 hk') t' ht' h).2.2.1
  · simp only [List.mem_flatMap, List.mem_range, List.mem_map] at he
    obtain ⟨k', hk', t', ht', rfl⟩ := he
    exact fun h => hne ⟨(hdec g' hg' j' hj' k' hk' t' ht' h).1, (hdec g' hg' j' hj' k' hk' t' ht' h).2.1⟩

/-- **`outPairs`' pairs at output row `j`'s message bit for word `k`, bit `t`**: one copy from `outSrc`. -/
theorem outPairs_count {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) (i : ℕ) :
    (outPairs c).count (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), i) =
      (if i = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
        (if i = outSrc c g j k t then 1 else 0) := by
  rw [outPairs_eq hrows.size, count_pair2, outCopies_filter R hrows hg hj hk ht]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

end Out

/-! ## The hidden statement and its flat twin -/

/-- Δ's count at a row with no output pair is `HmRow.delta`'s. -/
theorem deltaEntry_append_out {d : Array (ℕ × ℕ)} {c : Circuit} {r : ℕ} (hr : ∀ q ∈ outPairs c, q.1 ≠ r) (j : ℕ) :
    deltaEntry (d ++ (outPairs c).toArray) r j = deltaEntry d r j := by
  rw [deltaEntry_eq_count, deltaEntry_eq_count, Array.toList_append, List.count_append, List.toList_toArray,
    List.count_eq_zero.2 fun hm => hr _ hm rfl, Nat.add_zero]

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRows st.c)
include L L0 F0 hv1 hda hrows

/-- **An output row's message bit holds its copy** in every block of a satisfying witness of the hidden statement: at
template instance `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copy {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  have hpos : ∀ q < st.c.ports.size, 0 < st.c.ports[q]!.chunks := fun q hq => (hv1 _ (getElem_mem_toList' hq)).2.2
  obtain ⟨hp, hb, ht8⟩ := outRow_bound hrows hj hk ht
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hrows.1 hg hp hb ht8
  have hx := deltaMsgCol_lt R hrows.1 hg hp hb ht8
  have hfc := flat_count_msg L0 F0 F0.parse.rowWires hrows.1 hpos hg hp hb ht8
  have hδ : ∀ i : ℕ, deltaEntry st.da (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) ∧
      deltaEntry st.db (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) := by
    intro i
    have e : deltaEntry st.da (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) i =
        (((if i = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
          (if i = outSrc st.c g j k t then 1 else 0) : ℕ) : ZMod 2) := by
      rw [deltaEntry_eq_count, hda, Array.toList_append, List.count_append, List.toList_toArray,
        delta_flat_eq _ _ F0.parse.typed hv1 F0.parse.portWords F0.parse.v1Segs, hfc, Nat.zero_add,
        outPairs_count R hrows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact block_of_rows hz o hx fun i =>
    inBit_rows (st := st) R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ i

/-- **A satisfying witness of the hidden statement satisfies its flat twin**: the rows agree past `outPairs`' message
bits (`deltaEntry_append_out`), and there the twin's row is a self row (`flat_self`), which every bit satisfies. -/
theorem sat_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i => ?_, hz.2.1, hz.2.2⟩
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    rw [hrows.size] at hj
    obtain ⟨hp, hb, ht8⟩ := outRow_bound hrows hj hk ht
    have hx := deltaMsgCol_lt R hrows.1 hg hp hb ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_self L0 F0 hv1 rfl hg hp hb ht8 hx i'
      have e : (if i'.val = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) then (1 : ZMod 2) else 0) =
          if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0 := by
        simp [Fin.ext_iff]
      exact ⟨hs.1.trans e, hs.2.trans e⟩
    rw [mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').1),
      mulVec_single (S := L0.model regs) _ (fun i' => (hrow i').2)]
    have hsq : ∀ a : ZMod 2, a * a = a := by decide
    exact hsq _
  · push Not at hi
    have hdb : st.db = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := L.delta_b.trans hda
    have hA : ∀ i', (L0.model regs).A₀ i i' = (L.model regs).A₀ i i' := fun i' => by
      show rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).1 i.val i'.val =
        rangesEntry st (slotEntryA st) i.val i'.val + deltaEntry st.da i.val i'.val
      rw [hda, deltaEntry_append_out hi]
    have hB : ∀ i', (L0.model regs).B₀ i i' = (L.model regs).B₀ i i' := fun i' => by
      show rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry (HmRow.delta st.c st.pin).2 i.val i'.val =
        rangesEntry st (slotEntryB st) i.val i'.val + deltaEntry st.db i.val i'.val
      rw [hdb, deltaEntry_append_out hi, delta_snd _ _ F0.parse.typed]
    have eA : (L0.model regs).A₀.mulVec ((L0.model regs).block z o) i =
        (L.model regs).A₀.mulVec ((L.model regs).block z o) i := by
      simp only [Matrix.mulVec, dotProduct, hA]
      rfl
    have eB : (L0.model regs).B₀.mulVec ((L0.model regs).block z o) i =
        (L.model regs).B₀.mulVec ((L.model regs).block z o) i := by
      simp only [Matrix.mulVec, dotProduct, hB]
      rfl
    rw [eA, eB]
    exact hz.1 o i

end Hidden

end FlockSoundness.Discharge.Hidden
