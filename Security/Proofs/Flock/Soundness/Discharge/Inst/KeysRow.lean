import Proofs.Flock.Soundness.Discharge.Inst.Order
import Proofs.Flock.Soundness.Discharge.Inst.Keys
import Proofs.Flock.Soundness.Discharge.Layout.KeysRow

/-!
# `InstKeysRow` from the parse

`instProg_valPinned` takes `InstKeysRow`: every stacked unit input no wire fills that a row of the template instance
reads has the zero or a row bit as its key. At the row widths `16 · words` (`portRowW`), in an untyped circuit whose wires
`wiresAcyclic` passed, it holds once every input a unit row reads is a port bit (`PortReads`), whatever wires reach the
unit (`instKeysRow_of_parse`, `keysRow_of_cuts` without `CutsFromRows`):

* a composite input a row reads is a stacked input no wire fills (`inCols`), which a computed row of a unit copy reads, or
  a wire-fed input bit's copy row, as the wire's source (`addSelf` cancels the bit's own entry);
* a computed unit row reads it where the unit's own row does (`Rows.ofNet_a`), so it is a port bit (`comp_read_port`);
* a wire's source bit at a unit input is a port bit: `wiresAcyclic` checked every source port's mask even (`outReads`),
  and an input column outside every port has mask `1` (`src_read_port`);
* a port bit no wire fills has the zero or a row bit as its key (`rowKey_of_port`): a leaf port's, or a cut port's, whose
  one source (`check_facts`' `wired`) is then a leaf cut.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Compose _root_.FlockSoundness.Discharge.Layout

theorem built_lookup {net : Net} (h : Built net) : net.lookup = none := by
  obtain ⟨_, _, _, _, _, _, _, ho⟩ := h
  exact (Net.ofRows_ok ho).2.2.2.2.2.2.2.2.2.2.2.2.1

/-- The unit is neither of the rows' circuits. -/
theorem unit_ne {c : Circuit} {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) : c.unitNet ≠ c.shaNet ∧ c.unitNet ≠ c.hmNet := by
  obtain ⟨-, -, hunit, hsn, hhn⟩ := parse_facts_pre (fun _ hx => by simp at hx) h
  have hname : (c.nets.getD c.unitNet default).1 = "unit" := by
    obtain ⟨hlt, hp', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hunit
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    simpa using hp'
  refine ⟨fun he => ?_, fun he => ?_⟩
  · rw [← he, hname] at hsn
    exact absurd hsn (by unfold HmRow.SHA_NET; decide)
  · rw [← he, hname] at hhn
    exact absurd hhn (by unfold HmRow.HM_NET; decide)

/-! ## What a row of the template instance reads -/

/-- **A computed unit row reads an input where the unit's own row does**, so at a port bit (`PortReads`). -/
theorem comp_read_port {c : Circuit} {hU : Built c.unit} (hpr : PortReads c hU) (hlog : c.unit.unitLog ≤ 32)
    {o j : ℕ} (hlo : c.unit.inWords * WORD ≤ o) (hhi : o < c.unit.constPos) (hj : j < c.unit.inWords * WORD)
    (he : entA c c.unitNet o j = 1 ∨ entB c c.unitNet o j = 1) : j ∈ portBits c.unit.inGroups := by
  obtain ⟨ra, rb, -, hra, hrb⟩ := built_rowOrder hU
  have hW : (Rows.ofNet hU).nIn = c.unit.inWords * WORD := ofNet_nIn_eq
  have hk : o - (Rows.ofNet hU).nIn < (Rows.ofNet hU).nComp := by
    show o - (Rows.ofNet hU).nIn < c.unit.constPos - min (c.unit.inWords * WORD) c.unit.constPos
    rw [hW, Nat.min_eq_left (by omega)]
    omega
  have hor : (Rows.ofNet hU).nIn + (o - (Rows.ofNet hU).nIn) = o := by rw [hW]; omega
  have hne : ∀ {m : Sparse}, sparseEntry m o j = 1 → sparseEntry m o j ≠ 0 := fun h h0 => by
    rw [h0] at h; exact absurd h (by decide)
  have hmem : j ∈ ((Rows.ofNet hU).a ⟨_, hk⟩).map Fin.val ++ ((Rows.ofNet hU).b ⟨_, hk⟩).map Fin.val := by
    rw [Rows.ofNet_a hU hlog, Rows.ofNet_b hU hlog]
    simp only [hor]
    rcases he with he | he
    · exact List.mem_append_left _ (entry_mem hra (hne he))
    · refine List.mem_append_right _ (entry_mem hrb (hne ?_))
      unfold entB at he
      rwa [show (netOf c c.unitNet).lookup = none from built_lookup hU, lookupEntry, add_zero] at he
  rw [← List.map_append] at hmem
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hmem
  exact hpr _ y hy (by rw [hW]; exact hj)

/-- **A wire's source bit at an input is a port bit**: `wiresAcyclic` checked the source net's output ports' masks even
(`outReads`), and an input column outside every port has mask `1`. -/
theorem src_read_port {net : Net} (hlk : net.lookup.isSome = false) (hok : (ProvedScope.outReads net).2 = true)
    (hic : net.inWords * WORD ≤ net.constPos) {k t : ℕ} (hk : k < (portCols net.outGroups).size)
    (ht : t < ((portCols net.outGroups).getD k default).2)
    (hlt : ((portCols net.outGroups).getD k default).1 + t < net.inWords * WORD) :
    ((portCols net.outGroups).getD k default).1 + t ∈ portBits net.inGroups := by
  rw [outReads_eq _ hlk] at hok
  simp only [Bool.and_eq_true, Array.all_eq_true', beq_iff_eq] at hok
  obtain ⟨q, hq⟩ : ∃ q, q = (portCols net.outGroups).getD k default := ⟨_, rfl⟩
  rw [← hq] at ht hlt ⊢
  have hqm : q ∈ portCols net.outGroups := by
    rw [hq, FlockSoundness.Layout.getD_portCols hk, getElem!_pos (portCols net.outGroups) k hk]
    exact Array.getElem_mem hk
  have hm2 : ((Array.range q.2).foldl (fun m x => mUnion m (mCol net (mRows net) (q.1 + x))) 0) % 2 = 0 :=
    hok.2 _ (Array.mem_map.2 ⟨q, hqm, rfl⟩)
  have hcol : (mCol net (mRows net) (q.1 + t)).testBit 0 = false := by
    cases hb : (mCol net (mRows net) (q.1 + t)).testBit 0
    · rfl
    · have h1 := mask_sub net q ht 0 hb
      rw [Nat.testBit_zero] at h1
      simp only [Array.size_range, decide_eq_true_eq] at h1 hm2
      omega
  have hN : mN net = net.inWords * WORD := Nat.min_eq_left hic
  unfold mCol at hcol
  rw [ite_eq_left (show q.1 + t < mN net by rw [hN]; exact hlt)] at hcol
  rcases mIn_ok net (q.1 + t) with h1 | ⟨p, t', hp, ht', hx, -⟩
  · rw [h1] at hcol
    exact absurd hcol (by decide)
  · rw [hx, FlockSoundness.Layout.getD_portCols hp]
    exact port_mem hp (by rwa [FlockSoundness.Layout.getD_portCols hp] at ht')

/-- **A composite input a row reads is a port bit of its unit copy**: it is a stacked input no wire fills, which a computed
row of a unit copy reads (`comp_read_port`), or a wire-fed input bit's copy row, as the wire's source (`src_read_port`). -/
theorem inst_read_port {c : Circuit} (hF : ∀ n < c.nets.size, 0 < c.perVu.getD n 0 → NetFit c n) (C : CheckFacts c)
    (hsha : c.unitNet ≠ c.shaNet) (hhm : c.unitNet ≠ c.hmNet) (hR : ReadsOk (c := c)) {hU : Built c.unit}
    (hpr : PortReads c hU) (i : Fin (instRows c).nComp) (x : Fin (instRows c).w)
    (hx : x ∈ (instRows c).a i ++ (instRows c).b i) (hxn : x.val < (instRows c).nIn) :
    (inX c).getD x.val 0 % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups := by
  have hxn' : x.val < (inCols c).length := hxn
  have hl : x.val ∈ (lrow (logical c) i.val).1 ++ (lrow (logical c) i.val).2 := by
    have ha : ((instRows c).a i).map Fin.val = (lrow (logical c) i.val).1 := Rows.compose_a i
    have hb : ((instRows c).b i).map Fin.val = (lrow (logical c) i.val).2 := Rows.compose_b i
    rcases List.mem_append.1 hx with h | h
    · exact List.mem_append_left _ (by rw [← ha]; exact List.mem_map_of_mem (f := Fin.val) h)
    · exact List.mem_append_right _ (by rw [← hb]; exact List.mem_map_of_mem (f := Fin.val) h)
  rcases hr : (row c ((order c).getD ((inCols c).length + i.val) 0)).getD none with _ | pr
  · rw [lrow_none (L := logical c) (k := i.val) hr] at hl
    simp only [List.mem_append, List.mem_singleton, or_self] at hl
    have hlen : (inCols c).length ≤ (order c).length := (inv_order c).1.length_le
    have : x.val = (order c).length := hl
    omega
  rw [lrow_some (L := logical c) (k := i.val) hr] at hl
  obtain ⟨y, hy, hxy⟩ : ∃ y ∈ pr.1 ++ pr.2, lidx (logical c) y = x.val := by
    simp only [List.mem_append, List.mem_map] at hl
    rcases hl with ⟨y, hy, e⟩ | ⟨y, hy, e⟩
    · exact ⟨y, List.mem_append_left _ hy, e⟩
    · exact ⟨y, List.mem_append_right _ hy, e⟩
  have hxy' : (order c).idxOf y = x.val := hxy
  have hlen : (inCols c).length ≤ (order c).length := (inv_order c).1.length_le
  have hyo : y ∈ order c := List.idxOf_lt_length_iff.1 (by omega)
  have hyg : (order c).getD x.val 0 = y := by
    rw [← hxy', List.getD_eq_getElem _ _ (List.idxOf_lt_length_iff.2 hyo)]
    exact List.getElem_idxOf (List.idxOf_lt_length_iff.2 hyo)
  obtain ⟨hic, hX⟩ := inCols_getD hxn'
  rw [order_getD_in hxn', hic] at hyg
  obtain ⟨X, hXd⟩ : ∃ X, X = (inX c).getD x.val 0 := ⟨_, rfl⟩
  rw [← hXd] at hyg hX ⊢
  show X % netIn c c.unitNet ∈ portBits (netOf c c.unitNet).inGroups
  have hW : 0 < netIn c c.unitNet := Nat.pos_of_ne_zero fun h0 => by
    have := inX_lt hX
    rw [h0, Nat.mul_zero] at this
    omega
  have hmod : X % netIn c c.unitNet < netIn c c.unitNet := Nat.mod_lt _ hW
  have hrow : rowAt c (codeNet ((order c).getD ((inCols c).length + i.val) 0))
      (codeSlot ((order c).getD ((inCols c).length + i.val) 0))
      (codeOff ((order c).getD ((inCols c).length + i.val) 0)) = some (some pr) := row_of_getD hr
  generalize codeNet ((order c).getD ((inCols c).length + i.val) 0) = n' at hrow
  generalize codeSlot ((order c).getD ((inCols c).length + i.val) 0) = i' at hrow
  generalize codeOff ((order c).getD ((inCols c).length + i.val) 0) = o' at hrow
  obtain ⟨hn', hsha', hhm', hi', -⟩ := rowAt_dom (c := c) (n := n') (i := i') (o := o') (by rw [hrow]; simp)
  have hF' := hF n' hn' (by omega)
  unfold inCode at hyg
  rcases (rowAt_comp hrow).2 with ⟨hlo, hhi, rfl⟩ | ⟨-, s, hf, rfl⟩
  · have key : ∀ {e : ℕ → ZMod 2}, y ∈ reads c n' i' e → n' = c.unitNet ∧ e (X % netIn c c.unitNet) = 1 := by
      intro e he
      unfold reads at he
      obtain ⟨j, hj, rfl⟩ := List.mem_map.1 he
      have h1 := congrArg codeNet hyg
      have h3 := congrArg codeOff hyg
      simp only [codeNet_code, codeOff_code] at h1 h3
      refine ⟨h1.symm, ?_⟩
      rw [h3]
      exact of_decide_eq_true (List.mem_filter.1 hj).2
    rcases List.mem_append.1 hy with h | h
    · obtain ⟨rfl, he⟩ := key h
      exact comp_read_port hpr hF'.log hlo hhi hmod (Or.inl he)
    · obtain ⟨rfl, he⟩ := key h
      exact comp_read_port hpr hF'.log hlo hhi hmod (Or.inr he)
  · obtain ⟨hob, w, hwa, rfl⟩ := feed_some hf
    rw [reads_addSelf fun j => (port_rows hF' hob j).1, reads_addSelf fun j => (port_rows hF' hob j).2] at hy
    simp only [List.nil_append, List.mem_append, List.mem_singleton, or_self] at hy
    subst hy
    have h1 := congrArg codeNet hyg
    have h3 := congrArg codeOff hyg
    simp only [codeNet_code, codeOff_code] at h1 h3
    obtain ⟨p', t', hp', ht', hpt'⟩ := Aliased.mem_portBits hob
    have hpo : portOf (portCols (netOf c n').inGroups) o' = p' := by
      rw [hpt']; exact portOf_eq hF'.nodup hp' ht'
    rw [hpo, hpt', Nat.add_sub_cancel_left] at h3
    unfold wireAt at hwa
    rw [hpo] at hwa
    obtain ⟨-, -, hw, hdst⟩ := wire_one C hn' hsha' hhm' hi' hp' hwa
    obtain ⟨hsn, hsi, hsp, hwid⟩ := C.wire_src w hw
    rw [hdst] at hwid
    rw [← h1] at hsn hsi hsp hwid h3
    have hFu := hF _ hsn (by omega)
    have hok : (ProvedScope.outReads (netOf c c.unitNet)).2 = true :=
      hR _ hsn (by rw [fSlots_of c hsha hhm]; omega)
    have ht'' : t' < ((portCols (netOf c c.unitNet).outGroups).getD w.src.2.2 default).2 := by
      rw [FlockSoundness.Layout.getD_portCols hsp, ← hwid]; exact ht'
    rw [h3]
    exact src_read_port (net := netOf c c.unitNet) C.unit_lookup hok hFu.inConst hsp ht'' (by rw [← h3]; exact hmod)

/-! ## A port bit's key -/

/-- **A port bit no wire fills has a row key**: a leaf port bit's key, or a cut port bit's, whose one source is then a
leaf cut (`check_facts`' `wired`), is the zero or a row bit at the row widths `16 · words`. -/
theorem rowKey_of_port {c : Circuit} {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (h : HmRow.parse tags bytes tables = .ok c) (hU : Built c.unit) {X : ℕ} (hX : X ∈ inX c)
    (hm : X % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups) : RowKey c (portRowW c) X := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hin := check_in P.typed hls
  obtain ⟨hcin, hg⟩ := check_cuts P.typed hls
  obtain ⟨hsha, hhm⟩ := unit_ne h
  obtain ⟨u, hdiv⟩ : ∃ u, X / (c.unit.inWords * WORD) = u := ⟨_, rfl⟩
  have hu : u < c.upv := by
    rw [← hdiv]
    exact Nat.div_lt_of_lt_mul (lt_of_lt_of_eq (inX_lt hX) (Nat.mul_comm _ _))
  obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits hm
  have hnd := Aliased.portBits_nodup hU
  have hkey : Aliased.srcKey c X = Aliased.srcAt c u p t := by
    unfold Aliased.srcKey
    rw [ite_eq_left hm, hdiv, hpt, portOf_eq hnd hp ht, Nat.add_sub_cancel_left]
  unfold RowKey
  rw [hkey]
  unfold Aliased.srcAt
  by_cases hp' : p < (c.unit.inGroups.getD 0 default).bits.size
  swap
  · rw [ite_eq_right hp']
    have hpv : c.perVu.getD c.unitNet 0 = c.upv := by
      obtain ⟨hur, hk, hcount⟩ := unit_range_facts P C
      exact perVu_unit C hur hk hcount hg
    have hone := C.wired c.unitNet P.unitNet hsha hhm u (by rw [hpv]; exact hu) p
      (by rw [ite_eq_left rfl]; omega) hp
    have hw0 : (c.wires.toList.filter fun w => decide (w.dst = (c.unitNet, u, p))) = [] := by
      have hf : feed c c.unitNet (X / netIn c c.unitNet) (X % netIn c c.unitNet) = none :=
        Option.isNone_iff_eq_none.1 (List.mem_filter.1 hX).2
      unfold feed at hf
      rw [ite_eq_left (show X % netIn c c.unitNet ∈ portBits (netOf c c.unitNet).inGroups from hm)] at hf
      have hwa := Option.map_eq_none_iff.1 hf
      unfold wireAt at hwa
      rw [show X / netIn c c.unitNet = u from hdiv,
        show X % netIn c c.unitNet = (portCols (netOf c c.unitNet).inGroups)[p]!.1 + t from hpt,
        portOf_eq (gs := (netOf c c.unitNet).inGroups) hnd hp ht] at hwa
      exact List.head?_eq_none_iff.1 hwa
    have hcf : (c.leafCuts.toList.filter fun e => decide ((c.unitNet, e.1, e.2.1) = (c.unitNet, u, p))) =
        c.leafCuts.toList.filter fun e => decide (e.1 = u ∧ e.2.1 = p) :=
      List.filter_congr fun e _ => by simp
    rw [hw0, List.length_nil, Nat.zero_add, hcf] at hone
    obtain ⟨e, he⟩ := List.length_eq_one_iff.1 hone
    have hem : e ∈ c.leafCuts.toList :=
      List.mem_of_mem_filter (p := fun e => decide (e.1 = u ∧ e.2.1 = p)) (by rw [he]; exact List.mem_singleton_self e)
    simp only [he]
    split
    · rename_i hcond
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
      obtain ⟨ht16, -⟩ := hcond
      refine Or.inr ⟨_, _, _, rfl, ?_⟩
      obtain ⟨hq, hs1, hs2⟩ := starts_find c (hcin e hem) rfl
      refine ⟨hq, ?_, Nat.mod_lt _ (by decide)⟩
      unfold portRowW
      simp only [Fin.getElem_fin]
      omega
    · exact Or.inl rfl
  have ht16 : t < 16 := by
    obtain ⟨-, -, hw⟩ := firstGroup_port C.unit_groups.1 hp'
    rw [hw, C.unit_groups.2 _ (getElem_mem_toList' hp')] at ht
    exact ht
  rw [ite_eq_left hp']
  split
  · exact Or.inl rfl
  · rename_i hneg
    dsimp only
    split
    swap
    · exact Or.inl rfl
    refine Or.inr ⟨_, _, _, rfl, ?_⟩
    have hget : c.leavesIn.getD u #[] = c.leavesIn[u]! := by
      rw [Array.getD_eq_getD_getElem?, getElem!_pos c.leavesIn u (by rw [C.leaves.1]; exact hu),
        Array.getElem?_eq_getElem (by rw [C.leaves.1]; exact hu), Option.getD_some]
    rw [hget] at hneg ⊢
    have hpl : p < (c.leavesIn[u]!).size := by rw [C.leaves.2 u hu]; exact hp'
    have hlt := hin u hu ((c.leavesIn[u]!)[p]!) (by
      rw [getElem!_pos (c.leavesIn[u]!) p hpl]; exact Array.getElem_mem_toList _)
    have hxl : ((c.leavesIn[u]!)[p]!).toNat < c.inWords := by omega
    obtain ⟨hq, hs1, hs2⟩ := starts_find c hxl rfl
    refine ⟨hq, ?_, Nat.mod_lt _ (by decide)⟩
    unfold portRowW
    simp only [Fin.getElem_fin]
    omega

/-! ## `InstKeysRow` -/

/-- **`InstKeysRow` from the parse**: in an untyped circuit whose wires `wiresAcyclic` passed, at the row widths
`16 · words`, every stacked unit input no wire fills that a row of the template instance reads has the zero or a row bit
as its key, when every input a unit row reads is a port bit (`PortReads`). -/
theorem instKeysRow_of_parse {tags : Tags} {st : Stmt} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (L : FlockSoundness.Layout st) (h : HmRow.parse tags bytes tables = .ok st.c) {hU : Built st.c.unit}
    (hpr : PortReads st.c hU) (hw : ProvedScope.wiresAcyclic st.c = true) :
    InstKeysRow st.c (portRowW st.c) := by
  have P := parse_facts h
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  obtain ⟨-, hg⟩ := check_cuts P.typed hls
  obtain ⟨hsha, hhm⟩ := unit_ne h
  have hF : ∀ n < st.c.nets.size, 0 < st.c.perVu.getD n 0 → NetFit st.c n := fun n hn hi =>
    ⟨netLog_le L C hn hg hi, (net_const L P C hn hg hi).1, (L.fit n).1, (nets_ok P n).1.2.2,
      portBits_nodup_net P hn, P.nets _ (getD_mem_toList hn)⟩
  obtain ⟨hR, -⟩ := all_filled hF C (perVu_unit C L.unit_mem L.unit_kind L.unit_count hg) hw
  intro i x hx hxn
  obtain ⟨-, hX⟩ := inCols_getD (c := st.c) hxn
  exact rowKey_of_port h hU hX (inst_read_port hF C hsha hhm hR hpr i x hx hxn)

end FlockSoundness.Discharge.Inst
