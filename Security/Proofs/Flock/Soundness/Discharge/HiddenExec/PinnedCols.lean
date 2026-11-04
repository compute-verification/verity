import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.PinnedLinear

set_option Elab.async false

/-!
# A pinned statement's pinned bits, where its unit returns the outputs

`Hidden.pinnedCols` lists bit `t` of each pinned output word `o` at each template instance `g`, `HmOut.outCol c g o + t`.
Where the unit returns the outputs (`outNet = unitNet`), every output word is some copy's (`check_returned_all`), so
`outCol c g o` is column `16 j` of copy `u`'s returned-output port in unit slot `g · upv + u`, `o` copy `u`'s output `j`
(`outCol_leaf`); a pinned word's 16 bits are below the unit's constant (`pinned_word`, `linearRow_spec`), so inside the
slot. Hence:

* `pinnedCols_nodup`: the pinned bits are distinct (`slot_add_inj`: the slot, so `g` and `u`, and the offset, so `j`
  and `t`), and each is in `pinnedCols` once;
* `outCol_mem_pinned`: an output word whose port has no output row (`outPortOf` past `outRowPorts`, at
  `Hidden.Any.OutRowsP`) is pinned, so its bits are in `pinnedCols`;
* `pinBit_outH`: a template instance's column on such a word's bit is a pinned bit (`PinBit`), which `SitesH.pin` holds
  at 0.
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 FlockSoundness.Discharge.Layout FlockSoundness.Discharge.Inst

theorem mul_add_inj_lt {P g g' u u' : ℕ} (hu : u < P) (hu' : u' < P) (h : g * P + u = g' * P + u') :
    g = g' ∧ u = u' := by
  have hP : 0 < P := by omega
  have h1 : (g * P + u) / P = g := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hu, Nat.zero_add]
  have h2 : (g' * P + u') / P = g' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hP, Nat.div_eq_of_lt hu', Nat.zero_add]
  have hg : g = g' := by rw [← h1, ← h2, h]
  subst hg
  exact ⟨rfl, by omega⟩

/-- A list of `f g o t` over a box is distinct when `f` is injective on the box. -/
theorem nodup_box {a b : ℕ} (f : ℕ → ℕ → ℕ → ℕ)
    (hf : ∀ g o t g' o' t', g < a → o < b → t < 16 → g' < a → o' < b → t' < 16 → f g o t = f g' o' t' →
      g = g' ∧ o = o' ∧ t = t') :
    ((List.range a).flatMap fun g => (List.range b).flatMap fun o => (List.range 16).map fun t => f g o t).Nodup := by
  refine List.nodup_flatMap.2 ⟨fun g hg => List.nodup_flatMap.2 ⟨fun o ho => ?_, ?_⟩, ?_⟩
  · refine (List.nodup_range).map_on fun t ht t' ht' h => ?_
    rw [List.mem_range] at hg ho ht ht'
    exact (hf g o t g o t' hg ho ht hg ho ht' h).2.2
  · refine List.Pairwise.imp_of_mem (fun {o o'} ho ho' hne => ?_) List.nodup_range
    intro x hx hx'
    simp only [List.mem_map, List.mem_range] at hx hx'
    rw [List.mem_range] at hg ho ho'
    obtain ⟨t, ht, rfl⟩ := hx
    obtain ⟨t', ht', he⟩ := hx'
    exact hne (hf g o' t' g o t hg ho' ht' hg ho ht he).2.1.symm
  · refine List.Pairwise.imp_of_mem (fun {g g'} hg hg' hne => ?_) List.nodup_range
    intro x hx hx'
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx hx'
    rw [List.mem_range] at hg hg'
    obtain ⟨o, ho, t, ht, rfl⟩ := hx
    obtain ⟨o', ho', t', ht', he⟩ := hx'
    exact hne (hf g' o' t' g o t hg' ho' ht' hg ho ht he).1.symm

/-! ## The pinned words -/

theorem split_sum (a : Array (String × ℕ)) (k : ℕ) :
    (a.extract 0 (a.size - k)).foldl (· + ·.2) 0 + (a.extract (a.size - k) a.size).foldl (· + ·.2) 0 =
      a.foldl (· + ·.2) 0 := by
  rw [← Array.foldl_toList, ← Array.foldl_toList, ← Array.foldl_toList, Array.toList_extract, Array.toList_extract,
    foldl_snd_sum, foldl_snd_sum, foldl_snd_sum]
  simp only [Nat.zero_add, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero]
  rw [List.take_of_length_le (l := a.toList.drop (a.size - k)) (by simp), ← List.sum_append, ← List.map_append,
    List.take_append_drop]

/-- **The pinned words are the last output words.** -/
theorem pinnedFrom_add (c : Circuit) : HmOut.pinnedFrom c + HmOut.pinnedWords c = c.outWords :=
  split_sum c.outPorts (HmRow.pinnedCount c.metaJ)

/-- **The first pinned word** is the first word of the first pinned output port. -/
theorem pinnedFrom_eq (c : Circuit) :
    HmOut.pinnedFrom c = Hidden.outAt c (c.outPorts.size - HmRow.pinnedCount c.metaJ) := by
  unfold HmOut.pinnedFrom Hidden.outAt
  rw [← Array.foldl_toList, Array.toList_extract, foldl_snd_sum]
  simp only [Nat.zero_add, List.extract_eq_take_drop, List.drop_zero, Nat.sub_zero]

theorem outAt_le (c : Circuit) {i j : ℕ} (h : i ≤ j) : Hidden.outAt c i ≤ Hidden.outAt c j := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  unfold Hidden.outAt
  rw [List.take_add, List.map_append, List.sum_append]
  omega

/-- **A committed circuit's output rows sum to its unpinned output ports** (`Hidden.parse_rowPorts`, on the circuit
`HmRow.parse` read, after a templated file's unit net). -/
theorem committed_rowPorts {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (ht : tags.typed = false) (hC : Hidden.Committed tags cf tables c) :
    c.outRowPorts.foldl (· + ·) 0 = c.outPorts.size - HmRow.pinnedCount c.metaJ := by
  rcases hC with hC | hC
  · exact (Hidden.parse_rowPorts (Hidden.parse_untyped ht hC).1).2
  · obtain ⟨tu, c₀, hp, -, -, rfl⟩ := TmplValue.parseTmpl_cases hC
    exact (Hidden.parse_rowPorts hp).2

/-- **Output rows past which the ports are pinned**: at `Hidden.Any.OutRowsP`, the output rows are the output ports but
the pinned ones (`committed_rowPorts`). -/
theorem outRows_size {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (ht : tags.typed = false) (hC : Hidden.Committed tags cf tables c) (hrows : Hidden.Any.OutRowsP c) :
    c.outRowPorts.size = c.outPorts.size - HmRow.pinnedCount c.metaJ := by
  have := committed_rowPorts ht hC
  rw [hrows.1, Hidden.foldl_replicate_one] at this
  exact this

/-- **A word of an output port without an output row is pinned**, and so are its bits at every template instance. -/
theorem outCol_mem_pinned {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (ht : tags.typed = false) (hC : Hidden.Committed tags cf tables c) (hrows : Hidden.Any.OutRowsP c) {g w t : ℕ}
    (hg : g < c.g) (hw : w < c.outWords) (hjr : c.outRowPorts.size ≤ outPortOf c w) (ht16 : t < 16) :
    HmOut.outCol c g w + t ∈ Hidden.pinnedCols c := by
  have hpf : HmOut.pinnedFrom c ≤ w := by
    rw [pinnedFrom_eq, ← outRows_size ht hC hrows]
    exact (outAt_le c hjr).trans (outPortOf_spec c hw).2.1
  have hsplit := pinnedFrom_add c
  unfold Hidden.pinnedCols
  simp only [List.mem_flatMap, List.mem_range, List.mem_map]
  exact ⟨g, hg, w - HmOut.pinnedFrom c, by omega, t, ht16, by rw [Nat.add_sub_of_le hpf]⟩

/-! ## Where the unit returns the outputs -/

/-- `HmOut.outRow` in a circuit whose unit returns the outputs: the unit net, at the column of the returned-output port
`ret` names. -/
theorem outRow_eq {c : Circuit} (ho : c.outNet = c.unitNet) (o : ℕ) :
    HmOut.outRow c o = (c.unitNet, ((portCols #[c.unit.outGroups.getD 0 default]).getD
      ((retOf c (List.range c.upv) (Array.replicate c.outWords (0, 0))).getD o (0, 0)).2 default).1) := by
  unfold HmOut.outRow
  simp only [ho, bne_self_eq_false, Bool.false_eq_true, ↓reduceIte]
  rfl

/-- **Every output word is a copy's output**, in an untyped circuit whose unit returns the outputs
(`check_returned_all`). -/
theorem returned_leaf {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet = c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) {o : ℕ} (hlt : o < c.outWords) :
    ∃ u < c.upv, ∃ (j : ℕ) (hj : j < (c.leavesOut.getD u #[]).size), (c.leavesOut.getD u #[])[j] = o := by
  have hm := (check_returned_all ht ho h).2 o hlt
  simp only [returned, List.mem_flatMap, List.mem_range] at hm
  obtain ⟨u, hu, hm⟩ := hm
  obtain ⟨j, hj, he⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hm)
  exact ⟨u, hu, j, hj, he⟩

variable {T : Model.Statement}

/-- **The pinned bits are distinct**, in an accepted statement whose unit returns the outputs: pinned word `o`'s bit `t`
at template instance `g` is column `16 j + t` of the returned-output port in unit slot `g · upv + u`, `o` copy `u`'s
output `j`, and below the unit's constant, so the column names `g`, `u`, `j` and `t`, and `u` and `j` name `o`. -/
theorem pinnedCols_nodup (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet) : (Hidden.pinnedCols a.st.c).Nodup := by
  obtain ⟨hty, ⟨ls, hls⟩, hcr⟩ := Hidden.Committed.check a.ht a.committed
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  dsimp only at ho hty hls hcr ⊢
  have C := check_facts hty hls
  obtain ⟨h16, -, hlo⟩ := Layout.check_out hty ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hnd := check_returned hty ho hls
  have hlt' : ∀ u < st.c.upv, ∀ o ∈ (st.c.leavesOut.getD u #[]).toList, o < st.c.outWords :=
    fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho'
  have hsz : ∀ u < st.c.upv, (st.c.leavesOut.getD u #[]).size = (st.c.unit.outGroups.getD 0 default).bits.size :=
    fun u hu => by rw [hget u hu]; exact (hlo u hu).1
  have hcp := Hidden.checkPinned_of_checkRows hcr
  have hsplit := pinnedFrom_add st.c
  have h2 : st.c.unitRange.slotLog = st.c.unit.unitLog := L.unit_slotLog
  have h1 : (netOf st.c st.c.unitNet).constPos < 2 ^ st.c.unit.unitLog := (L.fit st.c.unitNet).1
  have key : ∀ g o t, g < st.c.g → o < HmOut.pinnedWords st.c → t < 16 → ∃ u < st.c.upv, ∃ j : ℕ,
      ∃ hj : j < (st.c.leavesOut.getD u #[]).size, (st.c.leavesOut.getD u #[])[j] = HmOut.pinnedFrom st.c + o ∧
        HmOut.outCol st.c g (HmOut.pinnedFrom st.c + o) + t =
          st.c.unitRange.slot (g * st.c.upv + u) + ((st.c.unit.outGroups.getD 0 default).col * WORD + 16 * j + t) ∧
        (st.c.unit.outGroups.getD 0 default).col * WORD + 16 * j + t < 2 ^ st.c.unitRange.slotLog := by
    intro g o t hg ho' ht
    have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
    obtain ⟨u, hu, j, hj, hw⟩ := returned_leaf hty ho hls (o := HmOut.pinnedFrom st.c + o) (by omega)
    refine ⟨u, hu, j, hj, hw, ?_, ?_⟩
    · rw [← hw, outCol_leaf hty ho hnd hlt' h16 hsz hpv g hu hj]
      omega
    · obtain ⟨-, -, -, -, hrow⟩ := pinned_word hcp (o := HmOut.pinnedFrom st.c + o) (by omega) (by omega)
      have hl := hrow t ht
      rw [outRow_eq ho, ← hw, ret_leaf hnd hlt' hu hj] at hl
      dsimp only at hl
      rw [portCols_16_get h16 (q := j) (by rw [← hsz u hu]; exact hj)] at hl
      obtain ⟨-, -, hhi, -⟩ := linearRow_spec hl
      rw [h2]
      omega
  unfold Hidden.pinnedCols
  refine nodup_box (fun g o t => HmOut.outCol st.c g (HmOut.pinnedFrom st.c + o) + t) ?_
  intro g o t g' o' t' hg ho1 ht hg' ho1' ht' he0
  have he : HmOut.outCol st.c g (HmOut.pinnedFrom st.c + o) + t =
      HmOut.outCol st.c g' (HmOut.pinnedFrom st.c + o') + t' := he0
  obtain ⟨u, hu, j, hj, hw, hc, hb⟩ := key g o t hg ho1 ht
  obtain ⟨u', hu', j', hj', hw', hc', hb'⟩ := key g' o' t' hg' ho1' ht'
  rw [hc, hc'] at he
  obtain ⟨hs, hoff⟩ := slot_add_inj hb hb' he
  obtain ⟨rfl, rfl⟩ := mul_add_inj_lt hu hu' hs
  have hjj : j = j' := by omega
  have htt : t = t' := by omega
  subst hjj htt
  refine ⟨rfl, ?_, rfl⟩
  have : (st.c.leavesOut.getD u #[])[j]'hj = (st.c.leavesOut.getD u #[])[j]'hj' := rfl
  omega

/-- **A template instance's column on copy `u'`'s output `j`, bit `t`** is bit `t` of output word `lo[j]`'s column
(`HmOut.outCol` of the committed circuit), in an accepted statement whose statement circuit is `c` and whose unit returns
the outputs: the committed column is `upos` of the logical one (`col_val_compH`, `Hidden.outCol_commit16`,
`outCol_leaf`). -/
theorem col_outH (a : AcceptedH T) {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c)
    (ho : c.outNet = c.unitNet) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' j t : ℕ} (hu' : u' < c.upv)
    (hj : j < (c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol c j t)
    (h2 : Layout.outCol c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol c j t -
        (unitRows a.st a.L.unit_built).nIn)))) :
    (a.col hg x).val = HmOut.outCol a.st.c g (c.leavesOut.getD u' #[])[j] + t := by
  obtain ⟨nets, he⟩ := Hidden.Committed.unrolled_eq a.ht a.committed
  have hc : c = { a.st.c with nets := nets, tmpl := none } := hce.symm.trans he
  subst hc
  have hu0 : u' < a.st.c.upv := hu'
  have hj0 : j < (a.st.c.leavesOut.getD u' #[]).size := hj
  have ho0 : a.st.c.outNet = a.st.c.unitNet := ho
  obtain ⟨hty, ⟨ls, hls⟩, -⟩ := Hidden.Committed.check a.ht a.committed
  obtain ⟨h16, -, hlo⟩ := Layout.check_out hty ho0 hls
  have hget : ∀ u < a.st.c.upv, a.st.c.leavesOut.getD u #[] = a.st.c.leavesOut[u]! := fun u hu => by
    rw [Hidden.getElem!_eq_getD']; rfl
  have hj16 : j < (a.st.c.unit.outGroups.getD 0 default).bits.size := by
    rw [← (hlo u' hu0).1, ← hget u' hu0]; exact hj0
  have hoc := Hidden.outCol_commit16 a.ht a.committed h16 hj16 ht
  rw [he] at hoc
  have hcol := col_val_compH a hg x hu0 h1 h2 hx
  rw [← hoc] at hcol
  have C := check_facts hty hls
  have hpv := perVu_unit C a.L.unit_mem a.L.unit_kind a.L.unit_count (by omega)
  have hleaf := outCol_leaf hty ho0 (check_returned hty ho0 hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu0 hj0
  have hw : (({ a.st.c with nets := nets, tmpl := none } : Flock.Circuit).leavesOut.getD u' #[])[j]'hj =
      (a.st.c.leavesOut.getD u' #[])[j]'hj0 := rfl
  rw [hw, hcol, hleaf]
  unfold Layout.outCol
  omega

/-- **A template instance's column on a bit of an output word without an output row is a pinned bit**, in an accepted
statement whose statement circuit is `c` and whose unit returns the outputs (`col_outH`, `outCol_mem_pinned`, once by
`pinnedCols_nodup`). -/
theorem pinBit_outH (a : AcceptedH T) {c : Flock.Circuit} (hce : HmOut.unrolled a.st.c = c)
    (ho : c.outNet = c.unitNet) (hrows : Hidden.Any.OutRowsP c) {g : ℕ} (hg : g < a.st.c.g) (x : Fin a.rows.w)
    {u' j t : ℕ} (hu' : u' < c.upv) (hj : j < (c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol c j t)
    (h2 : Layout.outCol c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    (hjr : c.outRowPorts.size ≤ outPortOf c (c.leavesOut.getD u' #[])[j]) :
    PinBit a.st.c (a.col hg x).val := by
  have hcol := col_outH a hce ho hg x hu' hj ht h1 h2 hx
  have hlo := a.leavesOut_ltU hce ho
  have hwlt : (c.leavesOut.getD u' #[])[j] < c.outWords := by
    have hget : c.leavesOut.getD u' #[] = c.leavesOut[u']! := by
      rw [Hidden.getElem!_eq_getD']; rfl
    refine hlo u' hu' _ ?_
    rw [← hget]
    exact Array.getElem_mem_toList _
  have ho0 := a.outNetU hce ho
  obtain ⟨nets, rfl⟩ := a.stmt_eq hce
  have hm := outCol_mem_pinned (c := a.st.c) a.ht a.committed hrows hg hwlt hjr ht
  unfold PinBit
  rw [hcol]
  exact (congrArg (fun k : ℕ => (k : ZMod 2)) (List.count_eq_one_of_mem (pinnedCols_nodup a ho0) hm)).trans
    Nat.cast_one

end FlockSoundness.Discharge.HiddenExec
