import FlockSoundness.Protocol.Discharge.Canonical.Format
import FlockSoundness.Protocol.Discharge.HiddenExec.SiteH

set_option Elab.async false

/-!
# Output rows of runs of output ports, exact length: Δ's copy into a data bit

`Hidden.msg_copyAll` and `HiddenExec.out_copyH` at canonical output rows: one per run of output ports
(`HmOut.outGroup`), each a `sha512/row/v2` row of exactly its run's `16 · words` bits. Such a row's data may end inside
its last block, the `hm96` slot's (`chunks = ⌈(2 · words + 17) / 128⌉ - 1`), so a data byte `b` is in compression
`b / 128` or, at `b / 128 = chunks`, in the `hm96` slot's `pad` (`deltaMsgCol_hm`). Either way:

* no copy of `HmRow.delta` is at a data bit (`msg_filterC`): `HmRow.constBit` is `none` below the row's length
  (`constBit_msgOff`), so neither the compressions' constant bits nor the `pad`'s write there;
* `outPairs` copies into it once, from `outSrc` (`outPairsC_count`): distinct data bits are distinct columns
  (`deltaMsgCol_injC`);
* so a satisfying witness holds `outSrc`'s bit there (`msg_copyC`), and a returned output's column holds its run's row
  bit (`out_copyHC`).

A tail that returns the outputs leaves the units no returned output (`check_tail`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Canonical

open FlockSoundness.Discharge.Hidden FlockSoundness.Discharge.Hm Compose Flock.DeriveAll

/-! ## A data bit's column -/

/-- **A data bit is no constant**: bit `t` of byte `b` below the row's length is `HmRow.constBit`'s `none` at its block
`b / 128` and block column `msgOff b t`. -/
theorem constBit_msgOff {p : Port} {tl : ByteArray} {b t : ℕ} (ht : t < 8) (hd : 8 * b + t < p.bits) :
    HmRow.constBit p tl (b / 128) (msgOff b t) = none := by
  have h1 : 128 * (b / 128) + (HmRow.blockBit (msgOff b t)).1 = b := by
    simp only [HmRow.blockBit, msgOff]
    omega
  have h2 : (HmRow.blockBit (msgOff b t)).2 = t := by
    simp only [HmRow.blockBit, msgOff]
    omega
  unfold HmRow.constBit
  simp only [h1, h2]
  exact ite_eq_left hd

/-- **A data byte past the row's compressions is in its `hm96` slot's `pad`**. -/
theorem deltaMsgCol_hm {c : Circuit} (hrn : HmNets.check c = .ok ()) {g p b t : ℕ}
    (hs : (c.ports.getD p default).segs = none) (hp : p < c.ports.size) (hk : c.ports[p]!.chunks ≤ b / 128) :
    deltaMsgCol c g p b t = c.slotCol c.hmNet g p + (512 + msgOff b t) := by
  have hk' : ¬ (HmRow.byteAt (c.ports.getD p default) b).1 < ((HmRow.rowSlots c.ports).2[p]!).size := by
    rw [Typed.byteAt_none hs, rowSlots_size hp]
    omega
  simp only [deltaMsgCol, hk', ↓reduceIte]
  simp only [Typed.byteAt_none hs, msgOff, (hmIn_cols hrn).2]
  omega

section Cols

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

theorem deltaMsgCol_atHm {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hk : c.ports[p]!.chunks ≤ b / 128)
    (ht : t < 8) :
    AtSlot c c.hmNet (g * c.perVu.getD c.hmNet 0 + p) (512 + msgOff b t) (deltaMsgCol c g p b t) ∧
      512 + msgOff b t < 3072 := by
  have hmo := msgOff_lt (b := b) ht
  rw [deltaMsgCol_hm R.rowNets (segs_getD hu hp) hp hk]
  exact ⟨R.hmIn_at hg hp (by omega), by omega⟩

/-- **Distinct data bits are distinct columns**: VU, row, byte and bit, whether in a compression or the `hm96` slot. -/
theorem deltaMsgCol_injC {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hb : b / 128 ≤ c.ports[p]!.chunks) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hb' : b' / 128 ≤ c.ports[p']!.chunks) (ht' : t' < 8) (h : deltaMsgCol c g p b t = deltaMsgCol c g' p' b' t') :
    g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  have hmo := msgOff_lt (b := b) ht
  have hmo' := msgOff_lt (b := b') ht'
  rcases Nat.lt_or_ge (b / 128) c.ports[p]!.chunks with hk | hk <;>
    rcases Nat.lt_or_ge (b' / 128) c.ports[p']!.chunks with hk' | hk'
  · exact deltaMsgCol_inj R hu hg hp (by omega) ht hg' hp' (by omega) ht' h
  · rw [deltaMsgCol_hm R.rowNets (segs_getD hu hp') hp' hk'] at h
    exact absurd h.symm (hm_ne_msg R hu hg' hp' (by omega) hg hp (by omega) ht)
  · rw [deltaMsgCol_hm R.rowNets (segs_getD hu hp) hp hk] at h
    exact absurd h (hm_ne_msg R hu hg hp (by omega) hg' hp' (by omega) ht')
  · rw [deltaMsgCol_hm R.rowNets (segs_getD hu hp) hp hk,
      deltaMsgCol_hm R.rowNets (segs_getD hu hp') hp' hk'] at h
    obtain ⟨rfl, rfl, ho⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) h
    unfold msgOff at ho
    refine ⟨rfl, rfl, ?_, ?_⟩ <;> omega

/-- **No copy of `HmRow.delta`'s rows or of the rows' wires is at a data bit**, for any unsegmented rows. -/
theorem msg_filterC {pin g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hb : b / 128 ≤ c.ports[p]!.chunks)
    (hd : 8 * b + t < c.ports[p]!.bits) (ht : t < 8) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  have hmo := msgOff_lt (b := b) ht
  have hcb := constBit_msgOff (p := c.ports[p]!) (tl := HmRow.tailOf c.ports[p]!) ht hd
  rcases Nat.lt_or_ge (b / 128) c.ports[p]!.chunks with hk | hk
  · have hx := deltaMsgCol_pos R hu (g := g) (t := t) hp hk
    rw [filter_rows R hu hg hp (Or.inl ⟨b / 128, 512 + msgOff b t, hk, by omega, hx⟩)]
    have hw : (wireFAll c g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] := filter_nil_of _ _ fun e he => by
      rw [hx]
      rcases wireFAll_mem R hu hp he with ⟨k', t', -, hk', ht', rfl⟩ | ⟨-, t', ht', rfl⟩
      · intro h
        have := (shaPos_inj R hu hg hp hk' (by omega) hg hp hk (by omega) h).2.2.2
        omega
      · exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
    have hr : (rowCopiesOf c pin g p).filter (fun e => e.1 = deltaMsgCol c g p b t) = [] :=
      filter_nil_of _ _ fun e he => by
        rw [hx]
        simp only [rowCopiesOf, List.mem_append] at he
        rcases he with (he | he) | he
        · rcases cvCopies_fst R hu hp he with ⟨h0, t', ht', he1⟩ | ⟨h, -⟩
          · rw [he1]
            intro h
            have := (shaPos_inj R hu hg hp h0 (by omega) hg hp hk (by omega) h).2.2.2
            omega
          · omega
        · obtain ⟨t', ht', he1⟩ := padCopies_mem R hu he
          rw [he1]
          exact fun h => shaPos_ne_hm R hu hg hp hk (by omega) hg hp (by omega) h.symm
        · simp only [msgCopies, List.mem_flatMap, List.mem_range, List.mem_range'_1, rowSlots_size hp] at he
          obtain ⟨k', hk', t', ht', he⟩ := he
          rw [bitCopy_fst he, msgColOf_pos R hu hp (by omega)]
          intro h
          obtain ⟨-, -, hkk, hoo⟩ := shaPos_inj R hu hg hp (by omega) (by omega) hg hp hk (by omega) h
          have htt : t' = msgOff b t := by omega
          subst hkk htt
          rw [hcb] at he
          simp [bitCopy] at he
    rw [hw, hr, List.append_nil]
  · have hk' : c.ports[p]!.chunks = b / 128 := le_antisymm hk hb
    have hx : deltaMsgCol c g p b t = padCol c g p + msgOff b t := by
      rw [deltaMsgCol_hm R.rowNets (segs_getD hu hp) hp hk, padCol_eq R.rowNets]
      omega
    rw [hx, pad_filterAll R hu hg hp (by omega), rowSlots_size hp, hk', hcb]
    rfl

end Cols

/-! ## Δ's pairs at an output row's data bit -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`HmRow.delta`'s pairs at a data bit**: none, in a compression or in the `hm96` slot. -/
theorem flat_count_msgC (hrw : RowWires st.c) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) {g p b t : ℕ}
    (hg : g < st.c.g) (hp : p < st.c.ports.size) (hb : b / 128 ≤ st.c.ports[p]!.chunks)
    (hd : 8 * b + t < st.c.ports[p]!.bits) (ht : t < 8) (j : ℕ) :
    (flatPairsAll st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  have hmo := msgOff_lt (b := b) ht
  rcases Nat.lt_or_ge (b / 128) st.c.ports[p]!.chunks with hk | hk
  · have hlt : b < 128 * st.c.ports[p]!.chunks := by omega
    obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hlt ht
    rw [flatAll_count L F hrw (Or.inl rfl) hat (shaNet_ofRows R.rowNets)
      (by rw [shaNet_inRows R.rowNets]; omega) j, msg_filterC R hu hg hp hb hd ht]
    rfl
  · obtain ⟨hat, -⟩ := deltaMsgCol_atHm R hu hg hp hk ht
    rw [flatAll_count L F hrw (Or.inr rfl) hat (hmNet_ofRows R.rowNets) (by rw [hmNet_inRows R.rowNets]; omega) j,
      msg_filterC R hu hg hp hb hd ht]
    rfl

end Twin

/-- `outPairs`' copies: each output row's data bit, from the output bit the circuit computes, one output row per run. -/
def outCopiesC (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outRowPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).map fun t => (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)

theorem outPairs_eqC (c : Circuit) : outPairs c = (outCopiesC c).flatMap pair2 := by
  simp only [outPairs, outCopiesC, List.flatMap_assoc, List.flatMap_map, pair2]

/-- **Canonical output rows**: unsegmented rows, output row `j` after the input rows, of `16 · words` bits, its
`2 · words` bytes inside its compressions and its `hm96` slot's block. -/
def OutRowsC (c : Circuit) : Prop :=
  (∀ p ∈ c.ports.toList, p.segs = none) ∧ ∀ j < c.outRowPorts.size, HmOut.nIn c + j < c.ports.size ∧
    (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words ∧
    2 * (c.ports.getD (HmOut.nIn c + j) default).words ≤ 128 * ((c.ports.getD (HmOut.nIn c + j) default).chunks + 1)

theorem outRowC_bound {c : Circuit} (hrows : OutRowsC c) {j k t : ℕ} (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    HmOut.nIn c + j < c.ports.size ∧ (2 * k + t / 8) / 128 ≤ c.ports[HmOut.nIn c + j]!.chunks ∧
      8 * (2 * k + t / 8) + t % 8 < c.ports[HmOut.nIn c + j]!.bits ∧ t % 8 < 8 := by
  obtain ⟨hp, hb, hw⟩ := hrows.2 j hj
  rw [getElem!_eq_getD']
  refine ⟨hp, by omega, by omega, by omega⟩

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRowsC c)
include R hrows

/-- **`outPairs`' one copy at output row `j`'s data bit for word `k`, bit `t`**: from `outSrc`. -/
theorem outCopiesC_filter {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    (outCopiesC c).filter (fun e => e.1 = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
      [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)] := by
  have hdec : ∀ g' < c.g, ∀ j' < c.outRowPorts.size, ∀ k' < (c.ports.getD (HmOut.nIn c + j') default).words,
      ∀ t' < 16, deltaMsgCol c g' (HmOut.nIn c + j') (2 * k' + t' / 8) (t' % 8) =
        deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) → g' = g ∧ j' = j ∧ k' = k ∧ t' = t := by
    intro g' hg' j' hj' k' hk' t' ht' he
    obtain ⟨hp', hb', -, ht''⟩ := outRowC_bound hrows hj' hk' ht'
    obtain ⟨hp, hb, -, ht'''⟩ := outRowC_bound hrows hj hk ht
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_injC R hrows.1 hg' hp' hb' ht'' hg hp hb ht''' he
    exact ⟨h1, by omega, by omega, by omega⟩
  unfold outCopiesC
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

/-- **`outPairs`' pairs at output row `j`'s data bit for word `k`, bit `t`**: one copy from `outSrc`. -/
theorem outPairsC_count {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) (i : ℕ) :
    (outPairs c).count (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), i) =
      (if i = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
        (if i = outSrc c g j k t then 1 else 0) := by
  rw [outPairs_eqC, count_pair2, outCopiesC_filter R hrows hg hj hk ht]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

end Out

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st))
  (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsC st.c)
include L L0 F0 hda hrows

/-- **A canonical output row's data bit holds its copy** in every block of a satisfying witness of the hidden
statement: at VU `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copyC {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outRowPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  obtain ⟨hp, hb, hd, ht8⟩ := outRowC_bound hrows hj hk ht
  have hfc := flat_count_msgC L0 F0 F0.parse.rowWires hrows.1 hg hp hb hd ht8
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
        delta_flat_all _ _ F0.parse.typed hrows.1, hfc, Nat.zero_add, outPairsC_count R hrows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  rcases Nat.lt_or_ge ((2 * k + t / 8) / 128) st.c.ports[HmOut.nIn st.c + j]!.chunks with hkc | hkc
  · have hlt : 2 * k + t / 8 < 128 * st.c.ports[HmOut.nIn st.c + j]!.chunks := by omega
    obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hrows.1 hg hp hlt ht8
    have hx := deltaMsgCol_lt R hrows.1 hg hp hlt ht8
    exact block_of_rows hz o hx fun i =>
      inBit_rows (st := st) R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ i
  · obtain ⟨hat, hoff⟩ := deltaMsgCol_atHm R hrows.1 hg hp hkc ht8
    have hx := hat.lt R.bound (R.range R.hmNet).1
    exact block_of_rows hz o hx fun i =>
      inBit_rows (st := st) R (Or.inr rfl) hat (hmNet_ofRows R.rowNets) (hmLaid_self hoff) hx hδ i

end Hidden

/-! ## At an accepted statement -/

/-- `formatOk`'s `outRowsExact`: output row `j` is `16 · words` bits, `words` its run's. -/
theorem formatOk_out {c : Circuit} (h : formatOk c = true) {j : ℕ} (hj : j < c.outRowPorts.size) :
    (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (HmOut.outGroup c j).2.2.2 := by
  unfold formatOk at h
  simp only [Bool.and_eq_true] at h
  have hout := h.1.1.1.1.2
  unfold outRowsExact at hout
  exact beq_iff_eq.1 (List.all_eq_true.1 hout j (List.mem_range.2 hj))

/-- **A canonical circuit's output rows are `OutRowsC`**, and output row `j`'s words are its run's: a `sha512/row/v2`
row of `16 · words` bits is `words` u16 leaves and `⌈(2 · words + 17) / 128⌉ - 1` compressions (`Hidden.rowPorts_v2`). -/
theorem outRowsC_of {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hU : Built c.unit) (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hfmt : formatOk c = true) :
    OutRowsC c ∧ ∀ j < c.outRowPorts.size,
      (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 := by
  obtain ⟨hrow, hout, -⟩ := formatOk_spec hU hfmt
  obtain ⟨⟨mj, typed, hrp⟩, -⟩ := parse_rowPorts hp
  have key : ∀ j < c.outRowPorts.size, HmOut.nIn c + j < c.ports.size ∧
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words ≤
        128 * ((c.ports.getD (HmOut.nIn c + j) default).chunks + 1) ∧
      (c.ports.getD (HmOut.nIn c + j) default).words = (HmOut.outGroup c j).2.2.2 := by
    intro j hj
    have hlt := (checkRows_out hc hj).1
    have hmem := getD_mem_toList (d := default) hlt
    have hb := hout j hj
    obtain ⟨hv, hs⟩ := hrow _ hmem
    obtain ⟨hw, hch, -⟩ := rowPorts_v2 hrp _ hmem hv
    have hch := hch hs
    generalize c.ports.getD (HmOut.nIn c + j) default = q at hb hw hch ⊢
    refine ⟨hlt, by omega, by omega, by omega⟩
  exact ⟨⟨fun p hp => (hrow p hp).2, fun j hj => ⟨(key j hj).1, (key j hj).2.1, (key j hj).2.2.1⟩⟩,
    fun j hj => (key j hj).2.2.2⟩

/-- **A tail that returns the outputs leaves the units none**: `HmRow.check` refuses a nonempty `leavesOut` when the
out net is not the unit's. -/
theorem check_tail {c : Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none) (ho : c.outNet ≠ c.unitNet)
    (h : HmRow.check c tags ls = .ok ()) (k : ℕ) : (c.leavesOut.getD k #[]).size = 0 := by
  unfold HmRow.check at h
  obtain ⟨_spans, -, h⟩ := except_bind_ok h
  obtain ⟨_u0, -, h⟩ := except_bind_ok h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  simp only [ht] at h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  obtain ⟨-, h⟩ := Layout.ite_throw_cond h
  rw [ite_eq_left (bne_iff_ne.2 ho)] at h
  obtain ⟨hc, -⟩ := Layout.ite_throw_cond h
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true', not_exists, not_and, Bool.not_eq_true'] at hc
  by_cases hu : k < c.leavesOut.size
  · have := hc.2 _ (Array.getElem_mem hu)
    simp only [Bool.not_eq_false, Array.isEmpty_iff_size_eq_zero] at this
    simpa [Array.getD, hu] using this
  · simp [Array.getD, hu]

open FlockSoundness.Discharge.HiddenExec in
/-- **A drawn unit's returned output holds its canonical output row's bit** (`out_copyH` at runs): in a satisfying
witness of an accepted statement whose unit returns the outputs and whose output rows are canonical (`OutRowsC`), at
block `o` and VU `g`, copy `u'`'s computed column `outCol j t` (its output `j`, bit `t`) holds the witness's bit at
output row `nIn + jr`'s message column for bit `16 k + t`, where the copy's output word `lo[j]` is word `k` of run
`jr`. -/
theorem out_copyHC {T : Model.Statement} (a : AcceptedH T) (ho : a.st.c.outNet = a.st.c.unitNet)
    (hrows : OutRowsC a.st.c) {z : Model.Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin a.rows.w) {u' j t : ℕ} (hu' : u' < a.st.c.upv)
    (hj : j < (a.st.c.leavesOut.getD u' #[]).size) (ht : t < 16)
    (h1 : (unitRows a.st a.L.unit_built).nIn ≤ Layout.outCol a.st.c j t)
    (h2 : Layout.outCol a.st.c j t < (unitRows a.st a.L.unit_built).nIn + (unitRows a.st a.L.unit_built).nComp)
    (hx : x.val = a.st.c.upv * (unitRows a.st a.L.unit_built).nIn +
      (u' * ((unitRows a.st a.L.unit_built).nComp + 1) + (1 + (Layout.outCol a.st.c j t -
        (unitRows a.st a.L.unit_built).nIn))))
    {jr k : ℕ} (hjr : jr < a.st.c.outRowPorts.size)
    (hk : k < (a.st.c.ports.getD (HmOut.nIn a.st.c + jr) default).words) (hkr : k < (HmOut.outGroup a.st.c jr).2.2.2)
    (hw : (HmOut.outGroup a.st.c jr).2.2.1 + k = (a.st.c.leavesOut.getD u' #[])[j]) :
    T.block z o (a.col hg x) =
      z (siteAddrH a o (Typed.msgCol a.st.c g (HmOut.nIn a.st.c + jr) ((16 * k + t) / 8) ((16 * k + t) % 8))) := by
  have hcol := col_val_compH a hg x hu' h1 h2 hx
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at ho hrows hg x hu' hj h1 h2 hx hjr hk hkr hw hcol ⊢
  have s := setupHidden_spec' h
  obtain ⟨hpa, -⟩ := Hidden.parse_untyped ht' s.parse
  have P := parse_facts hpa
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have L0 := flat_layout ht' h
  have F0 := flat_unitFacts ht' h
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := by
    rw [s.da, hidden_delta']
  obtain ⟨h16, -, hlo⟩ := Layout.check_out P.typed ho hls
  have hget : ∀ u < st.c.upv, st.c.leavesOut.getD u #[] = st.c.leavesOut[u]! := fun u hu => by
    rw [getElem!_eq_getD']; rfl
  have hupv : 0 < st.c.upv := upv_pos C ho
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (by omega)
  have hleaf := outCol_leaf P.typed ho (check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hu' hj
  have hsrc : outSrc st.c g jr k t = st.c.unitRange.slot (g * st.c.upv + u') + Layout.outCol st.c j t := by
    unfold outSrc
    simp only [hkr, ↓reduceIte]
    rw [hw, hleaf]
    unfold Layout.outCol
    omega
  have hmc := msg_copyC L L0 F0 hda hrows hz o hg hjr hk ht
  have hp := (hrows.2 jr hjr).1
  have hb : (16 * k + t) / 8 = 2 * k + t / 8 := by omega
  have hb8 : (16 * k + t) % 8 = t % 8 := by omega
  have hin := Layout.msgCol_lt_of_parse hpa hg hp (2 * k + t / 8) (t % 8) (Nat.mod_lt _ (by norm_num))
  refine bit_inj _ _ ?_
  rw [hb, hb8, bit_siteAddrH ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht', h, L, hR, regs, rfl⟩
    z o hin, msgCol_delta st.c g (HmOut.nIn st.c + jr) (2 * k + t / 8) (t % 8), hmc, hsrc, ← hcol,
    blockBit_of_lt _ z o (Fin.isLt _)]

end FlockSoundness.Discharge.Canonical
