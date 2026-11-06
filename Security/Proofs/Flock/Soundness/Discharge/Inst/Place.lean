import Proofs.Flock.Soundness.Discharge.Inst.Rows

/-!
# A template instance's rows, placed by the statement the verifier accepted

`instRows c` (`Inst/Rows.lean`) puts each column `code n i off` of template instance `g` at `pos c g`, its slot's
column plus `off`, and `one` at the pin (`composePlace`). The verifier's level-0 matrices (`placedA`, `placedB`) place
them there (`placement_inst`), from the layout facts (`Layout`) and the circuit's parse, check and Δ
(`Aliased.UnitFacts`):
- at a computed row, the ranges' entry is the net's local row in its slot (`rangesEntry_net`), and Δ adds nothing
  (`delta_comp_net`);
- at a slot's constant, the local row reads itself and Δ moves it to the pin (`delta_const_net`);
- at an input bit a wire fills, Δ has the wire's one copy pair (`delta_wired`): `HmRow.check` wires each input port of
  every slot but the rows' circuits' exactly once (`CheckFacts.wired`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Compose

/-! ## The wire into an input bit -/

theorem portBits_nodup_net {tags : Tags} {c : Circuit} (P : ParseFacts tags c) {n : ℕ} (hn : n < c.nets.size) :
    (portBits (netOf c n).inGroups).Nodup := by
  rcases P.nets _ (getD_mem_toList hn) with hb | ⟨_, _, _, _, _, hb⟩
  · exact Aliased.portBits_nodup hb
  · rw [(build_facts hb).2.1, Aliased.portBits_eq]
    simpa using List.nodup_range'

theorem feed_some {c : Circuit} {n i o s : ℕ} (h : feed c n i o = some s) :
    o ∈ portBits (netOf c n).inGroups ∧ ∃ w, wireAt c n i o = some w ∧
      s = code w.src.1 w.src.2.1 (((portCols (netOf c w.src.1).outGroups).getD w.src.2.2 default).1 +
        (o - (portCols (netOf c n).inGroups)[portOf (portCols (netOf c n).inGroups) o]!.1)) := by
  unfold feed at h
  split at h
  · rename_i hm
    refine ⟨hm, ?_⟩
    rcases hw : wireAt c n i o with _ | w
    · rw [hw] at h; cases h
    · rw [hw] at h; exact ⟨w, rfl, (Option.some.inj h).symm⟩
  · cases h

/-- **The wire into an input port `HmRow.check` wired** is its only one, and no cut reaches the port. -/
theorem wire_one {c : Circuit} (C : CheckFacts c) {n i p : ℕ}
    (hn : n < c.nets.size) (hsha : n ≠ c.shaNet) (hhm : n ≠ c.hmNet) (hi : i < c.perVu.getD n 0)
    (hp : p < (portCols (netOf c n).inGroups).size) {w : Wire}
    (hw : (c.wires.toList.filter fun w => decide (w.dst = (n, i, p))).head? = some w) :
    (c.wires.toList.filter fun w => decide (w.dst = (n, i, p))) = [w] ∧
      (c.leafCuts.toList.filter fun e => decide ((c.unitNet, e.1, e.2.1) = (n, i, p))) = [] ∧
      w ∈ c.wires.toList ∧ w.dst = (n, i, p) := by
  obtain ⟨rest, hrest⟩ := List.head?_eq_some_iff.1 hw
  have hwm : w ∈ c.wires.toList.filter fun w => decide (w.dst = (n, i, p)) := by
    rw [hrest]; exact List.mem_cons_self
  obtain ⟨hwm', hwd⟩ := List.mem_filter.1 hwm
  have hwd' : w.dst = (n, i, p) := of_decide_eq_true hwd
  have hle : (if n = c.unitNet then (c.unit.inGroups.getD 0 default).bits.size else 0) ≤ p := by
    split
    · rename_i hu
      have := C.wire_leaf w hwm' (by rw [hwd', hu])
      rw [hwd'] at this
      exact this
    · exact Nat.zero_le _
  have hone := C.wired n hn hsha hhm i hi p hle hp
  rw [hrest, List.length_cons] at hone
  refine ⟨?_, List.length_eq_zero_iff.1 (by omega), hwm', hwd'⟩
  rw [hrest, List.length_eq_zero_iff.1 (show rest.length = 0 by omega)]

/-! ## Net slots in the block -/

theorem slotPairs_net {c : Circuit} {pin : ℕ} {r : Range} {i : ℕ} (hk : r.kind = .net i) :
    slotPairs c pin r = (List.range r.count).flatMap fun s =>
      if (r.slot s + (netOf c i).constPos != pin) = true then
        [(r.slot s + (netOf c i).constPos, r.slot s + (netOf c i).constPos), (r.slot s + (netOf c i).constPos, pin)]
      else [] := by
  unfold slotPairs
  rw [hk]

section net

variable {st : Stmt} (L : FlockSoundness.Layout st)
include L

/-- Two columns of net ranges' slots, each below its slot's size, are equal only in one net's range, in the same slot at
the same offset. -/
theorem eq_slot {R R' : Range} (hR : R ∈ st.c.ranges.toList) (hR' : R' ∈ st.c.ranges.toList) {n n' : ℕ}
    (hk : R.kind = .net n) (hk' : R'.kind = .net n') {s a s' b : ℕ} (hs : s < R.count) (ha : a < 2 ^ R.slotLog)
    (hs' : s' < R'.count) (hb : b < 2 ^ R'.slotLog) (he : R.slot s + a = R'.slot s' + b) :
    n = n' ∧ s = s' ∧ a = b := by
  have hx : InSpan R' (R.slot s + a) := by rw [he]; exact inSpan_of_covers (covers_slot hs' hb)
  obtain rfl := eq_of_inSpan L.disj hR hR' (inSpan_of_covers (covers_slot hs ha)) hx
  exact ⟨Kind.net.inj (hk.symm.trans hk'), slot_add_inj ha hb he⟩

/-- A column of a range's slot is in the block. -/
theorem range_lt {R : Range} (hR : R ∈ st.c.ranges.toList) {s off : ℕ} (hs : s < R.count)
    (hoff : off < 2 ^ R.slotLog) : R.slot s + off < 2 ^ st.c.kLog := by
  have := (covers_iff _ _).1 (covers_slot hs hoff)
  have := count_mul_le_spanH R
  have := L.bound _ hR
  omega

/-- Net `n`'s range: in the ranges, of net `n`, `perVu n` slots per template instance, of its net's size. -/
theorem net_range (C : CheckFacts st.c) {n : ℕ} (hn : n < st.c.nets.size) :
    st.c.rangeOf n ∈ st.c.ranges.toList ∧ (st.c.rangeOf n).kind = .net n ∧
      (st.c.rangeOf n).count = st.c.g * st.c.perVu.getD n 0 ∧ (st.c.rangeOf n).slotLog = (netOf st.c n).unitLog := by
  obtain ⟨hm, hk, -⟩ := rangeOf_spec (C.one_range n hn)
  exact ⟨hm, hk, C.count _ hm _ hk, L.slotLog _ hm n hk⟩

theorem net_slot_lt (C : CheckFacts st.c) {n g i : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := by
  rw [(net_range L C hn).2.2.1]
  exact Rows.mul_add_lt hg hi

omit L in
theorem pos_eq (c : Circuit) (g n i o : ℕ) :
    pos c g (code n i o) = (c.rangeOf n).slot (g * c.perVu.getD n 0 + i) + o := by
  rw [pos_code]; rfl

/-- **A column of a net slot is in the block.** -/
theorem pos_lt (C : CheckFacts st.c) {n g i o : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) (ho : o < netW st.c n) : pos st.c g (code n i o) < 2 ^ st.c.kLog := by
  obtain ⟨hm, -, -, hsl⟩ := net_range L C hn
  rw [pos_eq]
  exact range_lt L hm (net_slot_lt L C hn hg hi) (by rw [hsl]; exact ho)

/-- A net with a slot in the block has its rows in `rd32`'s words. -/
theorem netLog_le (C : CheckFacts st.c) {n g i : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) : (netOf st.c n).unitLog ≤ 32 := by
  have hp := Nat.two_pow_pos (netOf st.c n).unitLog
  have h := pos_lt L C hn hg hi (o := 2 ^ (netOf st.c n).unitLog - 1)
    (by show 2 ^ (netOf st.c n).unitLog - 1 < 2 ^ (netOf st.c n).unitLog; omega)
  rw [pos_eq] at h
  have h2 : 2 ^ (netOf st.c n).unitLog ≤ 2 ^ st.c.kLog := by omega
  have := (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).1 h2
  have := L.kLog
  omega

/-- **A net with a slot reads itself at its constant**, its input rows before it. -/
theorem net_const {tags : Tags} (P : ParseFacts tags st.c) (C : CheckFacts st.c) {n g i : ℕ}
    (hn : n < st.c.nets.size) (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0) : ConstRow (netOf st.c n) :=
  ((nets_ok P n).2 hn).2 (netLog_le L C hn hg hi)

/-- **The ranges' entry at a row of a net slot**: the net's local entry, in the same slot. -/
theorem rangesEntry_net (C : CheckFacts st.c) {n g i o : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) (ho : o < netW st.c n) (e : Kind → ℕ → ℕ → ZMod 2) (j : ℕ) :
    rangesEntry st e (pos st.c g (code n i o)) j =
      if j / netW st.c n = (st.c.rangeOf n).pos0 + (g * st.c.perVu.getD n 0 + i) then
        e (.net n) o (j % netW st.c n) else 0 := by
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  have ho' : o < 2 ^ (st.c.rangeOf n).slotLog := by rw [hsl]; exact ho
  rw [pos_eq, rangesEntry_of_covers L.disj hm (covers_slot (net_slot_lt L C hn hg hi) ho') e j,
    (slot_div _ ho').1, (slot_div _ ho').2, hk, hsl]

/-- **Δ adds nothing at a net slot's computed rows.** -/
theorem delta_comp_net (C : CheckFacts st.c) {n g i o : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) (hlo : netIn st.c n ≤ o) (hhi : o < (netOf st.c n).constPos) (j : ℕ) :
    deltaEntry st.da (pos st.c g (code n i o)) j = 0 := by
  obtain ⟨rest, hsplit, hrest⟩ := L.delta
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  have hs := net_slot_lt L C hn hg hi
  have hoffW : o < 2 ^ (st.c.rangeOf n).slotLog := by rw [hsl]; exact lt_trans hhi (L.fit n).1
  rw [deltaEntry_eq_count, pos_eq, hsplit, List.count_append,
    List.count_eq_zero.2 (L.constPairs_not_below hm hk hs hhi j),
    List.count_eq_zero.2 (fun hm' => L.not_inBit hm hk hs hoffW hlo (hrest _ hm'))]
  simp

/-- **Δ at a net slot's constant row**: `(k, k)` and `(k, pin)`, unless the constant is the pin. -/
theorem delta_const_net (C : CheckFacts st.c) {n g i : ℕ} (hn : n < st.c.nets.size) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD n 0) (hin : netIn st.c n ≤ (netOf st.c n).constPos) (j : ℕ) :
    deltaEntry st.da (pos st.c g (code n i (netOf st.c n).constPos)) j =
      if pos st.c g (code n i (netOf st.c n).constPos) ≠ st.pin then
        (if j = pos st.c g (code n i (netOf st.c n).constPos) then 1 else 0) + (if j = st.pin then 1 else 0)
      else 0 := by
  obtain ⟨rest, hsplit, hrest⟩ := L.delta
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  have hs₀ := net_slot_lt L C hn hg hi
  rw [pos_eq]
  generalize g * st.c.perVu.getD n 0 + i = s₀ at hs₀
  have hoffW : (netOf st.c n).constPos < 2 ^ (st.c.rangeOf n).slotLog := by rw [hsl]; exact (L.fit n).1
  have hx := inSpan_of_covers (covers_slot hs₀ hoffW)
  rw [deltaEntry_eq_count, hsplit, List.count_append,
    List.count_eq_zero.2 (fun hm' => L.not_inBit hm hk hs₀ hoffW hin (hrest _ hm')),
    Nat.add_zero, constPairs, count_flatMap_apart L.disj hm hx _ _ (fun a ha hax => by
      refine List.count_eq_zero.2 fun hm' => hax ?_
      obtain ⟨i', hki, s, hs', heq, -⟩ := slotPairs_fst hm'
      simp only at heq
      rw [heq]
      exact inSpan_of_covers (L.const_covers ha hki hs'))]
  rw [slotPairs_net hk, count_flatMap_range _ _ hs₀ (fun s hs' hne => by
    refine List.count_eq_zero.2 fun hm' => hne ?_
    split at hm'
    · simp only [List.mem_cons, List.not_mem_nil, or_false, Prod.mk.injEq] at hm'
      exact (slot_add_inj (off := (netOf st.c n).constPos) hoffW hoffW
        (by rcases hm' with ⟨h, -⟩ | ⟨h, -⟩ <;> exact h)).1.symm
    · cases hm')]
  by_cases hxp : (st.c.rangeOf n).slot s₀ + (netOf st.c n).constPos = st.pin
  · have : ((st.c.rangeOf n).slot s₀ + (netOf st.c n).constPos != st.pin) = false := by simp [hxp]
    rw [ite_eq_right (by simp [this]), ite_eq_right (by simpa using hxp)]
    simp
  · rw [ite_eq_left (bne_iff_ne.2 hxp), ite_eq_left hxp]
    by_cases h1 : j = (st.c.rangeOf n).slot s₀ + (netOf st.c n).constPos
    · subst h1
      simp [hxp]
    · by_cases h2 : j = st.pin
      · subst h2
        simp [h1, Ne.symm h1]
      · simp [h1, h2, Ne.symm h1, Ne.symm h2]

/-- A net slot's input port bit, in another net's range, is none of that range's slots' columns. -/
theorem ne_net_at {tags : Tags} {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (hok : NetsFit st.c)
    (C : CheckFacts st.c) {n n₀ : ℕ}
    (hn : n < st.c.nets.size) (hn₀ : n₀ < st.c.nets.size) (hne : n ≠ n₀) {g i : ℕ} {q : ℕ × ℕ} {tt : ℕ}
    (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0) (hq : q ∈ (portCols (netOf st.c n).inGroups).toList)
    (htt : tt < q.2) {s₀ b : ℕ} (hs₀ : s₀ < (st.c.rangeOf n₀).count) (hb : b < netW st.c n₀) :
    st.c.slotCol n g i + q.1 + tt ≠ (st.c.rangeOf n₀).slot s₀ + b := by
  intro he
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  obtain ⟨hm₀, hk₀, -, hsl₀⟩ := net_range L C hn₀
  have hq' := (hok n).1.2.2 q hq
  have hfit := (L.fit n).2
  have ha : q.1 + tt < 2 ^ (st.c.rangeOf n).slotLog := by
    rw [hsl]; show q.1 + tt < 2 ^ (st.c.nets.getD n default).2.unitLog; omega
  exact hne (eq_slot L hm hm₀ hk hk₀ (net_slot_lt L C hn hg hi) ha hs₀ (by rw [hsl₀]; exact hb)
    (by rw [← he]; unfold Circuit.slotCol; omega)).1

/-- No pair of Δ's rows' loops is at a column of a net slot other than the rows' circuits'. -/
theorem rows_ne {tags : Tags} {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (hok : NetsFit st.c)
    (C : CheckFacts st.c) {n : ℕ} (hn : n < st.c.nets.size)
    (hsha : n ≠ st.c.shaNet) (hhm : n ≠ st.c.hmNet) {s₀ b : ℕ} (hs₀ : s₀ < (st.c.rangeOf n).count)
    (hb : b < netW st.c n) {x : ℕ} (hx : Flat.RowBit st.c x) : x ≠ (st.c.rangeOf n).slot s₀ + b := by
  intro heq
  have hws := rowSlots_spec st.c.ports
  have hall := rowSlots_all st.c.ports
  rcases hx with ⟨g, hg, pp, hpp, k, hk, i, hi, tt, htt, he⟩ | ⟨g, hg, pp, hpp, i, hi, tt, htt, he⟩
  · obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hpp) _ (getElem_mem_toList' hk)
    have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[pp]!)[k]!.2 + i < 6 := by omega
    refine ne_net_at L P hok C P.shaNet hn (Ne.symm hsha) hg (by rw [C.perVu_sha]; exact h1)
      (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx)) ?_ hs₀ hb (he ▸ heq)
    rw [C.sha_in.2 _ hidx, sha_width _ _ hi]
    exact htt
  · obtain ⟨hil, hw⟩ := hm_width C.hm_in hi
    refine ne_net_at L P hok C P.hmNet hn (Ne.symm hhm) hg ?_ (getElem_mem_toList' hil)
      (by rw [hw]; exact htt) hs₀ hb (he ▸ heq)
    rw [C.perVu_hm, ← hws]
    exact lt_of_lt_of_le hpp (le_nextPow2 _)

/-- No leaf pair is at a column of another net's slot than the unit's. -/
theorem leafPairs_ne (C : CheckFacts st.c) {n : ℕ} (hn : n < st.c.nets.size) (hne : n ≠ st.c.unitNet) {s₀ b : ℕ}
    (hs₀ : s₀ < (st.c.rangeOf n).count) (hb : b < netW st.c n) (j : ℕ) :
    (Flat.leafPairs st.c).count ((st.c.rangeOf n).slot s₀ + b, j) = 0 := by
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  unfold Flat.leafPairs
  refine count_flatMap_zero fun k hk' => count_flatMap_zero fun jj hjj => count_flatMap_zero fun tt htt =>
    count_pair_ne fun he => ?_
  have hk'' := List.mem_range.1 hk'
  have hupv : 0 < st.c.upv := by
    rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
    · rw [L.unit_count, h0, Nat.mul_zero] at hk''; omega
    · exact h0
  have hlt : k % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact Nat.mod_lt _ hupv
  have hli : jj < (st.c.unit.inGroups.getD 0 default).bits.size := by
    have := List.mem_range.1 hjj
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
      ← getElem!_pos st.c.leavesIn _ hlt, C.leaves.2 _ (Nat.mod_lt _ hupv)] at this
    exact this
  obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hli
  have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
    C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hli)
  have htw : tt < (portCols st.c.unit.inGroups)[jj]!.2 := by rw [hw, h16]; exact List.mem_range.1 htt
  unfold Flat.leafCol at he
  rw [hcol] at he
  exact hne (eq_slot L L.unit_mem hm L.unit_kind hk hk'' (L.port_lt_slot hjs htw) hs₀ (by rw [hsl]; exact hb)
    (by rw [← he]; omega)).1.symm

/-- No cut pair is at a column of another net's slot than the unit's. -/
theorem cutPairs_ne (C : CheckFacts st.c) {n : ℕ} (hn : n < st.c.nets.size) (hne : n ≠ st.c.unitNet) {s₀ b : ℕ}
    (hs₀ : s₀ < (st.c.rangeOf n).count) (hb : b < netW st.c n) (j : ℕ) :
    (Flat.cutPairs st.c).count ((st.c.rangeOf n).slot s₀ + b, j) = 0 := by
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  unfold Flat.cutPairs
  refine count_flatMap_zero fun g' hg' => count_flatMap_zero fun e he => count_flatMap_zero fun tt htt =>
    count_pair_ne fun heq => ?_
  have hg'' := List.mem_range.1 hg'
  obtain ⟨he1, he2⟩ := C.cuts e he
  have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (Nat.lt_of_le_of_lt (Nat.zero_le _) hg'')
  have hcol : Flat.cutCol st.c g' e + tt =
      st.c.unitRange.slot (g' * st.c.upv + e.1) + ((portCols st.c.unit.inGroups)[e.2.1]!.1 + tt) := by
    unfold Flat.cutCol Circuit.slotCol
    rw [hpv, Nat.add_assoc]
    rfl
  rw [hcol] at heq
  exact hne (eq_slot L L.unit_mem hm L.unit_kind hk (L.slot_lt hg'' he1) (L.port_lt_slot he2 (List.mem_range.1 htt))
    hs₀ (by rw [hsl]; exact hb) heq).1.symm

/-- **The wire pairs at a net slot's input port bit**: the copy pair of each wire into its port, in its template
instance. -/
theorem wirePairs_net {tags : Tags} {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (hok : NetsFit st.c)
    (C : CheckFacts st.c) {n g i p t : ℕ}
    (hn : n < st.c.nets.size) (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0)
    (hnd : (portBits (netOf st.c n).inGroups).Nodup) (hp : p < (portCols (netOf st.c n).inGroups).size)
    (ht : t < (portCols (netOf st.c n).inGroups)[p]!.2) (j : ℕ) :
    (Flat.wirePairs st.c).count ((st.c.rangeOf n).slot (g * st.c.perVu.getD n 0 + i) +
        ((portCols (netOf st.c n).inGroups)[p]!.1 + t), j) =
      ((st.c.wires.toList.filter fun w => decide (w.dst = (n, i, p))).flatMap fun w =>
        [((st.c.rangeOf n).slot (g * st.c.perVu.getD n 0 + i) + ((portCols (netOf st.c n).inGroups)[p]!.1 + t),
          (st.c.rangeOf n).slot (g * st.c.perVu.getD n 0 + i) + ((portCols (netOf st.c n).inGroups)[p]!.1 + t)),
          ((st.c.rangeOf n).slot (g * st.c.perVu.getD n 0 + i) + ((portCols (netOf st.c n).inGroups)[p]!.1 + t),
            Flat.wireSrc st.c g w + t)]).count
        ((st.c.rangeOf n).slot (g * st.c.perVu.getD n 0 + i) + ((portCols (netOf st.c n).inGroups)[p]!.1 + t), j) := by
  obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
  have hq : (portCols (netOf st.c n).inGroups)[p]!.1 + (portCols (netOf st.c n).inGroups)[p]!.2 ≤ netIn st.c n :=
    (hok n).1.2.2 _ (getElem_mem_toList' hp)
  have hfit : netIn st.c n ≤ netW st.c n := (L.fit n).2
  have hx : (portCols (netOf st.c n).inGroups)[p]!.1 + t < 2 ^ (st.c.rangeOf n).slotLog := by
    rw [hsl]; show _ < netW st.c n; omega
  unfold Flat.wirePairs
  refine count_vu_items _ _ _ (fun g w tt => Flat.wireDst st.c g w + tt) (fun g w tt => Flat.wireSrc st.c g w + tt) _
    hg (fun g' hg' w hw tt htt => ?_) (fun w _ hF => ?_)
  · obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
    obtain ⟨⟨dn, di, dp⟩, src⟩ := w
    simp only at hd1 hd2 hd3 htt ⊢
    obtain ⟨hmd, hkd, -, hsld⟩ := net_range L C hd1
    have hqd : (portCols (netOf st.c dn).inGroups)[dp]!.1 + (portCols (netOf st.c dn).inGroups)[dp]!.2 ≤
        netIn st.c dn := (hok dn).1.2.2 _ (getElem_mem_toList' hd3)
    have hfitd : netIn st.c dn ≤ netW st.c dn := (L.fit dn).2
    have hwd : Flat.wireWidth st.c ⟨(dn, di, dp), src⟩ = (portCols (netOf st.c dn).inGroups)[dp]!.2 := by
      unfold Flat.wireWidth; exact congrArg Prod.snd (FlockSoundness.Layout.getD_portCols hd3)
    rw [hwd] at htt
    have hcol : Flat.wireDst st.c g' ⟨(dn, di, dp), src⟩ + tt =
        (st.c.rangeOf dn).slot (g' * st.c.perVu.getD dn 0 + di) +
          ((portCols (netOf st.c dn).inGroups)[dp]!.1 + tt) := by
      unfold Flat.wireDst Circuit.slotCol
      simp only
      rw [FlockSoundness.Layout.getD_portCols hd3, Nat.add_assoc]
    have hxd : (portCols (netOf st.c dn).inGroups)[dp]!.1 + tt < 2 ^ (st.c.rangeOf dn).slotLog := by
      rw [hsld]; show _ < netW st.c dn; omega
    rw [hcol]
    constructor
    · intro heq
      obtain ⟨rfl, h1, h2⟩ := eq_slot L hmd hm hkd hk (net_slot_lt L C hd1 hg' hd2) hxd
        (net_slot_lt L C hn hg hi) hx heq
      obtain ⟨rfl, rfl⟩ := portBits_inj hnd hd3 hp htt ht h2
      obtain ⟨h3, h4⟩ := (vu_split hd2).1 h1
      obtain ⟨h5, h6⟩ := (vu_split hi).1 rfl
      exact ⟨h3.trans h5.symm, by rw [h4.trans h6.symm], rfl⟩
    · rintro ⟨rfl, hF, rfl⟩
      simp only [Prod.mk.injEq] at hF
      obtain ⟨rfl, rfl, rfl⟩ := hF
      rfl
  · unfold Flat.wireWidth
    rw [hF]
    show t < ((portCols (netOf st.c n).inGroups).getD p default).2
    rw [FlockSoundness.Layout.getD_portCols hp]
    exact ht

/-- **Δ at an input bit a wire fills**: its one copy pair, `(x, x)` and `(x, src)`, the wire's source bit (`feed`). -/
theorem delta_wired {tags : Tags} (F : Aliased.UnitFacts tags st) {n g i o s : ℕ} (hn : n < st.c.nets.size)
    (hsha : n ≠ st.c.shaNet) (hhm : n ≠ st.c.hmNet) (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0)
    (hf : feed st.c n i o = some s) (j : ℕ) :
    st.da.toList.count (pos st.c g (code n i o), j) =
      (if pos st.c g (code n i o) = j then 1 else 0) + if pos st.c g s = j then 1 else 0 := by
  have P := F.parse
  have C := F.check
  obtain ⟨ho, w, hw, rfl⟩ := feed_some hf
  have hnd : (portBits (netOf st.c n).inGroups).Nodup := F.nodup n hn
  obtain ⟨p, t, hp, ht, hpt⟩ := Aliased.mem_portBits ho
  have hpo : portOf (portCols (netOf st.c n).inGroups) o = p := by rw [hpt]; exact portOf_eq hnd hp ht
  have hwat : (st.c.wires.toList.filter fun w => decide (w.dst = (n, i, p))).head? = some w := by
    unfold wireAt at hw; rwa [hpo] at hw
  obtain ⟨hwf, hcf, hwm, hwd⟩ := wire_one C hn hsha hhm hi hp hwat
  have hs : pos st.c g (code w.src.1 w.src.2.1 (((portCols (netOf st.c w.src.1).outGroups).getD w.src.2.2 default).1 +
      (o - (portCols (netOf st.c n).inGroups)[portOf (portCols (netOf st.c n).inGroups) o]!.1))) =
      Flat.wireSrc st.c g w + t := by
    rw [hpo, pos_code, hpt, Nat.add_sub_cancel_left]
    show _ = st.c.slotCol w.src.1 g w.src.2.1 + ((portCols (netOf st.c w.src.1).outGroups).getD w.src.2.2 default).1 + t
    omega
  rw [hs, pos_eq, hpt]
  clear hs hf hw
  by_cases hu : n = st.c.unitNet
  · subst hu
    have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count (Nat.lt_of_le_of_lt (Nat.zero_le _) hg)
    have hi' : st.c.perVu.getD st.c.unitNet 0 = st.c.upv := hpv
    rw [hi'] at hi ⊢
    have hs₀ := L.slot_lt hg hi
    have hd := L.delta_portT P (fun n => (F.nets n).1) C F.delta F.sha F.hm (by rw [← L.unit_at.groups]; exact Aliased.portBits_nodup L.unit_built) hs₀ hp ht j
    have hflat : Flat.flatSrc st.c (g * st.c.upv + i) p t = Flat.wireSrc st.c g w + t := by
      obtain ⟨h1, h2⟩ := (vu_split hi).1 rfl
      have hleaf : ¬ p < (st.c.unit.inGroups.getD 0 default).bits.size := by
        have := C.wire_leaf w hwm (by rw [hwd])
        rw [hwd] at this
        exact Nat.not_lt.2 this
      have hcf' : (st.c.leafCuts.toList.filter fun e => decide (e.1 = (g * st.c.upv + i) % st.c.upv ∧ e.2.1 = p)) =
          [] := by
        rw [← h2]
        exact (List.filter_congr fun e _ => by simp).trans hcf
      have hwf' : (st.c.wires.toList.filter fun w =>
          decide (w.dst = (st.c.unitNet, (g * st.c.upv + i) % st.c.upv, p))) = [w] := by
        rw [← h2]; exact hwf
      unfold Flat.flatSrc
      rw [ite_eq_right hleaf, hcf', hwf', ← h1]
    exact hd.trans (by rw [hflat]; rfl)
  · have hs := net_slot_lt L C hn hg hi
    obtain ⟨hm, hk, -, hsl⟩ := net_range L C hn
    have hcr := ((F.nets n).2 hn).2 (netLog_le L C hn hg hi)
    have hq : (portCols (netOf st.c n).inGroups)[p]!.1 + (portCols (netOf st.c n).inGroups)[p]!.2 ≤ netIn st.c n :=
      (F.nets n).1.2.2 _ (getElem_mem_toList' hp)
    have hfit : netIn st.c n ≤ netW st.c n := (L.fit n).2
    have hb : (portCols (netOf st.c n).inGroups)[p]!.1 + t < netW st.c n := by omega
    obtain ⟨rows, hd, hrows⟩ := F.delta
    rw [hd]
    simp only [List.count_append]
    have hc1 : netIn st.c n ≤ (netOf st.c n).constPos := hcr.1
    have hlt : (portCols (netOf st.c n).inGroups)[p]!.1 + t < (netOf st.c n).constPos := by omega
    rw [List.count_eq_zero.2 (L.constPairs_not_below hm hk hs hlt j),
      List.count_eq_zero.2 (fun hm' => rows_ne L P F.nets C hn hsha hhm hs hb (hrows _ hm') rfl),
      leafPairs_ne L C hn hu hs hb, cutPairs_ne L C hn hu hs hb, wirePairs_net L P F.nets C hn hg hi hnd hp ht, hwf]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, count_pair_eq, Nat.zero_add]

end net

/-- **Δ at the input bits a wire fills**: at each, its one copy pair, `(x, x)` and `(x, src)`, the wire's source bit
(`feed`), at every template instance. -/
def WiredDelta (st : Stmt) : Prop :=
  ∀ {n g i o s : ℕ}, n < st.c.nets.size → n ≠ st.c.shaNet → n ≠ st.c.hmNet → g < st.c.g →
    i < st.c.perVu.getD n 0 → feed st.c n i o = some s → ∀ j,
      st.da.toList.count (pos st.c g (code n i o), j) =
        (if pos st.c g (code n i o) = j then 1 else 0) + if pos st.c g s = j then 1 else 0

/-- `delta_wired` as `WiredDelta`. -/
theorem wiredDelta {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st) :
    WiredDelta st :=
  fun hn hsha hhm hg hi hf j => delta_wired L F hn hsha hhm hg hi hf j

/-! ## Reads, placed -/

theorem zmod2_cast (v : ZMod 2) : (((if v = 1 then 1 else 0 : ℕ)) : ZMod 2) = v := by
  revert v; decide

/-- **A local row's reads, placed**: inside its slot `B`, the row's entry; outside, nothing. -/
theorem count_reads {c : Circuit} {g n i : ℕ} (e : ℕ → ZMod 2) {B : ℕ}
    (hpos : ∀ j', pos c g (code n i j') = B * netW c n + j') (j : ℕ) :
    ((((reads c n i e).map (pos c g)).count j : ℕ) : ZMod 2) = if j / netW c n = B then e (j % netW c n) else 0 := by
  have hW : 0 < netW c n := Nat.two_pow_pos _
  have hmap : (reads c n i e).map (pos c g) =
      ((List.range (netW c n)).filter fun j' => decide (e j' = 1)).map (fun j' => B * netW c n + j') := by
    unfold reads
    rw [List.map_map]
    exact List.map_congr_left fun j' _ => hpos j'
  rw [hmap]
  have hinj : Function.Injective (fun j' => B * netW c n + j') := fun a b h => by simpa using h
  by_cases h : j / netW c n = B
  · rw [ite_eq_left h]
    obtain ⟨r, hr, rfl⟩ : ∃ r < netW c n, j = B * netW c n + r :=
      ⟨j % netW c n, Nat.mod_lt _ hW, by have := Nat.div_add_mod j (netW c n); rw [h, Nat.mul_comm] at this; exact this.symm⟩
    have hm : (B * netW c n + r) % netW c n = r := by
      rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]
    rw [hm, List.count_map_of_injective _ _ hinj]
    by_cases he : e r = 1
    · rw [List.count_filter (by simpa using he), List.count_eq_one_of_mem List.nodup_range (List.mem_range.2 hr), he]
      simp
    · rw [List.count_eq_zero.2 (fun hmem => he (by simpa using (List.mem_filter.1 hmem).2))]
      have : e r = 0 := by revert he; generalize e r = v; revert v; decide
      rw [this]; simp
  · rw [ite_eq_right h, List.count_eq_zero.2, Nat.cast_zero]
    intro hmem
    obtain ⟨j', hj', he⟩ := List.mem_map.1 hmem
    have hj'W := List.mem_range.1 (List.mem_filter.1 hj').1
    apply h
    rw [← he, Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt hj'W, Nat.zero_add]

theorem slot_iff {W B o j : ℕ} (hW : 0 < W) (ho : o < W) : B * W + o = j ↔ j / W = B ∧ j % W = o := by
  constructor
  · rintro rfl
    refine ⟨?_, ?_⟩
    · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hW, Nat.div_eq_of_lt ho, Nat.zero_add]
    · rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ho]
  · rintro ⟨h1, h2⟩
    have := Nat.div_add_mod j W
    rw [h1, h2] at this
    rw [← this, Nat.mul_comm]

/-- A wired row's sum: the local row, Δ's `(x, x)` and `(x, src)`, read as the local row with `x` added, and `src`. -/
theorem wired_sum {W B o j : ℕ} (hW : 0 < W) (ho : o < W) (e : ℕ → ZMod 2) (b : Prop) [Decidable b] :
    ((if j / W = B then e (j % W) else 0) + ((((if B * W + o = j then 1 else 0) + if b then 1 else 0 : ℕ)) : ZMod 2)) =
      (if j / W = B then addSelf e o (j % W) else 0) + (((if b then 1 else 0 : ℕ)) : ZMod 2) := by
  have hiff := slot_iff (B := B) (j := j) hW ho
  unfold addSelf
  by_cases h1 : j / W = B <;> by_cases h2 : j % W = o
  · simp only [h1, h2, hiff, and_self, ↓reduceIte, Nat.cast_add, Nat.cast_one]; ring
  · simp only [h1, h2, hiff, and_false, ↓reduceIte, zero_add, add_zero]
  · simp only [h1, h2, hiff, false_and, ↓reduceIte, zero_add]
  · simp only [h1, h2, hiff, false_and, ↓reduceIte, zero_add]

/-! ## The template instance's columns -/

theorem inX_lt {c : Circuit} {x : ℕ} (hx : x ∈ inX c) : x < c.upv * netIn c c.unitNet :=
  List.mem_range.1 (List.mem_filter.1 hx).1

theorem inCols_getD {c : Circuit} {ℓ : ℕ} (hℓ : ℓ < (inCols c).length) :
    (inCols c).getD ℓ 0 = inCode c ((inX c).getD ℓ 0) ∧ (inX c).getD ℓ 0 ∈ inX c := by
  have hℓ' : ℓ < (inX c).length := by simpa [inCols] using hℓ
  rw [List.getD_eq_getElem _ _ hℓ]
  simp only [inCols, List.getElem_map]
  rw [List.getD_eq_getElem _ _ hℓ']
  exact ⟨rfl, List.getElem_mem hℓ'⟩

/-- A column's parts are a net, one of its slots, and a column of its slot. -/
def Valid (c : Circuit) (x : ℕ) : Prop :=
  codeNet x < c.nets.size ∧ codeSlot x < c.perVu.getD (codeNet x) 0 ∧ codeOff x < netW c (codeNet x)

section placement

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
include L

/-- **Every column of the order is a net slot's**: the unit's stacked inputs, then columns with a row. -/
theorem order_valid {tags : Tags} {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (C : CheckFacts st.c)
    (hg0 : 0 < st.c.g) {ℓ : ℕ}
    (hℓ : ℓ < (order st.c).length) : Valid st.c ((order st.c).getD ℓ 0) := by
  by_cases hn : ℓ < (inCols st.c).length
  · rw [order_getD_in hn]
    obtain ⟨he, hm⟩ := inCols_getD hn
    rw [he]
    have hx := inX_lt hm
    generalize (inX st.c).getD ℓ 0 = x at hx
    have hW : 0 < netIn st.c st.c.unitNet := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
    have hpv := perVu_unit C L.unit_mem L.unit_kind L.unit_count hg0
    have hfit : netIn st.c st.c.unitNet ≤ netW st.c st.c.unitNet := (L.fit _).2
    refine ⟨by simpa [inCode] using P.unitNet, ?_, ?_⟩
    · simp only [inCode, codeNet_code, codeSlot_code]
      rw [hpv]
      exact (Nat.div_lt_iff_lt_mul hW).2 hx
    · simp only [inCode, codeNet_code, codeOff_code]
      exact lt_of_lt_of_le (Nat.mod_lt _ hW) hfit
  · have := rowAt_dom (order_row hℓ (by omega))
    exact ⟨this.1, this.2.2.2.1, this.2.2.2.2⟩

/-- The order's columns are in the block, at every template instance, from the parse and the check. -/
theorem inst_hbOf {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (C : CheckFacts st.c) {g : ℕ}
    (hg : g < st.c.g) :
    ∀ ℓ < (logical st.c).order.length, pos st.c g ((logical st.c).order.getD ℓ 0) < 2 ^ (L.model regs).kLog := by
  intro ℓ hℓ
  obtain ⟨h1, h2, h3⟩ := order_valid L P C (by omega) hℓ
  have := pos_lt L C h1 hg h2 h3
  rwa [code_parts] at this

include F in
/-- The order's columns are in the block, at every template instance. -/
theorem inst_hb {g : ℕ} (hg : g < st.c.g) :
    ∀ ℓ < (logical st.c).order.length, pos st.c g ((logical st.c).order.getD ℓ 0) < 2 ^ (L.model regs).kLog :=
  inst_hbOf L regs F.parse F.check hg

/-- **The block rows at a column with a computed row read its row's columns**, placed: a computed row's local rows, or
an input bit's local self row with Δ's copy pair (`WiredDelta`). -/
theorem placed_comp (C : CheckFacts st.c) (hD : WiredDelta st) {g n i o : ℕ} (hg : g < st.c.g)
    {p : List ℕ × List ℕ} (hrow : rowAt st.c n i o = some (some p))
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = pos st.c g (code n i o)) :
    placedA st r j = (((p.1.map (pos st.c g)).count j.val : ℕ) : ZMod 2) ∧
      placedB st r j = (((p.2.map (pos st.c g)).count j.val : ℕ) : ZMod 2) := by
  obtain ⟨hn, hsha, hhm, hi, ho⟩ := rowAt_dom (c := st.c) (n := n) (i := i) (o := o) (by rw [hrow]; simp)
  obtain ⟨-, hcase⟩ := rowAt_comp hrow
  obtain ⟨-, -, -, hsl⟩ := net_range L C hn
  have hW : 0 < netW st.c n := Nat.two_pow_pos _
  have hpos : ∀ j', pos st.c g (code n i j') =
      ((st.c.rangeOf n).pos0 + (g * st.c.perVu.getD n 0 + i)) * netW st.c n + j' := by
    intro j'
    rw [pos_eq]
    unfold Range.slot
    rw [hsl]
  rcases hcase with ⟨hlo, hhi, rfl⟩ | ⟨-, s, hf, rfl⟩
  · constructor
    · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
      rw [hr, rangesEntry_net L C hn hg hi ho, delta_comp_net L C hn hg hi hlo hhi, add_zero, count_reads _ hpos]
      rfl
    · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
      rw [L.delta_b, hr, rangesEntry_net L C hn hg hi ho, delta_comp_net L C hn hg hi hlo hhi, add_zero,
        count_reads _ hpos]
      rfl
  · have hcnt := hD hn hsha hhm hg hi hf j.val
    rw [hpos o] at hcnt
    have hR : ∀ e, ((((reads st.c n i e ++ [s]).map (pos st.c g)).count j.val : ℕ) : ZMod 2) =
        (if j.val / netW st.c n = (st.c.rangeOf n).pos0 + (g * st.c.perVu.getD n 0 + i) then
          e (j.val % netW st.c n) else 0) + (((if pos st.c g s = j.val then 1 else 0 : ℕ)) : ZMod 2) := by
      intro e
      rw [List.map_append, List.count_append, Nat.cast_add, count_reads e hpos, List.map_singleton]
      simp only [List.count_singleton, beq_iff_eq]
    have hA : slotEntryA st (.net n) o = entA st.c n o := rfl
    have hB : slotEntryB st (.net n) o = entB st.c n o := rfl
    constructor
    · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
      rw [hr, rangesEntry_net L C hn hg hi ho, deltaEntry_eq_count, hpos o, hcnt, hA]
      show _ = ((((reads st.c n i (addSelf (entA st.c n o) o) ++ [s]).map (pos st.c g)).count j.val : ℕ) : ZMod 2)
      rw [hR]
      exact wired_sum hW ho _ _
    · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
      rw [L.delta_b, hr, rangesEntry_net L C hn hg hi ho, deltaEntry_eq_count, hpos o, hcnt, hB]
      show _ = ((((reads st.c n i (addSelf (entB st.c n o) o) ++ [s]).map (pos st.c g)).count j.val : ℕ) : ZMod 2)
      rw [hR]
      exact wired_sum hW ho _ _

/-- **The block rows at a slot's constant read the pin.** -/
theorem placed_const (hok : NetsFit st.c) (C : CheckFacts st.c) {g n i o : ℕ} (hg : g < st.c.g)
    (hrow : rowAt st.c n i o = some none)
    (r j : Fin (2 ^ (L.model regs).kLog)) (hr : r.val = pos st.c g (code n i o)) :
    (L.model regs).A₀ r j = (if j = (L.model regs).pin then 1 else 0) ∧
      (L.model regs).B₀ r j = if j = (L.model regs).pin then 1 else 0 := by
  obtain ⟨hn, -, -, hi, ho⟩ := rowAt_dom (c := st.c) (n := n) (i := i) (o := o) (by rw [hrow]; simp)
  obtain rfl := rowAt_const hrow
  obtain ⟨-, -, -, hsl⟩ := net_range L C hn
  have hcr : ConstRow (netOf st.c n) := ((hok n).2 hn).2 (netLog_le L C hn hg hi)
  have hW : 0 < netW st.c n := Nat.two_pow_pos _
  have hpos : ∀ j', pos st.c g (code n i j') =
      ((st.c.rangeOf n).pos0 + (g * st.c.perVu.getD n 0 + i)) * netW st.c n + j' := by
    intro j'
    rw [pos_eq]
    unfold Range.slot
    rw [hsl]
  have hS : (L.model regs).pin = (⟨st.pin, L.pin_lt⟩ : Fin (2 ^ (L.model regs).kLog)) := rfl
  constructor
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val = _
    rw [hr, rangesEntry_net L C hn hg hi ho, delta_const_net L C hn hg hi hcr.1,
      ite_slot_eq (v := slotEntryA st (.net n) (netOf st.c n).constPos) hW ho hcr.2.1, ← hpos, hS,
      ← ite_fin_val j L.pin_lt]
    exact const_row_sum _ _ _
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val = _
    rw [L.delta_b, hr, rangesEntry_net L C hn hg hi ho, delta_const_net L C hn hg hi hcr.1,
      ite_slot_eq (v := slotEntryB st (.net n) (netOf st.c n).constPos) hW ho hcr.2.2, ← hpos, hS,
      ← ite_fin_val j L.pin_lt]
    exact const_row_sum _ _ _

/-- **The pin's row reads the pin.** -/
theorem placed_pin (j : Fin (2 ^ (L.model regs).kLog)) :
    (L.model regs).A₀ (L.model regs).pin j = (if j = (L.model regs).pin then 1 else 0) ∧
      (L.model regs).B₀ (L.model regs).pin j = if j = (L.model regs).pin then 1 else 0 := by
  have hS : (L.model regs).pin = (⟨st.pin, L.pin_lt⟩ : Fin (2 ^ (L.model regs).kLog)) := rfl
  obtain ⟨rp, hrp, ip, hk, hc, hpin, hcr⟩ := L.pin
  have hoffW : (st.c.nets.getD ip default).2.constPos < 2 ^ rp.slotLog := by
    rw [L.slotLog rp hrp ip hk]; exact (L.fit ip).1
  have hp : st.pin = rp.slot 0 + (st.c.nets.getD ip default).2.constPos := by rw [hpin, start_eq_slot]
  constructor
  · show rangesEntry st (slotEntryA st) (L.model regs).pin.val j.val + deltaEntry st.da (L.model regs).pin.val j.val = _
    rw [hS]
    show rangesEntry st (slotEntryA st) st.pin j.val + deltaEntry st.da st.pin j.val = _
    rw [L.delta_pin, add_zero, ← ite_fin_val j L.pin_lt]
    rw [hp, rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1,
      (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryA st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW
        hcr.2.1]
    rfl
  · show rangesEntry st (slotEntryB st) (L.model regs).pin.val j.val + deltaEntry st.db (L.model regs).pin.val j.val = _
    rw [hS]
    show rangesEntry st (slotEntryB st) st.pin j.val + deltaEntry st.db st.pin j.val = _
    rw [L.delta_b, L.delta_pin, add_zero, ← ite_fin_val j L.pin_lt]
    rw [hp, rangesEntry_of_covers L.disj hrp (L.const_covers hrp hk hc) _ _, (slot_div 0 hoffW).1,
      (slot_div 0 hoffW).2, hk,
      ite_slot_eq (v := slotEntryB st (.net ip) (st.c.nets.getD ip default).2.constPos) (Nat.two_pow_pos _) hoffW
        hcr.2.2]
    rfl

omit L in
/-- The order's columns past the inputs have rows. -/
theorem order_row' {c : Circuit} {k : ℕ} (hk : k < (logical c).order.length - (logical c).nIn) :
    row c ((logical c).order.getD ((logical c).nIn + k) 0) ≠ none :=
  order_row (show (logical c).nIn + k < (order c).length by
    have : (logical c).order = order c := rfl; rw [← this]; omega) (Nat.le_add_right _ _)

/-- **A template instance's rows, placed by the accepted statement**, from the parse, the check and Δ at the input
bits a wire fills (`WiredDelta`): template instance `g`'s columns at their positions (`pos`), `one` at the pin
(`composePlace`). -/
theorem placement_instOf {pre : Array (String × Net)} (P : ParseFactsU tags pre st.c) (hok : NetsFit st.c)
    (C : CheckFacts st.c) (hD : WiredDelta st) {g : ℕ} (hg : g < st.c.g) :
    Placement (L.model regs) (instRows st.c)
      (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g) (inst_hbOf L regs P C hg)) := by
  refine placement_of_realizes ?_ ?_ ?_ ?_ ?_ ?_
  · intro k _ p hp r j hr
    exact (placed_comp L C hD hg (row_of_getD hp) r j (by rw [code_parts]; exact hr)).1
  · intro k _ p hp r j hr
    exact (placed_comp L C hD hg (row_of_getD hp) r j (by rw [code_parts]; exact hr)).2
  · intro k hk hnone r j hr
    have hne := order_row' hk
    have hrow : row st.c ((logical st.c).order.getD ((logical st.c).nIn + k) 0) = some none := by
      revert hnone hne
      show (row st.c _).getD none = none → _
      rcases row st.c ((logical st.c).order.getD ((logical st.c).nIn + k) 0) with _ | _ | q <;> simp
    exact (placed_const L regs hok C hg hrow r j (by rw [code_parts]; exact hr)).1
  · intro k hk hnone r j hr
    have hne := order_row' hk
    have hrow : row st.c ((logical st.c).order.getD ((logical st.c).nIn + k) 0) = some none := by
      revert hnone hne
      show (row st.c _).getD none = none → _
      rcases row st.c ((logical st.c).order.getD ((logical st.c).nIn + k) 0) with _ | _ | q <;> simp
    exact (placed_const L regs hok C hg hrow r j (by rw [code_parts]; exact hr)).2
  · exact fun j => (placed_pin L regs j).1
  · exact fun j => (placed_pin L regs j).2

include F in
/-- **A template instance's rows, placed by the accepted statement**: template instance `g`'s columns at their
positions (`pos`), `one` at the pin (`composePlace`). -/
theorem placement_inst {g : ℕ} (hg : g < st.c.g) :
    Placement (L.model regs) (instRows st.c)
      (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g) (inst_hb L F regs hg)) :=
  placement_instOf L regs F.parse F.nets F.check (wiredDelta L F) hg

end placement

end FlockSoundness.Discharge.Inst
