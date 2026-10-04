import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Discharge.ScopeG.Place.Flat
import FlockSoundness.Discharge.ScopeG.Place.Pos
import FlockSoundness.ExecLookup

set_option Elab.async false

/-!
# P1 piece: Δ at a VU's input and computed rows, in a flat statement

In a flat statement (`Layout st`, `Aliased.UnitFacts tags st`: Δ is `HmRow.delta`'s, the circuit untyped), at net `k`'s
slot `i` of VU `g` for `k ∈ vuNets`:

* `delta_inA` / `delta_inB`: at a wired input row (`x < netIn`, `wireSrcG` names `s`), the net's own row there plus Δ's
  pairs read exactly `s`: the own row is `[x]` (a wired column is a port bit), Δ's `(x, x)` cancels it, Δ's `(x, s)`
  stays. `B` also carries the slot's lookup share (`lookupAt`), which is 0 at input rows.
* `delta_own`: at every other row below the constant (a computed row, or an input row no wire copies of a net other
  than the unit), Δ writes nothing, so the row reads its net's rows (`slotRowG`).

**Twin of** `Layout.delta_unit_comp` (`ExecPlacement.lean`) for `delta_own`, at every net instead of the unit; and, for
the wired input rows, of the unit-only copy facts `Layout.delta_port` / `Aliased.copyAt` read as matrix entries instead of
values.

**Route.** Δ exactly: `delta_flat` (`FlatDelta`: the slot constants' `constPairs`, the rows' copies, `Flat.leafPairs`,
`Flat.cutPairs`, `Flat.wirePairs`). Count `(r, j)` in each part: `constPairs` are at constants, the row copies at
`sha512x3` / `hm96` slots, leaf and cut pairs at the unit's leaf and cut bits (never a wired port, nor another net's
column: `CheckFacts.wire_leaf`, `CheckFacts.cut_leaf`, `CheckFacts.disj`), and `wirePairs` gives `(dst, dst)` and
`(dst, src)` once per wired bit (`CheckFacts.wired`: every non-leaf input port wired exactly once, the wire `find?`
returns). The net's own row at a port bit is `[x]` (`Net.ofRows`'s `inputsOk`, or `Lookup.build`'s input rows;
`ParseFacts.nets`; `net_port_row`, `lookup_input_row`), and the ports' bits are distinct (`net_portBits_nodup`). At a
computed row, `Layout.delta`'s pairs past `constPairs` are input bits (`InBit`); an unwired input row of a net other
than the unit is in no port (`CheckFacts.wired` wires each of its ports), so no wire pair lands there.
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3

/-- **A lookup slot's input row reads itself**, on both sides, with no table share. -/
theorem lookup_input_row {nm : String} {tb : ByteArray} {n lo bits : ℕ} {net : Net}
    (h : Lookup.build nm tb n lo bits = .ok net) (hlog : net.unitLog ≤ 32) {x : ℕ} (hx : x < n) :
    net.a.row x = #[x] ∧ net.b.row x = #[x] ∧ ∀ j, lookupEntry net.lookup x j = 0 := by
  obtain ⟨hlo, hln, -, hn, -, d1, d2, r3, hd1, hd2, -, -, ⟨kk, hr3a, hr3b, -⟩, hA, hB, -, -, -, hlk⟩ :=
    build_spec h
  have I0 := inputRows_eq n
  have hI0a : (inputRows n).a.size = 128 := by rw [I0]; simp
  have hI0b : (inputRows n).b.size = 128 := by rw [I0]; simp
  have hlo0 : List.range lo ≠ [] := by simp; omega
  have hhi0 : (List.range (n - lo)).map (lo + ·) ≠ [] := by simp; omega
  obtain ⟨A1, -, -, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) lo (inputRows n) (List.range lo) hlo0 (by simp)
  rw [← hd1] at A1
  obtain ⟨A2, -, -, -⟩ := decode_spec (K := ZMod 2) (fun _ => 0) (n - lo) d1.1
    ((List.range (n - lo)).map (lo + ·)) hhi0 (by simp)
  rw [← hd2] at A2
  have PE := productRows_eq d2.2.1 bits d2.1
  have AP : Appends d2.1 (productRows d2.2.1 bits d2.1) := Appends.of_eq _ _ (by rw [PE]) (by rw [PE]) (by simp)
  have A3 : Appends (productRows d2.2.1 bits d2.1) r3 := Appends.of_eq _ _ hr3a hr3b (by simp)
  have FE := outputRows_eq d2.1.a.size d2.2.1.size bits r3
  have AF : Appends r3 (outputRows d2.1.a.size d2.2.1.size bits r3) :=
    Appends.of_eq _ _ (by rw [FE]) (by rw [FE]) (by simp)
  have e0 : (inputRows n).a.size = (inputRows n).b.size := by rw [hI0a, hI0b]
  have A12 := A1.trans A2
  have AF0 := A12.trans (AP.trans (A3.trans AF))
  have hx128 : x < 128 := by omega
  have e := AF0.getD e0 (by rw [hI0a]; exact hx128)
  have hFs := (AF0.size).1
  have hFe := (AF0.size).2 e0
  have hw : 2 ^ net.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
  have hK : x ≠ Lookup.KONST := by unfold Lookup.KONST; omega
  have hrow : (inputRows n).a.getD x #[] = #[x] ∧ (inputRows n).b.getD x #[] = #[x] := by
    rw [I0]; simp [Array.getD_eq_getD_getElem?, hx128, hx]
  refine ⟨?_, ?_, fun j => ?_⟩
  · rw [ofRows_row hA hw, fixKonst_getD _ _ _ (by omega), e.1, hrow.1]
    simp [hK]
  · rw [ofRows_row hB hw, fixKonst_getD _ _ _ (by omega), e.2, hrow.2]
    simp [hK]
  · have h12 := (A12.size).1
    rw [hlk]
    simp only [lookupEntry, ite_eq_right_iff]
    intro hc
    omega

variable {tags : Tags} {c : Circuit}

/-- A parsed circuit's net: built, or a lookup slot. -/
theorem net_cases (P : ParseFacts tags c) {k : ℕ} (hk : k < c.nets.size) :
    Built (c.nets.getD k default).2 ∨
      ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok (c.nets.getD k default).2 :=
  P.nets _ (getD_mem_toList hk)

/-- **A parsed net's input port bits are distinct columns.** -/
theorem net_portBits_nodup (P : ParseFacts tags c) {k : ℕ} (hk : k < c.nets.size) :
    (portBits (c.nets.getD k default).2.inGroups).Nodup := by
  rcases net_cases P hk with hb | ⟨nm, tb, n, lo, bits, hl⟩
  · exact Aliased.portBits_nodup hb
  · rw [(build_facts hl).2.1, Aliased.portBits_eq]
    simp [List.nodup_range']

/-- **A parsed net's input port bit reads itself**, on both sides, with no table share. -/
theorem net_port_row (P : ParseFacts tags c) {k : ℕ} (hk : k < c.nets.size)
    (hlog : (c.nets.getD k default).2.unitLog ≤ 32) {x : ℕ} (hx : x ∈ portBits (c.nets.getD k default).2.inGroups) :
    (c.nets.getD k default).2.a.row x = #[x] ∧ (c.nets.getD k default).2.b.row x = #[x] ∧
      ∀ j, lookupEntry (c.nets.getD k default).2.lookup x j = 0 := by
  rcases net_cases P hk with hb | ⟨nm, tb, n, lo, bits, hl⟩
  · have hlk := (built_spec hb).1
    obtain ⟨_, _, _, ig, _, ra, rb, ho⟩ := hb
    obtain ⟨-, ha, hb', -, -, hig, -⟩ := Net.ofRows_ok ho
    have hin := (Net.ofRows_inputs ho).2
    rw [hig, Aliased.portBits_eq] at hx
    obtain ⟨g, hg, hxr⟩ := List.mem_flatMap.1 hx
    rw [List.mem_range'_1] at hxr
    have hlt : x - g.col * WORD < g.bits.foldl (· + ·) 0 := by
      rw [← Array.foldl_toList, foldl_add_eq_sum]; omega
    obtain ⟨hra, hrb⟩ := hin g hg _ hlt
    rw [show g.col * WORD + (x - g.col * WORD) = x by omega] at hra hrb
    have hw : 2 ^ (c.nets.getD k default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) hlog
    refine ⟨by rw [ofRows_row ha hw, hra], by rw [ofRows_row hb' hw, hrb], fun j => ?_⟩
    rw [hlk]
    rfl
  · have hx' : x < n := by
      rw [(build_facts hl).2.1, Aliased.portBits_eq] at hx
      simpa using hx
    exact lookup_input_row hl hlog hx'


theorem count_shift_self (b x j : ℕ) :
    ((((#[x] : Array ℕ).toList.map (b + ·)).count j : ℕ) : ZMod 2) = if b + x = j then 1 else 0 := by
  by_cases h : b + x = j
  · subst h; simp
  · simp [h]

theorem zmod2_cancel (a b : Prop) [Decidable a] [Decidable b] :
    ((if a then 1 else 0 : ZMod 2)) + (((if a then 1 else 0) + (if b then 1 else 0) : ℕ) : ZMod 2) =
      ((if b then 1 else 0 : ℕ) : ZMod 2) := by
  by_cases ha : a <;> by_cases hb : b <;> simp [ha, hb] <;> decide

section counts

variable {tags : Tags} {st : Stmt} (L : Layout st) {k g i : ℕ}
include L

/-- Net `k`'s slot of VU `g` in its range. -/
theorem vuSlot_at (C : CheckFacts st.c) (hkn : k < st.c.nets.size) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)
    {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) :
    st.c.rangeOf k ∈ st.c.ranges.toList ∧ (st.c.rangeOf k).kind = .net k ∧
      g * st.c.perVu.getD k 0 + i < (st.c.rangeOf k).count ∧ x < 2 ^ (st.c.rangeOf k).slotLog := by
  obtain ⟨hm, hkd, -⟩ := rangeOf_spec (C.one_range k hkn)
  refine ⟨hm, hkd, by rw [C.count _ hm _ hkd]; exact Rows.mul_add_lt hg hi, ?_⟩
  rw [L.slotLog _ hm k hkd]; exact lt_of_le_of_lt hx (L.fit k).1

/-- A column of another net's slot than the unit's is none of the unit slots' columns. -/
theorem ne_unit_col (C : CheckFacts st.c) (hkn : k < st.c.nets.size) (hku : k ≠ st.c.unitNet) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) {s₀ b : ℕ}
    (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) :
    st.c.slotCol k g i + x ≠ st.c.unitRange.slot s₀ + b := fun he => by
  obtain ⟨hm, hkd, hs, ha⟩ := vuSlot_at L C hkn hg hi hx
  exact hku (L.eq_unit hm hkd hs ha hs₀ hb he).1

/-- A row circuit's bit Δ fills is no column of a VU net's slot. -/
theorem rowBit_ne (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hk : k ∈ vuNets st.c) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) {y : ℕ}
    (hy : Flat.RowBit st.c y) : y ≠ st.c.slotCol k g i + x := by
  obtain ⟨hkn, hks, hkh⟩ := mem_vuNets.1 hk
  intro heq
  have hws := rowSlots_spec st.c.ports
  have hall := rowSlots_all st.c.ports
  rcases hy with ⟨g', hg', pp, hpp, k', hk', i', hi', tt, htt, he⟩ | ⟨g', hg', pp, hpp, i', hi', tt, htt, he⟩
  · obtain ⟨h1, h2⟩ := hall _ (getElem_mem_toList' hpp) _ (getElem_mem_toList' hk')
    have hidx : 2 * ((HmRow.rowSlots st.c.ports).2[pp]!)[k']!.2 + i' < 6 := by omega
    refine ne_net_at L P C P.shaNet hkn (Ne.symm hks) hg' (by rw [C.perVu_sha]; exact h1)
      (getElem_mem_toList' (by rw [C.sha_in.1]; exact hidx)) ?_ hg hi hx (he ▸ heq)
    rw [C.sha_in.2 _ hidx, sha_width _ _ hi']
    exact htt
  · obtain ⟨hil, hw⟩ := hm_width C.hm_in hi'
    refine ne_net_at L P C P.hmNet hkn (Ne.symm hkh) hg' ?_ (getElem_mem_toList' hil)
      (by rw [hw]; exact htt) hg hi hx (he ▸ heq)
    rw [C.perVu_hm, ← hws]
    exact lt_of_lt_of_le hpp (le_nextPow2 _)

/-- No leaf pair lands in another net's slot than the unit's. -/
theorem leafPairs_zero (C : CheckFacts st.c) (hkn : k < st.c.nets.size) (hku : k ≠ st.c.unitNet) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) (j : ℕ) :
    (Flat.leafPairs st.c).count (st.c.slotCol k g i + x, j) = 0 := by
  unfold Flat.leafPairs
  refine count_flatMap_zero fun k' hk' => count_flatMap_zero fun jj hjj => count_flatMap_zero fun tt htt =>
    count_pair_ne fun he => ?_
  have hk'' := List.mem_range.1 hk'
  have hupv : 0 < st.c.upv := by
    rcases Nat.eq_zero_or_pos st.c.upv with h0 | h0
    · rw [L.unit_count, h0, Nat.mul_zero] at hk''; omega
    · exact h0
  have hmod := Nat.mod_lt k' hupv
  have hlt : k' % st.c.upv < st.c.leavesIn.size := by rw [C.leaves.1]; exact hmod
  have hli : (st.c.leavesIn.getD (k' % st.c.upv) #[]).size = (st.c.unit.inGroups.getD 0 default).bits.size := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some,
      ← getElem!_pos st.c.leavesIn _ hlt]
    exact C.leaves.2 _ hmod
  have hjj' : jj < (st.c.unit.inGroups.getD 0 default).bits.size := by rw [← hli]; exact List.mem_range.1 hjj
  obtain ⟨hjs, hcol, hw⟩ := firstGroup_port C.unit_groups.1 hjj'
  have h16 : (st.c.unit.inGroups.getD 0 default).bits[jj]! = 16 :=
    C.unit_groups.2 _ (getElem_mem_toList' (xs := (st.c.unit.inGroups.getD 0 default).bits) hjj')
  have htt' : tt < (portCols st.c.unit.inGroups)[jj]!.2 := by rw [hw, h16]; exact List.mem_range.1 htt
  refine ne_unit_col L C hkn hku hg hi hx hk'' (lt_of_lt_of_le (L.port_lt hjs htt') (L.fit st.c.unitNet).2) ?_
  rw [← he]
  unfold Flat.leafCol
  rw [hcol]
  omega

/-- No cut pair lands in another net's slot than the unit's. -/
theorem cutPairs_zero (C : CheckFacts st.c) (hkn : k < st.c.nets.size) (hku : k ≠ st.c.unitNet) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) (j : ℕ) :
    (Flat.cutPairs st.c).count (st.c.slotCol k g i + x, j) = 0 := by
  unfold Flat.cutPairs
  refine count_flatMap_zero fun g' hg' => count_flatMap_zero fun e he => count_flatMap_zero fun tt htt =>
    count_pair_ne fun heq => ?_
  have hg'' := List.mem_range.1 hg'
  obtain ⟨he1, he2⟩ := C.cuts e he
  have hs₀ : g' * st.c.upv + e.1 < st.c.unitRange.count := by rw [L.unit_count]; exact Rows.mul_add_lt hg'' he1
  have hpv := L.perVu_eq C hs₀
  have htt' : tt < (portCols st.c.unit.inGroups)[e.2.1]!.2 := List.mem_range.1 htt
  refine ne_unit_col L C hkn hku hg hi hx hs₀ (lt_of_lt_of_le (L.port_lt he2 htt') (L.fit st.c.unitNet).2) ?_
  rw [← heq]
  unfold Flat.cutCol Circuit.slotCol
  rw [hpv, Nat.add_assoc]
  rfl

/-- A wire pair's row in net `k`'s slot of VU `g` is a bit of one of its ports, in that slot. -/
theorem wireDst_at (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hk : k ∈ vuNets st.c) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos) {g' : ℕ} {w : Wire}
    {tt : ℕ} (hg' : g' < st.c.g) (hw : w ∈ st.c.wires.toList) (htt : tt < Flat.wireWidth st.c w)
    (he : Flat.wireDst st.c g' w + tt = st.c.slotCol k g i + x) :
    w.dst.1 = k ∧ g' = g ∧ w.dst.2.1 = i ∧ w.dst.2.2 < (portCols (st.c.nets.getD k default).2.inGroups).size ∧
      (portCols (st.c.nets.getD k default).2.inGroups)[w.dst.2.2]!.1 + tt = x ∧
      tt < (portCols (st.c.nets.getD k default).2.inGroups)[w.dst.2.2]!.2 := by
  obtain ⟨hkn, -, -⟩ := mem_vuNets.1 hk
  obtain ⟨hd1, hd2, hd3⟩ := C.wires w hw
  by_cases hdk : w.dst.1 = k
  · obtain ⟨⟨dn, di, dp⟩, src⟩ := w
    simp only at hdk hd2 hd3 htt he ⊢
    subst hdk
    have hW : Flat.wireWidth st.c ⟨(dn, di, dp), src⟩ = (portCols (st.c.nets.getD dn default).2.inGroups)[dp]!.2 := by
      unfold Flat.wireWidth; exact congrArg Prod.snd (Layout.getD_portCols hd3)
    rw [hW] at htt
    obtain ⟨hm, hkd, hs, ha⟩ := vuSlot_at L C hkn hg hi hx
    have hq' := (nets_ok P dn).1.2.2 _ (getElem_mem_toList' hd3)
    have ha' : (portCols (st.c.nets.getD dn default).2.inGroups)[dp]!.1 + tt < 2 ^ (st.c.rangeOf dn).slotLog := by
      rw [L.slotLog _ hm dn hkd]; have := (L.fit dn).2; omega
    have hcol : Flat.wireDst st.c g' ⟨(dn, di, dp), src⟩ + tt = (st.c.rangeOf dn).slot (g' * st.c.perVu.getD dn 0 + di) +
        ((portCols (st.c.nets.getD dn default).2.inGroups)[dp]!.1 + tt) := by
      unfold Flat.wireDst Circuit.slotCol
      simp only
      rw [Layout.getD_portCols hd3, Nat.add_assoc]
    rw [hcol] at he
    obtain ⟨h1, h2⟩ := slot_add_inj ha' ha he
    have h3 := (vu_split hd2).1 h1
    have h4 := (vu_split (g := g) (s₀ := g * st.c.perVu.getD dn 0 + i) hi).1 rfl
    exact ⟨rfl, h3.1.trans h4.1.symm, h3.2.trans h4.2.symm, hd3, h2, htt⟩
  · exact absurd he (ne_net_at L P C hd1 hkn hdk hg' hd2 (getD_mem_toList hd3) htt hg hi hx)

/-- No wire pair lands at a column of net `k`'s slot in none of its ports. -/
theorem wirePairs_zero (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hk : k ∈ vuNets st.c) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {x : ℕ} (hx : x ≤ (st.c.nets.getD k default).2.constPos)
    (hno : ∀ p < (portCols (st.c.nets.getD k default).2.inGroups).size,
      ¬ ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 ≤ x ∧
        x < (portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 +
          (portCols (st.c.nets.getD k default).2.inGroups)[p]!.2)) (j : ℕ) :
    (Flat.wirePairs st.c).count (st.c.slotCol k g i + x, j) = 0 := by
  unfold Flat.wirePairs
  refine count_flatMap_zero fun g' hg' => count_flatMap_zero fun w hw => count_flatMap_zero fun tt htt =>
    count_pair_ne fun he => ?_
  obtain ⟨-, -, -, hp, h1, h2⟩ := wireDst_at L P C hk hg hi hx (List.mem_range.1 hg') hw (List.mem_range.1 htt) he
  exact hno _ hp ⟨by omega, by omega⟩

/-- **The wire pairs at a port bit of net `k`'s slot of VU `g`**: the copy pair of each wire into its port. -/
theorem wirePairs_at (P : ParseFacts tags st.c) (C : CheckFacts st.c) (hk : k ∈ vuNets st.c) (hg : g < st.c.g)
    (hi : i < st.c.perVu.getD k 0) {p t : ℕ} (hp : p < (portCols (st.c.nets.getD k default).2.inGroups).size)
    (ht : t < (portCols (st.c.nets.getD k default).2.inGroups)[p]!.2)
    (hx : (portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t ≤ (st.c.nets.getD k default).2.constPos)
    (j : ℕ) :
    (Flat.wirePairs st.c).count (st.c.slotCol k g i + ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t), j) =
      ((st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p))).flatMap fun w =>
        [(st.c.slotCol k g i + ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t),
          st.c.slotCol k g i + ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t)),
          (st.c.slotCol k g i + ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t),
            Flat.wireSrc st.c g w + t)]).count
        (st.c.slotCol k g i + ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t), j) := by
  obtain ⟨hkn, -, -⟩ := mem_vuNets.1 hk
  have hnd := net_portBits_nodup P hkn
  unfold Flat.wirePairs
  refine count_vu_items _ _ _ (fun g' w tt => Flat.wireDst st.c g' w + tt) (fun g' w tt => Flat.wireSrc st.c g' w + tt)
    (fun w => w.dst = (k, i, p)) hg (fun g' hg' w hw tt htt => ?_) (fun w _ hF => ?_)
  · constructor
    · intro he
      obtain ⟨hdk, hgg, hdi, hdp, h1, h2⟩ := wireDst_at L P C hk hg hi hx hg' hw htt he
      obtain ⟨hpp, htt'⟩ := portBits_inj hnd hdp hp h2 ht h1
      exact ⟨hgg, Prod.ext hdk (Prod.ext hdi hpp), htt'⟩
    · rintro ⟨rfl, hF, rfl⟩
      show Flat.wireDst st.c g' w + tt = _
      unfold Flat.wireDst
      rw [hF]
      simp only
      rw [Layout.getD_portCols hp, Nat.add_assoc]
  · unfold Flat.wireWidth
    rw [hF]
    show t < ((portCols (st.c.nets.getD k default).2.inGroups).getD p default).2
    rw [Layout.getD_portCols hp]
    exact ht

end counts

section delta

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st) {k g i : ℕ}
  (hk : k ∈ vuNets st.c) (hg : g < st.c.g) (hi : i < st.c.perVu.getD k 0)

include L F hk hg hi

omit L hg in
/-- `wireSrcG` named a column: the port its bit is in, and the one wire into that port. -/
theorem wireSrcG_some {x s : ℕ} (hw : wireSrcG st.c g k i x = some s) :
    ∃ p t, p < (portCols (st.c.nets.getD k default).2.inGroups).size ∧
      t < (portCols (st.c.nets.getD k default).2.inGroups)[p]!.2 ∧
      x = (portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t ∧ ∃ w,
      (st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p))) = [w] ∧ Flat.wireSrc st.c g w + t = s ∧
      (st.c.leafCuts.toList.filter fun e => (st.c.unitNet, e.1, e.2.1) = (k, i, p)).length = 0 := by
  obtain ⟨hkn, hks, hkh⟩ := mem_vuNets.1 hk
  unfold wireSrcG at hw
  simp only at hw
  split at hw
  · cases hw
  rename_i p hfp
  split at hw
  · cases hw
  rename_i w hfw
  have hp : p < (portCols (st.c.nets.getD k default).2.inGroups).size := List.mem_range.1 (List.mem_of_find?_eq_some hfp)
  have hpx := List.find?_some hfp
  simp only [decide_eq_true_eq] at hpx
  have hwm := List.mem_of_find?_eq_some hfw
  have hwd : w.dst = (k, i, p) := by simpa using List.find?_some hfw
  have hone := F.check.wired k hkn hks hkh i hi p (by
    by_cases hku : k = st.c.unitNet
    · simp only [hku, ite_true]
      have := F.check.wire_leaf w hwm (by rw [hwd]; exact hku)
      rw [hwd] at this
      exact this
    · simp only [hku, ite_false]; exact Nat.zero_le _) hp
  have hmemf : w ∈ st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p)) :=
    List.mem_filter.2 ⟨hwm, by simp [hwd]⟩
  have hpos := List.length_pos_of_mem hmemf
  obtain ⟨w', hw'⟩ := List.length_eq_one_iff.1 (show (st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p))).length = 1 by omega)
  rw [hw', List.mem_singleton] at hmemf
  subst hmemf
  refine ⟨p, x - (portCols (st.c.nets.getD k default).2.inGroups)[p]!.1, hp, by omega, by omega, w, hw', ?_, by omega⟩
  rw [← Option.some.inj hw]
  rfl

/-- **Δ at a wired input row**: its one copy pair, `(x, x)` and `(x, s)`. -/
theorem delta_wired {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s)
    (j : ℕ) :
    st.da.toList.count (st.c.slotCol k g i + x, j) =
      (if st.c.slotCol k g i + x = j then 1 else 0) + if s = j then 1 else 0 := by
  obtain ⟨hkn, hks, hkh⟩ := mem_vuNets.1 hk
  have P := F.parse
  have C := F.check
  have hxc : x < (st.c.nets.getD k default).2.constPos := lt_of_lt_of_le hx (min_le_right _ _)
  obtain ⟨hm, hkd, hs, ha⟩ := vuSlot_at L C hkn hg hi hxc.le
  obtain ⟨p, t, hp, ht, rfl, w, hwf, hsrc, hc0⟩ := wireSrcG_some F hk hi hw
  obtain ⟨rows, hd, hrows⟩ := F.delta
  have h0 : (constPairs st.c st.pin).count (st.c.slotCol k g i +
      ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t), j) = 0 :=
    List.count_eq_zero.2 (L.constPairs_not_below hm hkd hs hxc j)
  rw [hd]
  simp only [List.count_append]
  rw [h0, List.count_eq_zero.2 (fun hmem => rowBit_ne L P C hk hg hi hxc.le (hrows _ hmem) rfl),
    wirePairs_at L P C hk hg hi hp ht hxc.le j, hwf, List.flatMap_cons, List.flatMap_nil, List.append_nil, count_pair_eq,
    hsrc]
  have hl : (Flat.leafPairs st.c).count (st.c.slotCol k g i +
      ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t), j) = 0 := by
    by_cases hku : k = st.c.unitNet
    · subst hku
      have hs₀ : g * st.c.perVu.getD st.c.unitNet 0 + i < st.c.unitRange.count := hs
      have hwm : w ∈ st.c.wires.toList := List.mem_of_mem_filter (by rw [hwf]; exact List.mem_singleton_self _)
      have hwd : w.dst = (st.c.unitNet, i, p) := by
        have := (List.mem_filter.1 (by rw [hwf]; exact List.mem_singleton_self w)).2
        simpa using this
      have hpl := C.wire_leaf w hwm (by rw [hwd])
      rw [hwd] at hpl
      have := L.leafPairs_count C (net_portBits_nodup P hkn) hs₀ hp ht j
      have hnl : ¬ p < (st.c.unit.inGroups.getD 0 default).bits.size := by simp only at hpl; omega
      simp only [hnl, ite_false] at this
      exact this
    · exact leafPairs_zero L C hkn hku hg hi hxc.le j
  have hc : (Flat.cutPairs st.c).count (st.c.slotCol k g i +
      ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 + t), j) = 0 := by
    by_cases hku : k = st.c.unitNet
    · subst hku
      have hs₀ : g * st.c.perVu.getD st.c.unitNet 0 + i < st.c.unitRange.count := hs
      have hpv := L.perVu_eq C hs₀
      have := L.cutPairs_count C (net_portBits_nodup P hkn) hs₀ hp ht j
      have hmod : (g * st.c.perVu.getD st.c.unitNet 0 + i) % st.c.upv = i := by
        rw [hpv] at hi ⊢; exact ((vu_split hi).1 rfl).2.symm
      rw [hmod] at this
      have hf0 : (st.c.leafCuts.toList.filter fun e => decide (e.1 = i ∧ e.2.1 = p)) = [] := by
        rw [List.length_eq_zero_iff, List.filter_eq_nil_iff] at hc0
        rw [List.filter_eq_nil_iff]
        intro e he
        have := hc0 e he
        simp only [decide_eq_true_eq, Prod.mk.injEq, true_and, not_and] at this ⊢
        exact this
      rw [hf0] at this
      exact this
    · exact cutPairs_zero L C hkn hku hg hi hxc.le j
  rw [hl, hc]
  omega

/-- A wired input row is a port bit, and so a self row with no table share. -/
theorem wired_row {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s) :
    (st.c.nets.getD k default).2.a.row x = #[x] ∧ (st.c.nets.getD k default).2.b.row x = #[x] ∧
      ∀ j, lookupEntry (st.c.nets.getD k default).2.lookup x j = 0 := by
  obtain ⟨hkn, -, -⟩ := mem_vuNets.1 hk
  have hxc : x < (st.c.nets.getD k default).2.constPos := lt_of_lt_of_le hx (min_le_right _ _)
  obtain ⟨hm, hkd, hs, ha⟩ := vuSlot_at L F.check hkn hg hi hxc.le
  obtain ⟨p, t, hp, ht, rfl, -⟩ := wireSrcG_some F hk hi hw
  have hlog : (st.c.nets.getD k default).2.unitLog ≤ 32 := by
    rw [← L.slotLog _ hm k hkd]
    exact le_trans (AtSlot.slotLog_le L.bound hm (show AtSlot st.c k _ _ _ from ⟨hs, ha, rfl⟩)) L.kLog
  refine net_port_row F.parse hkn hlog (List.mem_flatMap.2 ⟨_, getElem_mem_toList' hp, ?_⟩)
  exact List.mem_range'_1.2 ⟨by omega, by omega⟩

/-- **`A` at a wired VU input row**: the net's own row plus Δ's pairs read exactly the wire's source column. -/
theorem delta_inA {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.a.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        deltaEntry st.da r.val j.val = (([s].count j.val : ℕ) : ZMod 2) := by
  rw [(wired_row L F hk hg hi hx hw).1, count_shift_self, deltaEntry_eq_count, hr, delta_wired L F hk hg hi hx hw,
    List.count_singleton, zmod2_cancel]
  simp only [beq_iff_eq]

/-- **`B` at a wired VU input row**: the net's own row, the slot's lookup share and Δ's pairs read exactly the wire's
source column. -/
theorem delta_inB {x s : ℕ} (hx : x < netIn (st.c.nets.getD k default).2) (hw : wireSrcG st.c g k i x = some s)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    ((((st.c.nets.getD k default).2.b.row x).toList.map (st.c.slotCol k g i + ·)).count j.val : ZMod 2) +
        lookupAt st.c k g i x j.val + deltaEntry st.db r.val j.val = (([s].count j.val : ℕ) : ZMod 2) := by
  have hl : lookupAt st.c k g i x j.val = 0 := by
    unfold lookupAt
    split_ifs
    · exact (wired_row L F hk hg hi hx hw).2.2 _
    · rfl
  rw [hl, add_zero, L.delta_b, (wired_row L F hk hg hi hx hw).2.1, count_shift_self, deltaEntry_eq_count, hr,
    delta_wired L F hk hg hi hx hw, List.count_singleton, zmod2_cancel]
  simp only [beq_iff_eq]

/-- **Δ writes nothing at a VU's other rows**: a computed row, or an unwired input row of a net other than the unit
(`Layout.delta_unit_comp` at every net of `vuNets`). -/
theorem delta_own {x : ℕ} (hx : x < (st.c.nets.getD k default).2.constPos)
    (hno : x < netIn (st.c.nets.getD k default).2 → wireSrcG st.c g k i x = none ∧ k ≠ st.c.unitNet)
    (r j : Fin (2 ^ st.c.kLog)) (hr : r.val = st.c.slotCol k g i + x) :
    deltaEntry st.da r.val j.val = 0 ∧ deltaEntry st.db r.val j.val = 0 := by
  obtain ⟨hkn, hks, hkh⟩ := mem_vuNets.1 hk
  have P := F.parse
  have C := F.check
  obtain ⟨hm, hkd, hs, ha⟩ := vuSlot_at L C hkn hg hi hx.le
  suffices h : deltaEntry st.da r.val j.val = 0 from ⟨h, by rw [L.delta_b]; exact h⟩
  rw [hr, deltaEntry_eq_count]
  by_cases hlt : x < netIn (st.c.nets.getD k default).2
  · obtain ⟨hnone, hku⟩ := hno hlt
    have hnop : ∀ p < (portCols (st.c.nets.getD k default).2.inGroups).size,
        ¬ ((portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 ≤ x ∧
          x < (portCols (st.c.nets.getD k default).2.inGroups)[p]!.1 +
            (portCols (st.c.nets.getD k default).2.inGroups)[p]!.2) := by
      intro p hp hpx
      unfold wireSrcG at hnone
      simp only at hnone
      split at hnone
      · rename_i hf
        rw [List.find?_eq_none] at hf
        exact hf p (List.mem_range.2 hp) (by simpa using hpx)
      rename_i p0 hfp
      have hp0 := List.mem_range.1 (List.mem_of_find?_eq_some hfp)
      have hone := C.wired k hkn hks hkh i hi p0 (by simp only [hku, ite_false]; exact Nat.zero_le _) hp0
      have hc0 : (st.c.leafCuts.toList.filter fun e => (st.c.unitNet, e.1, e.2.1) = (k, i, p0)).length = 0 := by
        rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
        intro e _
        simp only [decide_eq_true_eq, Prod.mk.injEq, not_and]
        intro h
        exact absurd h.symm hku
      obtain ⟨w, hwf⟩ := List.length_eq_one_iff.1
        (show (st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p0))).length = 1 by omega)
      have hwm := List.mem_filter.1 (show w ∈ st.c.wires.toList.filter fun w => decide (w.dst = (k, i, p0)) by
        rw [hwf]; exact List.mem_singleton_self w)
      split at hnone
      · rename_i hf
        rw [List.find?_eq_none] at hf
        exact hf w hwm.1 (by simpa using hwm.2)
      · cases hnone
    obtain ⟨rows, hd, hrows⟩ := F.delta
    have h0 : (constPairs st.c st.pin).count (st.c.slotCol k g i + x, j.val) = 0 :=
      List.count_eq_zero.2 (L.constPairs_not_below hm hkd hs hx j.val)
    rw [hd]
    simp only [List.count_append]
    rw [h0, List.count_eq_zero.2 (fun hmem => rowBit_ne L P C hk hg hi hx.le (hrows _ hmem) rfl),
      leafPairs_zero L C hkn hku hg hi hx.le, cutPairs_zero L C hkn hku hg hi hx.le,
      wirePairs_zero L P C hk hg hi hx.le hnop]
    rfl
  · have hge : (st.c.nets.getD k default).2.inWords * WORD ≤ x := by
      unfold netIn at hlt; omega
    obtain ⟨rest, hsplit, hrest⟩ := L.delta
    have h0 : (constPairs st.c st.pin).count (st.c.slotCol k g i + x, j.val) = 0 :=
      List.count_eq_zero.2 (L.constPairs_not_below hm hkd hs hx j.val)
    rw [hsplit, List.count_append, h0, List.count_eq_zero.2 (fun hmem => L.not_inBit hm hkd hs ha hge (hrest _ hmem))]
    rfl

end delta

end FlockSoundness.Discharge.ScopeG
