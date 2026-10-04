import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.ScopeW.CopiesAt
import FlockSoundness.Discharge.HiddenExec.Copies

set_option Elab.async false

/-!
# Δ's copies at the widened scope's rows: frozen statements

The row lemmas read Δ only at a row net's slot's bits, and there every statement the headlines take has the same Δ:
`HmRow.delta`'s pairs, plus pairs (`extra`) only at bits `HmRow.delta` never writes, filtered only at bits outside the
row nets (`RowDelta`). Three instances:

* `rowDelta_flat`: `Stmt.setupH`'s Δ, `HmRow.delta`'s;
* `rowDelta_hidden`: `Stmt.setupHidden`'s, `HmOut.delta`'s (`HmRow.delta`'s and `Hidden.outPairs`, `hidden_delta'`),
  whose pairs at the output rows' message bits `HmRow.delta` never writes (`outPairs_freshW`, `flat_count_msgW`);
* `rowDelta_hmIn`: a statement with the verifier's public inputs, `HmIn.delta`'s, which is `HmOut.delta`'s with no public
  port (`hmIn_delta_empty`) and otherwise filters only public leaves' bits, none of a row net's slot (`hpub`, which
  ZkPubIn's `publicCol_rowNet` gives).

So each copy lemma is stated once, at `RowDelta`, with the twin's circuit facts (`Aliased.UnitFacts` at
`Hidden.flatTwin st`, which reads only the circuit and its pin) as ZkPubIn's are; the plain forms are corollaries
(`twin_of_flat`, `inBit_copyLW_flat`, `row_copies_flatW_flat`).

* `inBit_copyLW`: an input bit of a row net's slot holds its one copy of the rows' copies and wires (`rowCopiesW`);
* `row_copies_flatW`: VU `g`'s row `p`'s copies, for any row, read off `HmRow.chainStart` and `HmRow.msgBit`;
* `row_copies_wholeW`, `row_copies_shortW`, `row_copies_segW`: what they give at each shape;
* `msg_copyW`: an output row's message bit holds its copy, for output rows of every shape (`OutRowsS`).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm Compose Flock.DeriveAll Types.Dag

/-! ## What the row lemmas read of Δ -/

/-- **Δ at the rows**: `HmRow.delta`'s pairs and `extra`'s, filtered by `keep` at their row; `extra` writes no row that
`HmRow.delta` writes, and `keep` keeps every bit of a row net's slot. -/
def RowDelta (st : Stmt) (extra : List (ℕ × ℕ)) (keep : ℕ → Bool) : Prop :=
  st.da = ((HmRow.delta st.c st.pin).1 ++ extra.toArray).filter (fun e => keep e.1) ∧
    (∀ e ∈ extra, ∀ j, (HmRow.delta st.c st.pin).1.toList.count (e.1, j) = 0) ∧
    ∀ {n s off x : ℕ}, (n = st.c.shaNet ∨ n = st.c.hmNet) → AtSlot st.c n s off x → keep x = true

/-- `Stmt.setupH`'s Δ. -/
theorem rowDelta_flat {st : Stmt} (hda : st.da = (HmRow.delta st.c st.pin).1) : RowDelta st [] (fun _ => true) :=
  ⟨by rw [hda, Array.filter_eq_self.2 fun _ _ => rfl]; simp, by simp, fun _ _ => rfl⟩

/-- **`HmIn.delta` with no public port is `HmOut.delta`.** -/
theorem hmIn_delta_empty (c : Circuit) {pi : HmIn.Ports} (pin : ℕ) (h : pi.ports.isEmpty = true) :
    HmIn.delta c pi pin = HmOut.delta c pin := by
  simp [HmIn.delta, h]

/-- An untyped statement whose Δ is `HmRow.delta`'s is its own flat twin. -/
theorem flatTwin_eq {st : Stmt} (L : Layout st) (ht : st.c.typed = none)
    (hda : st.da = (HmRow.delta st.c st.pin).1) : Hidden.flatTwin st = st := by
  have hdb : st.db = (HmRow.delta st.c st.pin).2 := by rw [L.delta_b, hda, delta_snd _ _ ht]
  rw [Hidden.flatTwin, ← hda, ← hdb]

/-- **The plain forms' twin facts**: at a statement whose Δ is `HmRow.delta`'s, its own. -/
theorem twin_of_flat {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
    (hda : st.da = (HmRow.delta st.c st.pin).1) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) ∧ Layout (Hidden.flatTwin st) := by
  rw [flatTwin_eq L F.parse.typed hda]
  exact ⟨F, L⟩

/-- **The output rows `msg_copyW` reads**: every row of a widened shape (`scopeOkW_ports`), each output row's port after
the input rows. -/
def OutRowsS (c : Circuit) : Prop :=
  (∀ p ∈ c.ports.toList, rowShapeW p = true) ∧ ∀ j < c.outRowPorts.size, HmOut.nIn c + j < c.ports.size

/-- Δ's count at a bit `keep` keeps: `HmRow.delta`'s and `extra`'s. -/
theorem rowDelta_count {st : Stmt} {extra : List (ℕ × ℕ)} {keep : ℕ → Bool} (hΔ : RowDelta st extra keep) {x : ℕ}
    (hx : keep x = true) (i : ℕ) :
    st.da.toList.count (x, i) = (HmRow.delta st.c st.pin).1.toList.count (x, i) + extra.count (x, i) := by
  rw [hΔ.1, Array.toList_filter, List.count_filter (p := fun e : ℕ × ℕ => keep e.1) (a := (x, i)) hx,
    Array.toList_append, List.count_append, List.toList_toArray]

/-- `Hidden.outPairs`'s copies: VU `g`'s output row `j`'s message bit from `Hidden.outSrc`. -/
def outCopiesR (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outRowPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).map fun t =>
        (Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), Hidden.outSrc c g j k t)

theorem outPairs_eqR (c : Circuit) : Hidden.outPairs c = (outCopiesR c).flatMap pair2 := by
  simp only [Hidden.outPairs, outCopiesR, List.flatMap_assoc, List.flatMap_map, pair2]

/-- **`Hidden.outPairs`'s one copy at an output row's message bit**, at output rows of any widened shape. -/
theorem filter_outCopiesR {tags : Tags} {c : Circuit} (R : RowRanges c) (P : ParseFacts tags c) (hrows : OutRowsS c)
    {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    (outCopiesR c).filter (fun e => e.1 = Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
      [(Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), Hidden.outSrc c g j k t)] := by
  have hw : ∀ j', (c.ports.getD (HmOut.nIn c + j') default).words = c.ports[HmOut.nIn c + j']!.words := fun j' => by
    rw [Hidden.getElem!_eq_getD']
  have hinj : ∀ {g' j' k' t' : ℕ}, g' < c.g → j' < c.outRowPorts.size →
      k' < (c.ports.getD (HmOut.nIn c + j') default).words → t' < 16 →
      Hidden.deltaMsgCol c g' (HmOut.nIn c + j') (2 * k' + t' / 8) (t' % 8) =
        Hidden.deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) →
      g' = g ∧ j' = j ∧ k' = k ∧ t' = t := by
    intro g' j' k' t' hg' hj' hk' ht' h
    have hp := hrows.2 j hj
    have hp' := hrows.2 j' hj'
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_injW R P hg' hp' (hrows.1 _ (getElem_mem_toList' hp'))
      (by rw [← hw]; omega) (Nat.mod_lt _ (by omega)) hg hp (hrows.1 _ (getElem_mem_toList' hp))
      (by rw [← hw]; omega) (Nat.mod_lt _ (by omega)) h
    exact ⟨h1, by omega, by omega, by omega⟩
  unfold outCopiesR
  rw [filter_flatMap2_single _ _ _ _ g j hg hj fun g' hg' j' hj' hne e he hex => by
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at he
    obtain ⟨k', hk', t', ht', rfl⟩ := he
    obtain ⟨h1, h2, -⟩ := hinj hg' hj' hk' ht' hex
    exact hne ⟨h1, h2⟩]
  rw [filter_flatMap_single _ List.nodup_range _ _ k (List.mem_range.2 hk) fun k' hk' hne =>
    filter_nil_of _ _ fun e he hex => by
      simp only [List.mem_map, List.mem_range] at he
      obtain ⟨t', ht', rfl⟩ := he
      exact hne (hinj hg hj (List.mem_range.1 hk') ht' hex).2.2.1]
  rw [List.map_eq_flatMap, filter_flatMap_single _ List.nodup_range _ _ t (List.mem_range.2 ht) fun t' ht' hne =>
    filter_nil_of _ _ fun e he hex => by
      simp only [List.mem_singleton] at he
      subst he
      exact hne (hinj hg hj hk (List.mem_range.1 ht') hex).2.2.2]
  simp

section Count

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
include L F

/-- **Δ's pairs at a row net's input bit `x`**, for rows of any shape: `(x, x), (x, src)` for each copy `(x, src)` of
the rows' copies or the rows' wires there; the slot constants, the leaf and cut bits and the other wires write nothing
there. -/
theorem flatW_count {n s off x : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
    {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD) (j : ℕ) :
    (flatPairsW st.c st.pin).count (x, j) =
      (((rowCopiesW st.c st.pin ++
          wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
        (fun e => e.1 = x)).map fun e => (if j = x then 1 else 0) + (if j = e.2 then 1 else 0)).sum := by
  have R := RowRanges.ofL L F.parse F.check
  have P : ParseFacts tags st.c := F.parse
  have C : CheckFacts st.c := F.check
  have hrw : RowWires st.c := P.rowWires
  have hnn := R.rowNet_lt hn
  obtain ⟨hm, hk, -, -⟩ := R.range hnn
  obtain ⟨-, -, -, -, hcp, -, -, hin, -⟩ := Net.ofRows_ok hl
  have hne_at : ∀ {n' s' off' : ℕ}, n' < st.c.nets.size → n' ≠ n → AtSlot st.c n' s' off' x → False :=
    fun hn' hne hat' => by
      obtain ⟨hm1, hk1, -⟩ := rangeOf_spec (C.one_range n hnn)
      obtain ⟨hm2, hk2, -⟩ := rangeOf_spec (C.one_range _ hn')
      exact hne (hat'.inj L.disj hm2 hk2 hm1 hk1 hat).1
  have hun : st.c.unitNet ≠ n := by
    rcases hn with rfl | rfl
    · exact F.sha
    · exact F.hm
  obtain ⟨hum, huk, huc⟩ := unit_range_facts P C
  have hunit : ∀ s₀ b, s₀ < st.c.unitRange.count → b < 2 ^ st.c.unitRange.slotLog →
      st.c.unitRange.slot s₀ + b ≠ x := fun s₀ b hs₀ hb he =>
    hne_at P.unitNet hun (show AtSlot st.c st.c.unitNet s₀ b x from ⟨hs₀, hb, he.symm⟩)
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
    obtain ⟨hwsha, hwhm⟩ := hrest w hw
    have hne : w.dst.1 ≠ n := by
      rcases hn with rfl | rfl
      · exact hwsha
      · exact hwhm
    obtain ⟨hm', hk', -⟩ := rangeOf_spec (C.one_range _ hd1)
    have hq := getD_mem_toList hd3 (d := default)
    have hle := (nets_ok P w.dst.1).1.2.2 _ hq
    have hfit := (L.fit w.dst.1).2
    intro he
    refine hne_at hd1 hne (s' := g * st.c.perVu.getD w.dst.1 0 + w.dst.2.1)
      (off' := ((portCols (st.c.nets.getD w.dst.1 default).2.inGroups).getD w.dst.2.2 default).1 + tt)
      ⟨by rw [C.count _ hm' _ hk']; exact Rows.mul_add_lt (List.mem_range.1 hg) hd2,
        by rw [L.slotLog _ hm' _ hk']; unfold Flat.wireWidth at htt; omega, ?_⟩
    rw [← he]
    unfold Flat.wireDst Circuit.slotCol
    rw [Nat.add_assoc]
  have hf' : (rowCopiesW st.c st.pin ++ wireCopies st.c).filter (fun e => e.1 = x) =
      (rowCopiesW st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter (fun e => e.1 = x) := by
    rw [List.filter_append, hwire, ← List.filter_append]
  simp only [flatPairsW, wirePairs_eq, List.count_append, hC, hleaf, hcut, Nat.zero_add, Nat.add_zero]
  rw [← List.count_append, ← List.flatMap_append, count_pair2, hf']

end Count

section Msg

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
include L F

/-- **`HmRow.delta` writes no row's message bit**, at a row of any widened shape: a row bit of a whole row's compressions,
of a short row's `hm96` slot or of a segment's data blocks. -/
theorem flat_count_msgW {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hs : rowShapeW st.c.ports[p]! = true) (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8) (j : ℕ) :
    (flatPairsW st.c st.pin).count (Hidden.deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  have P : ParseFacts tags st.c := F.parse
  obtain ⟨k, tb, hk, htb, hd, hm⟩ := deltaMsgCol_at P (g := g) hp hs hb ht
  rw [hd, msgAt_pos R hp]
  refine inPos_cases R (off := 512 + tb) hg hp hk (by omega) fun {_ _ _} hn hat {_ _} hl hoff _ => ?_
  rw [flatW_count L F hn hat hl hoff j, ← msgAt_pos R hp, filter_msgW R hg hp hk htb, hm]
  rfl

/-- **`Hidden.outPairs` writes no row `HmRow.delta` writes.** -/
theorem outPairs_freshW (hrows : OutRowsS st.c) :
    ∀ e ∈ Hidden.outPairs st.c, ∀ j, (HmRow.delta st.c st.pin).1.toList.count (e.1, j) = 0 := by
  have P : ParseFacts tags st.c := F.parse
  intro e he j
  rw [delta_flat_eqW _ _ P.typed]
  simp only [Hidden.outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at he
  obtain ⟨g, hg, jj, hjj, k, hk, t, ht, he⟩ := he
  have hp := hrows.2 jj hjj
  have hw : (st.c.ports.getD (HmOut.nIn st.c + jj) default).words = st.c.ports[HmOut.nIn st.c + jj]!.words := by
    rw [Hidden.getElem!_eq_getD']
  have h1 : e.1 = Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + jj) (2 * k + t / 8) (t % 8) := by
    rcases he with rfl | rfl <;> rfl
  rw [h1]
  exact flat_count_msgW L F hg hp (hrows.1 _ (getElem_mem_toList' hp)) (by rw [← hw]; omega)
    (Nat.mod_lt _ (by omega)) j

/-- `Stmt.setupHidden`'s Δ (`hidden_delta'`). -/
theorem rowDelta_hidden (hrows : OutRowsS st.c)
    (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray) :
    RowDelta st (Hidden.outPairs st.c) (fun _ => true) :=
  ⟨by rw [hda, Array.filter_eq_self.2 fun _ _ => rfl], outPairs_freshW L F hrows, fun _ _ => rfl⟩

/-- **A statement with the verifier's public inputs**: `HmIn.delta`'s Δ, given that no row net's slot's bit is a public
column (ZkPubIn's `publicCol_rowNet`). -/
theorem rowDelta_hmIn (hrows : OutRowsS st.c) {pi : HmIn.Ports} (hda : st.da = (HmIn.delta st.c pi st.pin).1)
    (hpub : ∀ {n s off x : ℕ}, (n = st.c.shaNet ∨ n = st.c.hmNet) → AtSlot st.c n s off x →
      HmIn.publicCol st.c pi x = false) :
    RowDelta st (Hidden.outPairs st.c) (fun x => pi.ports.isEmpty || !HmIn.publicCol st.c pi x) := by
  refine ⟨?_, outPairs_freshW L F hrows, fun hn hat => by simp [hpub hn hat]⟩
  rw [hda, HmIn.delta, Hidden.hidden_delta']
  by_cases h : pi.ports.isEmpty
  · simp only [h, ↓reduceIte, Bool.true_or]
    exact (Array.filter_eq_self.2 fun _ _ => rfl).symm
  · simp [h]

end Msg

/-! ## The rows' copies -/

section Rows

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {extra : List (ℕ × ℕ)} {keep : ℕ → Bool} (hΔ : RowDelta st extra keep)
include L F hΔ

/-- **An input bit of a row net's slot holds its one copy**, for rows of any shape. At a self row `x` of a `sha512x3` or
`hm96` slot below the net's input rows, where the rows' copies and wires write exactly one `(x, src)`, every block of a
satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copyLW {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopiesW st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src := by
  sorry

variable (Lf : Layout (Hidden.flatTwin st))
include Lf

/-- **VU `g`'s row `p`'s copies, for a row of any shape**, in every block of a satisfying witness: a chain start's `cv`
its midstate, every other compression's `cv` (the `hm96` slot's included) the previous one's output, a constant message
bit its constant, a digest bit the segment's last output's. -/
theorem row_copies_flatW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ j, HmRow.chainStart st.c.ports[p]! k = some j → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = wordsBit (startMid st.c.ports[p]! j) t) ∧
    (∀ k, 0 < k → k ≤ st.c.ports[p]!.chunks → HmRow.chainStart st.c.ports[p]! k = none → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p (k - 1) + t)) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit st.c.ports[p]! k t = .const v →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = v) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit st.c.ports[p]! k t = .digest kk col →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p kk + col)) := by
  sorry

/-- **The four copies into a row of whole blocks** (`RowWhole`), as `Hidden.row_copies_tgt`'s for any rows: the first
compression's `cv` the row prefix's midstate, each later one's the previous output, the `hm96` slot's `cv` the last
output, and its `pad` the padding block. -/
theorem row_copies_wholeW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowWhole st.c.ports[p]!) :
    (∀ t < 512, blockBit (L.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (L.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t)) ∧
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (L.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t)) ∧
    (∀ t < 1024, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) := by
  sorry

/-- **The copies into a short row** (`RowV2Short`, no compression): its `hm96` slot's `cv` the row prefix's midstate, and
its `pad` past the row's `2 · words` bytes the row's padding. -/
theorem row_copies_shortW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowV2Short st.c.ports[p]!) :
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ t < 1024, 2 * st.c.ports[p]!.words ≤ (HmRow.blockBit t).1 →
      blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
        tailBit (HmRow.tailOf st.c.ports[p]!) (2 * st.c.ports[p]!.words) t) := by
  sorry

/-- **The copies into a whole-block segmented row** (`RowSegWhole`): each segment's chain from its midstate through its
data blocks, its last block the padding block; the outer chain from the row prefix's midstate into the `hm96` slot, its
message the segments' digests, each byte copied from its segment's last output, then the padding. -/
theorem row_copies_segW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowSegWhole st.c.ports[p]!) :
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! j) + t) =
        wordsBit (HmRow.segMidstate st.c.ports[p]! j) t) ∧
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ i < ((segsOf st.c.ports[p]!)[j]!).1 / 1024, ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! j + i + 1) + t) =
        blockBit (L.model regs) z o (outAt st.c g p (segFirst st.c.ports[p]! j + i) + t)) ∧
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ t < 1024,
      blockBit (L.model regs) z o
          (msgAt st.c g p (segFirst st.c.ports[p]! j + ((segsOf st.c.ports[p]!)[j]!).1 / 1024) + t) =
        wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf st.c.ports[p]!)[j]!).1 / 8)) 16) t) ∧
    (∀ t < 512, blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ k, segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size ≤ k → k < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (k + 1) + t) = blockBit (L.model regs) z o (outAt st.c g p k + t)) ∧
    (∀ e < 64 * (segsOf st.c.ports[p]!).size, ∀ b < 8,
      blockBit (L.model regs) z o (msgAt st.c g p (segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size + e / 128) +
          (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b)) =
        blockBit (L.model regs) z o (outAt st.c g p (segFirst st.c.ports[p]! (e / 64 + 1) - 1) +
          (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b))) ∧
    (∀ t < 1024, 64 * (segsOf st.c.ports[p]!).size % 128 ≤ (HmRow.blockBit t).1 →
      blockBit (L.model regs) z o (msgAt st.c g p st.c.ports[p]!.chunks + t) =
        tailBit (HmRow.tailOf st.c.ports[p]!) (64 * (segsOf st.c.ports[p]!).size % 128) t) := by
  sorry

end Rows

section Out

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {keep : ℕ → Bool} (hΔ : RowDelta st (Hidden.outPairs st.c) keep) (hrows : OutRowsS st.c)
include L F hΔ hrows

/-- **An output row's message bit holds its copy**, for output rows of every widened shape and runs of output ports: at
VU `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copyW {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outRowPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (Hidden.outSrc st.c g j k t) := by
  sorry

end Out

/-! ## The plain forms -/

section Plain

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
  (hda : st.da = (HmRow.delta st.c st.pin).1)
include L F hda

/-- `inBit_copyLW` at a statement whose Δ is `HmRow.delta`'s (`Stmt.setupH`). -/
theorem inBit_copyLW_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopiesW st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src :=
  inBit_copyLW L (twin_of_flat L F hda).1 (rowDelta_flat hda) hz o hn hat hl hoff hself hf

/-- `row_copies_flatW` at a statement whose Δ is `HmRow.delta`'s (`Stmt.setupH`). -/
theorem row_copies_flatW_flat (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ j, HmRow.chainStart st.c.ports[p]! k = some j → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = wordsBit (startMid st.c.ports[p]! j) t) ∧
    (∀ k, 0 < k → k ≤ st.c.ports[p]!.chunks → HmRow.chainStart st.c.ports[p]! k = none → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p (k - 1) + t)) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit st.c.ports[p]! k t = .const v →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = v) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit st.c.ports[p]! k t = .digest kk col →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p kk + col)) :=
  row_copies_flatW L (twin_of_flat L F hda).1 (rowDelta_flat hda) (twin_of_flat L F hda).2 hupv hz o hg hp

end Plain

end FlockSoundness.Discharge.ScopeW
