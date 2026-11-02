import Proofs.Flock.Soundness.Discharge.Hidden.Wide.V2

/-!
# Output rows whose last bytes share the `hm96` pad block

`Hidden.Flat`'s and `Wide.V2`'s output-row lemmas at rows that fit (`FitRow`): unsegmented, `16 · words` bits, the
row's `2 · words` bytes filling its compressions' blocks and ending, with SHA-512's padding, in its `hm96` slot's block.
A `sha512/row/v1` row is one, and so is a `sha512/row/v2` row whose bytes past its compressions are fewer than 112. A
data bit of such a row is an input bit of its compression's `sha512x3` slot or of its `hm96` slot's `pad`
(`msg_at`, `deltaMsgCol_hm`), where `HmRow.delta` writes no copy (`msg_filterF`: the `pad`'s constant bits are past the
row's bits, `constBit_msg`), so the flat twin's row there is a self row (`flat_selfF`), `outPairs` copies into it once
(`outPairs_countF`), and a satisfying witness of the hidden statement satisfies its flat twin (`sat_flatF`) and holds
each output row's message bit's copy (`msg_copyF`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- **A row the hidden-output statement's proof covers**: unsegmented, `16 · words` bits, its `2 · words` bytes filling
its `chunks` compressions' blocks and ending, with SHA-512's 17 bytes of padding, in its `hm96` slot's block. -/
def FitRow (p : Port) : Prop :=
  p.segs = none ∧ p.bits = 16 * p.words ∧ 128 * p.chunks ≤ 2 * p.words ∧ 2 * p.words + 17 ≤ 128 * (p.chunks + 1)

/-- Rows that fit, one output row per output port, each after the input rows. -/
def OutRowsF (c : Circuit) : Prop :=
  (∀ p ∈ c.ports.toList, FitRow p) ∧ c.outRowPorts = Array.replicate c.outPorts.size 1 ∧
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size

namespace OutRowsF

variable {c : Circuit} (h : OutRowsF c)
include h

theorem segs : ∀ p ∈ c.ports.toList, p.segs = none := fun p hp => (h.1 p hp).1

theorem size : c.outRowPorts.size = c.outPorts.size := by rw [h.2.1, Array.size_replicate]

theorem fit {p : ℕ} (hp : p < c.ports.size) : FitRow c.ports[p]! := by
  rw [getElem!_pos c.ports p hp]
  exact h.1 _ (Array.getElem_mem_toList hp)

theorem bound {j k t : ℕ} (hj : j < c.outPorts.size) (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words)
    (ht : t < 16) :
    HmOut.nIn c + j < c.ports.size ∧ 2 * k + t / 8 < 2 * c.ports[HmOut.nIn c + j]!.words ∧ t % 8 < 8 := by
  rw [getElem!_eq_getD']
  exact ⟨h.2.2 j hj, by omega, by omega⟩

end OutRowsF

/-- `OutRowsW`'s rows fit. -/
theorem OutRowsW.outRowsF {c : Circuit} (h : OutRowsW c) (hfit : ∀ p ∈ c.ports.toList, FitRow p) : OutRowsF c :=
  ⟨hfit, h.1.2.1, fun j hj => (h.1.2.2 j hj).1⟩

/-- **A row's constant bits past its bits**: at a data byte `b < 2 · words` of a row of `16 · words` bits, block
`b / 128`'s message column of byte `b`'s bit `t` holds no constant. -/
theorem constBit_msg {p : Port} {tl : ByteArray} {b t : ℕ} (hbits : p.bits = 16 * p.words) (hb : b < 2 * p.words)
    (ht : t < 8) : HmRow.constBit p tl (b / 128) (msgOff b t) = none := by
  unfold HmRow.constBit HmRow.blockBit msgOff
  exact ite_eq_left_iff.2 fun h => absurd (by omega) h

section Pos

variable {c : Circuit} (R : RowRanges c) (hu : ∀ p ∈ c.ports.toList, p.segs = none)
include R hu

omit R in
/-- **A data bit past a row's compressions** is in its `hm96` slot's `pad`. -/
theorem deltaMsgCol_hm {g p b t : ℕ} (hp : p < c.ports.size) (hk : c.ports[p]!.chunks ≤ b / 128) :
    deltaMsgCol c g p b t = padCol c g p + msgOff b t := by
  have hk' : ¬ (HmRow.byteAt (c.ports.getD p default) b).1 < ((HmRow.rowSlots c.ports).2[p]!).size := by
    rw [Typed.byteAt_none (segs_getD hu hp), rowSlots_size hp]
    omega
  simp only [deltaMsgCol, hk', ↓reduceIte]
  simp only [Typed.byteAt_none (segs_getD hu hp), padCol, msgOff]
  omega

/-- **A fit row's data bit is a self input bit of one of its slots**: of its compression's `sha512x3` slot, past the
`cv`, or of its `hm96` slot's `pad`. -/
theorem msg_at {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht : t < 8) :
    ∃ n s off, (n = c.shaNet ∨ n = c.hmNet) ∧ AtSlot c n s off (deltaMsgCol c g p b t) ∧
      off < (c.nets.getD n default).2.inWords * Flock.WORD ∧
      ∃ (l : Flock.HmNets.Laid) (name : String),
        Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (c.nets.getD n default).2 ∧
          l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off] := by
  have hmo := msgOff_lt (b := b) ht
  by_cases hk : b / 128 < c.ports[p]!.chunks
  · obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp (b := b) (t := t) (by omega) ht
    exact ⟨_, _, _, Or.inl rfl, hat, by rw [shaNet_inRows R.rowNets]; omega, _, _, shaNet_ofRows R.rowNets,
      shaLaid_self hoff⟩
  · have hx : deltaMsgCol c g p b t = c.slotCol c.hmNet g p + (512 + msgOff b t) := by
      rw [deltaMsgCol_hm hu hp (by omega), padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx]
    exact ⟨_, _, _, Or.inr rfl, R.hmIn_at hg hp (by omega), by rw [hmNet_inRows R.rowNets]; omega, _, _,
      hmNet_ofRows R.rowNets, hmLaid_self (by omega)⟩

theorem deltaMsgCol_ltF {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (ht : t < 8) :
    deltaMsgCol c g p b t < 2 ^ c.kLog := by
  obtain ⟨n, s, off, hn, hat, -⟩ := msg_at R hu (b := b) hg hp ht
  exact hat.lt R.bound (R.range (R.rowNet_lt hn)).1

/-- **No copy of `HmRow.delta`'s rows or of the rows' wires is at a fit row's data bit.** -/
theorem msg_filterF {pin g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hfit : FitRow c.ports[p]!)
    (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) :
    (rowCopiesAll c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  obtain ⟨-, hbits, h1, h2⟩ := hfit
  by_cases hk : b / 128 < c.ports[p]!.chunks
  · exact msg_filterAll R hu hg hp (by omega) (by omega) ht
  · rw [deltaMsgCol_hm hu hp (by omega), pad_filterAll R hu hg hp (msgOff_lt ht), rowSlots_size hp,
      show c.ports[p]!.chunks = b / 128 by omega, constBit_msg hbits hb ht]
    rfl

/-- **Distinct data bits of fit rows are distinct columns**: template instance, row, byte and bit. -/
theorem deltaMsgCol_injF {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hfit : FitRow c.ports[p]!)
    (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hfit' : FitRow c.ports[p']!) (hb' : b' < 2 * c.ports[p']!.words) (ht' : t' < 8)
    (h : deltaMsgCol c g p b t = deltaMsgCol c g' p' b' t') : g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  have hmo := msgOff_lt (b := b) ht
  have hmo' := msgOff_lt (b := b') ht'
  have hx : ∀ {g p b t : ℕ}, p < c.ports.size → c.ports[p]!.chunks ≤ b / 128 →
      deltaMsgCol c g p b t = c.slotCol c.hmNet g p + (512 + msgOff b t) := fun hp hk => by
    rw [deltaMsgCol_hm hu hp hk, padCol_eq R.rowNets, Nat.add_assoc]
  by_cases hk : b / 128 < c.ports[p]!.chunks <;> by_cases hk' : b' / 128 < c.ports[p']!.chunks
  · exact deltaMsgCol_inj R hu hg hp (by omega) ht hg' hp' (by omega) ht' h
  · rw [hx hp' (by omega)] at h
    exact absurd h.symm (hm_ne_msg R hu hg' hp' (by omega) hg hp (by omega) ht)
  · rw [hx hp (by omega)] at h
    exact absurd h (hm_ne_msg R hu hg hp (by omega) hg' hp' (by omega) ht')
  · rw [hx hp (by omega), hx hp' (by omega)] at h
    obtain ⟨rfl, rfl, ho⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) h
    obtain ⟨-, -, h1, h2⟩ := hfit
    unfold msgOff at ho
    refine ⟨rfl, rfl, by omega, by omega⟩

end Pos

/-! ## `outPairs` at a fit output row's message bit -/

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRowsF c)
include R hrows

/-- **`outPairs`' one copy at output row `j`'s message bit for word `k`, bit `t`**, at fit rows: from `outSrc`. -/
theorem outCopies_filterF {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    (outCopies c).filter (fun e => e.1 = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
      [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)] := by
  have hdec : ∀ g' < c.g, ∀ j' < c.outPorts.size, ∀ k' < (c.ports.getD (HmOut.nIn c + j') default).words,
      ∀ t' < 16, deltaMsgCol c g' (HmOut.nIn c + j') (2 * k' + t' / 8) (t' % 8) =
        deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) → g' = g ∧ j' = j ∧ k' = k ∧ t' = t := by
    intro g' hg' j' hj' k' hk' t' ht' he
    obtain ⟨hp', hb', ht''⟩ := hrows.bound hj' hk' ht'
    obtain ⟨hp, hb, ht'''⟩ := hrows.bound hj hk ht
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_injF R hrows.segs hg' hp' (hrows.fit hp') hb' ht'' hg hp (hrows.fit hp) hb
      ht''' he
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

/-- **`outPairs`' pairs at output row `j`'s message bit for word `k`, bit `t`**, at fit rows: one copy from `outSrc`. -/
theorem outPairs_countF {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) (i : ℕ) :
    (outPairs c).count (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), i) =
      (if i = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
        (if i = outSrc c g j k t then 1 else 0) := by
  rw [outPairs_eq hrows.size, count_pair2, outCopies_filterF R hrows hg hj hk ht]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

end Out

/-! ## The flat Δ at a fit row's data bit -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`flatPairsAll` at a fit row's data bit**: none. -/
theorem flat_count_msgF (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) {g p b t : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hfit : FitRow st.c.ports[p]!) (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8)
    (j : ℕ) : (flatPairsAll st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨n, s, off, hn, hat, hoff, l, name, hl, -⟩ := msg_at R hu (b := b) hg hp ht
  rw [flatAll_count L F F.parse.rowWires hn hat hl hoff j, msg_filterF R hu hg hp hfit hb ht]
  rfl

/-- **The flat Δ's row at a fit row's data bit is a self row**, in `A` and in `B`. -/
theorem flat_selfF (hda : st.da = (HmRow.delta st.c st.pin).1) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none)
    {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hfit : FitRow st.c.ports[p]!)
    (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨n, s, off, hn, hat, -, l, name, hl, hself⟩ := msg_at R hu (b := b) hg hp ht
  have hfc := flat_count_msgF L F hu hg hp hfit hb ht
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
      rw [deltaEntry_eq_count, hda, delta_flat_all _ _ F.parse.typed hu, hfc]
      split_ifs <;> rfl
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact inBit_rows R hn hat hl hself hx hδ j

end Twin

/-! ## The hidden statement and its flat twin, at fit rows -/

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsF st.c)
include L L0 F0 hda hrows

/-- **An output row's message bit holds its copy** in every block of a satisfying witness of the hidden statement, at
fit rows: at template instance `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copyF {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
  obtain ⟨n, s, off, hn, hat, -, l, name, hl, hself⟩ := msg_at R hrows.segs (b := 2 * k + t / 8) hg hp ht8
  have hx := deltaMsgCol_ltF R hrows.segs (b := 2 * k + t / 8) hg hp ht8
  have hfc := flat_count_msgF L0 F0 hrows.segs hg hp (hrows.fit hp) hb ht8
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
        delta_flat_all _ _ F0.parse.typed hrows.segs, hfc, Nat.zero_add, outPairs_countF R hrows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact block_of_rows hz o hx fun i => inBit_rows (st := st) R hn hat hl hself hx hδ i

/-- **A satisfying witness of the hidden statement satisfies its flat twin**, at fit rows: the rows agree past
`outPairs`' message bits (`deltaEntry_append_out`), and there the twin's row is a self row (`flat_selfF`). -/
theorem sat_flatF {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
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
    obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
    have hx := deltaMsgCol_ltF R hrows.segs (b := 2 * k + t / 8) hg hp ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfF L0 F0 rfl hrows.segs hg hp (hrows.fit hp) hb ht8 hx i'
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

/-! ## Rows of one kind -/

/-- Whether a circuit's rows are `sha512/row/v2` rows: some row is. -/
def rowV2 (c : Circuit) : Bool := c.ports.any (·.v2)

/-- **The rows the hidden-output headlines cover**: rows that fit, one output row per output port after the input rows
(`OutRowsF`), all `sha512/row/v2` rows or none. -/
def RowsFit (c : Circuit) : Prop := OutRowsF c ∧ ∀ p ∈ c.ports.toList, p.v2 = rowV2 c

/-- **The first block of a row of `w` words**: `sha512/row/v2`'s for `16 w` bits, or `sha512/row/v1`'s. -/
def rowPre (v2 : Bool) (w : ℕ) : ByteArray := if v2 then HmRow.rowPrefixV2 (16 * w) else HmRow.rowPrefix w

theorem size_rowPre (v2 : Bool) (w : ℕ) : (rowPre v2 w).size = 128 := by
  unfold rowPre
  split
  · exact size_rowPrefixV2 _
  · exact size_rowPrefix _

/-- **A row's own prefix is `rowPre`** at rows that fit, of one kind. -/
theorem prefixOf_fit {c : Circuit} (h : RowsFit c) {p : ℕ} (hp : p < c.ports.size) :
    HmRow.prefixOf c.ports[p]! = rowPre (rowV2 c) c.ports[p]!.words := by
  obtain ⟨hs, hb, -, -⟩ := h.1.fit hp
  have hv : c.ports[p]!.v2 = rowV2 c := h.2 _ (by rw [getElem!_pos c.ports p hp]; exact Array.getElem_mem_toList hp)
  unfold HmRow.prefixOf rowPre
  rw [hs, hv, hb]
  rfl

end FlockSoundness.Discharge.Hidden
