import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Fit
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegRowBc
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegV2

/-!
# Unsegmented output rows, beside input rows of any kind

`Wide.Fit`'s output-row lemmas with only the output rows unsegmented (`Any.OutRowsS`): each output row a `PadRow`, its
padding perhaps starting in its last compression's block, and an input row may be a `sha512/row-seg/v1` row. The proofs
are `Wide.Fit`'s, with the per-row facts of `Wide.SegCopies` (`Any.msgData_filterAll`, `Any.pad_filterAll`) and
`Wide.SegDelta` (`delta_flat_any`) in place of the all-rows-unsegmented ones: an output row's data bit is a self input
bit of its compression's `sha512x3` slot or of its `hm96` slot's `pad` (`Any.msg_at`), where Δ writes no copy
(`Any.msg_filterF`: its constants there are the padding's, past the data bits), so `outPairs` copies into it once
(`Any.outPairs_countF`), and a satisfying witness of the hidden statement satisfies its flat twin (`Any.sat_flatF`) and
holds each output row's message bit's copy (`Any.msg_copyF`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

namespace Any

/-- **An unsegmented row the hidden-output statement's proof covers**: `16 · words` bits, its `2 · words` bytes and
SHA-512's 17 bytes of padding filling its `chunks` compressions' blocks and ending in its `hm96` slot's block. Unlike a
`FitRow`, the last compression's block may hold the bytes' end and the padding's start, the `hm96` block then only
padding (`2 · words mod 128 ≥ 112`). -/
def PadRow (p : Port) : Prop :=
  p.segs = none ∧ p.bits = 16 * p.words ∧ 128 * p.chunks ≤ 2 * p.words + 16 ∧ 2 * p.words + 17 ≤ 128 * (p.chunks + 1)

theorem padRow_of_fit {p : Port} (h : FitRow p) : PadRow p := ⟨h.1, h.2.1, by have := h.2.2.1; omega, h.2.2.2⟩

/-- **Output rows beside input rows of any kind**: one output row per output port, each after the input rows and
unsegmented (`PadRow`). -/
def OutRowsS (c : Circuit) : Prop :=
  c.outRowPorts = Array.replicate c.outPorts.size 1 ∧
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧ PadRow c.ports[HmOut.nIn c + j]!

namespace OutRowsS

variable {c : Circuit} (h : OutRowsS c)
include h

theorem size : c.outRowPorts.size = c.outPorts.size := by rw [h.1, Array.size_replicate]

theorem segs {j : ℕ} (hj : j < c.outPorts.size) : (c.ports.getD (HmOut.nIn c + j) default).segs = none := by
  rw [← getElem!_eq_getD']
  exact (h.2 j hj).2.1

theorem bound {j k t : ℕ} (hj : j < c.outPorts.size) (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words)
    (ht : t < 16) :
    HmOut.nIn c + j < c.ports.size ∧ 2 * k + t / 8 < 2 * c.ports[HmOut.nIn c + j]!.words ∧ t % 8 < 8 := by
  rw [getElem!_eq_getD']
  exact ⟨(h.2 j hj).1, by omega, by omega⟩

end OutRowsS

/-- **Output rows beside input rows of any kind, past them output ports without a row**: one output row per run of
`out_row_ports`, each run one output port, each row after the input rows and unsegmented (`PadRow`). The output ports
past the runs have no row: META `pinned_outputs`' (`HmRow.pinnedCount`). With every output port in a run it is
`OutRowsS` (`OutRowsS.toP`, `OutRowsP.toS`). -/
def OutRowsP (c : Circuit) : Prop :=
  c.outRowPorts = Array.replicate c.outRowPorts.size 1 ∧
    ∀ j < c.outRowPorts.size, HmOut.nIn c + j < c.ports.size ∧ PadRow c.ports[HmOut.nIn c + j]!

theorem OutRowsS.toP {c : Circuit} (h : OutRowsS c) : OutRowsP c :=
  ⟨by rw [h.size]; exact h.1, fun j hj => h.2 j (by rw [← h.size]; exact hj)⟩

/-- An `OutRowsS` proof where an `OutRowsP` one is asked for (`OutRowsS.toP`). -/
instance instCoeOutRowsSOutRowsP {c : Circuit} : Coe (OutRowsS c) (OutRowsP c) := ⟨OutRowsS.toP⟩

theorem OutRowsP.toS {c : Circuit} (h : OutRowsP c) (hs : c.outRowPorts.size = c.outPorts.size) : OutRowsS c :=
  ⟨by rw [← hs]; exact h.1, fun j hj => h.2 j (by rw [hs]; exact hj)⟩

namespace OutRowsP

variable {c : Circuit} (h : OutRowsP c)
include h

theorem segs {j : ℕ} (hj : j < c.outRowPorts.size) : (c.ports.getD (HmOut.nIn c + j) default).segs = none := by
  rw [← getElem!_eq_getD']
  exact (h.2 j hj).2.1

theorem bound {j k t : ℕ} (hj : j < c.outRowPorts.size) (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words)
    (ht : t < 16) :
    HmOut.nIn c + j < c.ports.size ∧ 2 * k + t / 8 < 2 * c.ports[HmOut.nIn c + j]!.words ∧ t % 8 < 8 := by
  rw [getElem!_eq_getD']
  exact ⟨(h.2 j hj).1, by omega, by omega⟩

end OutRowsP

/-- `outCopies` over the output rows (`out_row_ports`' runs) rather than the output ports. -/
def outCopiesR (c : Circuit) : List (ℕ × ℕ) :=
  (List.range c.g).flatMap fun g => (List.range c.outRowPorts.size).flatMap fun j =>
    (List.range (c.ports.getD (HmOut.nIn c + j) default).words).flatMap fun k =>
      (List.range 16).map fun t => (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)

theorem outPairs_eqR (c : Circuit) : outPairs c = (outCopiesR c).flatMap pair2 := by
  simp only [outPairs, outCopiesR, List.flatMap_assoc, List.flatMap_map, pair2]

/-- Rows that fit give `OutRowsS`. -/
theorem OutRowsF.outRowsS {c : Circuit} (h : OutRowsF c) : OutRowsS c :=
  ⟨h.2.1, fun j hj => ⟨h.2.2 j hj, padRow_of_fit (h.fit (h.2.2 j hj))⟩⟩

section Pos

variable {c : Circuit} (R : RowRanges c)
include R

omit R in
/-- **A data bit past an unsegmented row's compressions** is in its `hm96` slot's `pad`. -/
theorem deltaMsgCol_hm {g p b t : ℕ} (hp : p < c.ports.size) (hs : (c.ports.getD p default).segs = none)
    (hk : c.ports[p]!.chunks ≤ b / 128) : deltaMsgCol c g p b t = padCol c g p + msgOff b t := by
  have hk' : ¬ (HmRow.byteAt (c.ports.getD p default) b).1 < ((HmRow.rowSlots c.ports).2[p]!).size := by
    rw [Typed.byteAt_none hs, rowSlots_size hp]
    omega
  simp only [deltaMsgCol, hk', ↓reduceIte]
  simp only [Typed.byteAt_none hs, padCol, msgOff]
  omega

/-- **An unsegmented row's data bit is a self input bit of one of its slots**: of its compression's `sha512x3` slot,
past the `cv`, or of its `hm96` slot's `pad`. -/
theorem msg_at {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hs : (c.ports.getD p default).segs = none)
    (ht : t < 8) :
    ∃ n s off, (n = c.shaNet ∨ n = c.hmNet) ∧ AtSlot c n s off (deltaMsgCol c g p b t) ∧
      off < (c.nets.getD n default).2.inWords * Flock.WORD ∧
      ∃ (l : Flock.HmNets.Laid) (name : String),
        Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (c.nets.getD n default).2 ∧
          l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off] := by
  have hmo := msgOff_lt (b := b) ht
  by_cases hk : b / 128 < c.ports[p]!.chunks
  · obtain ⟨hat, hoff⟩ := Any.deltaMsgCol_at R hg hp hs (b := b) (t := t) (by omega) ht
    exact ⟨_, _, _, Or.inl rfl, hat, by rw [shaNet_inRows R.rowNets]; omega, _, _, shaNet_ofRows R.rowNets,
      shaLaid_self hoff⟩
  · have hx : deltaMsgCol c g p b t = c.slotCol c.hmNet g p + (512 + msgOff b t) := by
      rw [deltaMsgCol_hm hp hs (by omega), padCol_eq R.rowNets, Nat.add_assoc]
    rw [hx]
    exact ⟨_, _, _, Or.inr rfl, R.hmIn_at hg hp (by omega), by rw [hmNet_inRows R.rowNets]; omega, _, _,
      hmNet_ofRows R.rowNets, hmLaid_self (by omega)⟩

theorem deltaMsgCol_ltF {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hs : (c.ports.getD p default).segs = none) (ht : t < 8) : deltaMsgCol c g p b t < 2 ^ c.kLog := by
  obtain ⟨n, s, off, hn, hat, -⟩ := msg_at R (b := b) hg hp hs ht
  exact hat.lt R.bound (R.range (R.rowNet_lt hn)).1

/-- **No copy of `HmRow.delta`'s rows, of any kind, or of the rows' wires is at an unsegmented row's data bit.** -/
theorem msg_filterF {pin g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hfit : PadRow c.ports[p]!)
    (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) :
    (rowCopiesAny c pin ++ wireCopiesOf c (HmRow.rowWires c.ports c.shaNet c.hmNet).toList).filter
      (fun e => e.1 = deltaMsgCol c g p b t) = [] := by
  obtain ⟨hup, hbits, h1, h2⟩ := hfit
  have hs : (c.ports.getD p default).segs = none := by rw [← getElem!_eq_getD']; exact hup
  by_cases hk : b / 128 < c.ports[p]!.chunks
  · exact Any.msgData_filterAll R hg hp hup hbits hb hk ht
  · rw [deltaMsgCol_hm hp hs (by omega), Any.pad_filterAll R hg hp hup (msgOff_lt ht), rowSlots_size hp,
      show c.ports[p]!.chunks = b / 128 by omega, constBit_msg hbits hb ht]
    rfl

/-- An `hm96` slot's input bit is no unsegmented row's message bit. -/
theorem hm_ne_msg {g' p' off' g p b t : ℕ} (hg' : g' < c.g) (hp' : p' < c.ports.size) (ho' : off' < 4096)
    (hg : g < c.g) (hp : p < c.ports.size) (hs : (c.ports.getD p default).segs = none)
    (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) :
    c.slotCol c.hmNet g' p' + off' ≠ deltaMsgCol c g p b t := by
  intro h
  have hn := R.range R.shaNet
  have hm := R.range R.hmNet
  exact sha_ne_hm R.rowNets ((Any.deltaMsgCol_at R hg hp hs hb ht).1.inj R.disj hn.1 hn.2.1 hm.1 hm.2.1
    (by rw [← h]; exact R.hmIn_at hg' hp' ho')).1

/-- **Distinct data bits of unsegmented rows are distinct columns**: template instance, row, byte and bit. -/
theorem deltaMsgCol_injF {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hfit : PadRow c.ports[p]!)
    (hb : b < 2 * c.ports[p]!.words) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hfit' : PadRow c.ports[p']!) (hb' : b' < 2 * c.ports[p']!.words) (ht' : t' < 8)
    (h : deltaMsgCol c g p b t = deltaMsgCol c g' p' b' t') : g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  have hs : (c.ports.getD p default).segs = none := by rw [← getElem!_eq_getD']; exact hfit.1
  have hs' : (c.ports.getD p' default).segs = none := by rw [← getElem!_eq_getD']; exact hfit'.1
  have hmo := msgOff_lt (b := b) ht
  have hmo' := msgOff_lt (b := b') ht'
  have hx : ∀ {g p b t : ℕ}, p < c.ports.size → (c.ports.getD p default).segs = none → c.ports[p]!.chunks ≤ b / 128 →
      deltaMsgCol c g p b t = c.slotCol c.hmNet g p + (512 + msgOff b t) := fun hp hs hk => by
    rw [deltaMsgCol_hm hp hs hk, padCol_eq R.rowNets, Nat.add_assoc]
  by_cases hk : b / 128 < c.ports[p]!.chunks <;> by_cases hk' : b' / 128 < c.ports[p']!.chunks
  · exact Any.deltaMsgCol_inj R hg hp hs (by omega) ht hg' hp' hs' (by omega) ht' h
  · rw [hx hp' hs' (by omega)] at h
    exact absurd h.symm (hm_ne_msg R hg' hp' (by omega) hg hp hs (by omega) ht)
  · rw [hx hp hs (by omega)] at h
    exact absurd h (hm_ne_msg R hg hp (by omega) hg' hp' hs' (by omega) ht')
  · rw [hx hp hs (by omega), hx hp' hs' (by omega)] at h
    obtain ⟨rfl, rfl, ho⟩ := R.hm_dec hg hp (by omega) hg' hp' (by omega) h
    obtain ⟨-, -, h1, h2⟩ := hfit
    unfold msgOff at ho
    refine ⟨rfl, rfl, by omega, by omega⟩

end Pos

/-! ## `outPairs` at an output row's message bit -/

section OutP

variable {c : Circuit} (R : RowRanges c) (hrows : OutRowsP c)
include R hrows

/-- **`outPairs`' one copy at output row `j`'s message bit for word `k`, bit `t`**: from `outSrc`. -/
theorem outCopies_filterP {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) :
    (outCopiesR c).filter (fun e => e.1 = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8)) =
      [(deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), outSrc c g j k t)] := by
  have hdec : ∀ g' < c.g, ∀ j' < c.outRowPorts.size, ∀ k' < (c.ports.getD (HmOut.nIn c + j') default).words,
      ∀ t' < 16, deltaMsgCol c g' (HmOut.nIn c + j') (2 * k' + t' / 8) (t' % 8) =
        deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) → g' = g ∧ j' = j ∧ k' = k ∧ t' = t := by
    intro g' hg' j' hj' k' hk' t' ht' he
    obtain ⟨hp', hb', ht''⟩ := hrows.bound hj' hk' ht'
    obtain ⟨hp, hb, ht'''⟩ := hrows.bound hj hk ht
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_injF R hg' hp' (hrows.2 j' hj').2 hb' ht'' hg hp (hrows.2 j hj).2 hb
      ht''' he
    exact ⟨h1, by omega, by omega, by omega⟩
  unfold outCopiesR
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
theorem outPairs_countP {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outRowPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) (i : ℕ) :
    (outPairs c).count (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), i) =
      (if i = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
        (if i = outSrc c g j k t then 1 else 0) := by
  rw [outPairs_eqR, count_pair2, outCopies_filterP R hrows hg hj hk ht]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero]

end OutP

section Out

variable {c : Circuit} (R : RowRanges c) (hrows : OutRowsS c)
include R hrows

/-- `outPairs_countP` at one output row per output port. -/
theorem outPairs_countF {g j k t : ℕ} (hg : g < c.g) (hj : j < c.outPorts.size)
    (hk : k < (c.ports.getD (HmOut.nIn c + j) default).words) (ht : t < 16) (i : ℕ) :
    (outPairs c).count (deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8), i) =
      (if i = deltaMsgCol c g (HmOut.nIn c + j) (2 * k + t / 8) (t % 8) then 1 else 0) +
        (if i = outSrc c g j k t then 1 else 0) :=
  outPairs_countP R hrows.toP hg (by rw [hrows.size]; exact hj) hk ht i

end Out

/-! ## The flat Δ at an unsegmented row's data bit, beside rows of any kind -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`flatPairsAny` at an unsegmented row's data bit**: none. -/
theorem flat_count_msgF {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (hfit : PadRow st.c.ports[p]!)
    (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8) (j : ℕ) :
    (flatPairsAny st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  have hs : (st.c.ports.getD p default).segs = none := by rw [← getElem!_eq_getD']; exact hfit.1
  obtain ⟨n, s, off, hn, hat, hoff, l, name, hl, -⟩ := msg_at R (b := b) hg hp hs ht
  rw [Any.flatAll_count L F F.parse.rowWires hn hat hl hoff j, msg_filterF R hg hp hfit hb ht]
  rfl

/-- **The flat Δ's row at an unsegmented row's data bit is a self row**, in `A` and in `B`. -/
theorem flat_selfF (hda : st.da = (HmRow.delta st.c st.pin).1) {g p b t : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hfit : PadRow st.c.ports[p]!) (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8)
    (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog) (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  have hs : (st.c.ports.getD p default).segs = none := by rw [← getElem!_eq_getD']; exact hfit.1
  obtain ⟨n, s, off, hn, hat, -, l, name, hl, hself⟩ := msg_at R (b := b) hg hp hs ht
  have hfc := flat_count_msgF L F hg hp hfit hb ht
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
      rw [deltaEntry_eq_count, hda, delta_flat_any _ _ F.parse.typed, hfc]
      split_ifs <;> rfl
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact inBit_rows R hn hat hl hself hx hδ j

end Twin

/-! ## The hidden statement and its flat twin, with unsegmented output rows -/

section HiddenP

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsP st.c)
include L L0 F0 hda hrows

/-- **An output row's message bit holds its copy** in every block of a satisfying witness of the hidden statement, with
unsegmented output rows and input rows of any kind: at template instance `g`, output row `j`'s word `k`, bit `t`, what the
witness holds at `outSrc`. -/
theorem msg_copyP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outRowPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
  obtain ⟨n, s, off, hn, hat, -, l, name, hl, hself⟩ := msg_at R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hx := deltaMsgCol_ltF R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
  have hfc := flat_count_msgF L0 F0 hg hp (hrows.2 j hj).2 hb ht8
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
        delta_flat_any _ _ F0.parse.typed, hfc, Nat.zero_add, outPairs_countP R hrows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact block_of_rows hz o hx fun i => inBit_rows (st := st) R hn hat hl hself hx hδ i

/-- **A satisfying witness of the hidden statement satisfies its flat twin**, with unsegmented output rows and input rows of
any kind: the rows agree past `outPairs`' message bits (`deltaEntry_append_out`), and there the twin's row is a self row
(`flat_selfF`). -/
theorem sat_flatP {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i => ?_, hz.2.1, hz.2.2⟩
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    obtain ⟨hp, hb, ht8⟩ := hrows.bound hj hk ht
    have hx := deltaMsgCol_ltF R (b := 2 * k + t / 8) hg hp (hrows.segs hj) ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfF L0 F0 rfl hg hp (hrows.2 j hj).2 hb ht8 hx i'
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

end HiddenP

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsS st.c)
include L L0 F0 hda hrows

/-- `msg_copyP` at one output row per output port. -/
theorem msg_copyF {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) :=
  msg_copyP L L0 F0 hda hrows.toP hz o hg (by rw [hrows.size]; exact hj) hk ht

/-- `sat_flatP` at one output row per output port. -/
theorem sat_flatF {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z :=
  sat_flatP L L0 F0 hda hrows.toP hz

end Hidden

end Any

end FlockSoundness.Discharge.Hidden
