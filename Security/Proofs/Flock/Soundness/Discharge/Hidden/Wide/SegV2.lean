import Proofs.Flock.Soundness.Discharge.Hidden.Wide.SegRowBc
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.V2

/-!
# Output rows unsegmented, input rows of any kind (`OutputsCommittedSeg`, `OutputsBindingSeg`)

`Wide.V2` with only the output rows unsegmented, so an input row may be a `sha512/row-seg/v1` row (the PoUW tile's
`x`): the flat twin's Δ is `flatPairsAny` (`delta_flat_any`), whose copies at an output row's message bit are none
(`Any.msg_filterAll`), so a satisfying witness of the hidden statement satisfies its flat twin (`Any.sat_flatAll`) and
holds each output row's message bit's copy (`Any.msg_copyAll`); the row's `b ‖ c` is under its own prefix
(`Any.row_bc_tgt`). An output row's columns and `outPairs`' copies need only the output rows unsegmented
(`Any.OutRows`, `Any.deltaMsgCol_inj`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- **`OutputsCommittedSeg`**: `OutputsCommittedV2` with only the output rows unsegmented (an input row may be a
`sha512/row-seg/v1` row). -/
def OutputsCommittedSeg : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    0 < st.c.upv →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g j : ℕ}, g < st.c.g → g < 2 ^ st.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    (pop.digest D.proved[o.val * st.c.g + g] (HmOut.nIn st.c + j) st.c.ports.size).data.toList =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
        (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))

/-- **`OutputsBindingSeg`** (under `cr/sha-512`): `OutputsBindingV2` under `OutputsCommittedSeg`'s conditions. -/
def OutputsBindingSeg : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins coins' : Coins} {tables : List (String × ByteArray)} {d d' : Lean.Json}
    {partition partition' : Option Lean.Json} {program program' : Option ByteArray} {st st' : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    Stmt.setupHidden tags cf pf coins' tables (some d') partition' program' = .ok st' →
    ∀ {pop : Public}, HmOut.loadPublic tags st.c pf = .ok pop → pop.tables = none →
    ∀ {D D' : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    Draw.UnitDraw.ofJson d' = .ok D' → (∀ u ∈ D'.proved, u < D'.population) →
    0 < st.c.upv →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).segs = none) →
    st.c.outRowPorts = Array.replicate st.c.outPorts.size 1 →
    (∀ j < st.c.outPorts.size, (st.c.ports.getD (HmOut.nIn st.c + j) default).words % 64 = 0) →
    ∀ (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ (L' : Layout st') (regs' : List (Model.Region st'.c.kLog (st'.m - st'.c.kLog))),
    (∀ r ∈ st'.regions.toList, Refine.regionOf st'.c.kLog (st'.m - st'.c.kLog) r ∈ regs') →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ {z' : Model.Witness (L'.model regs').m}, (L'.model regs').Satisfies z' →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) (o' : Fin (2 ^ (L'.model regs').nbl)) {g g' j : ℕ},
    g < st.c.g → g < 2 ^ st.c.g.log2 → g' < st'.c.g → g' < 2 ^ st'.c.g.log2 → j < st.c.outPorts.size →
    ∀ (hr : o.val * st.c.g + g < D.proved.size) (hr' : o'.val * st'.c.g + g' < D'.proved.size),
    D.proved[o.val * st.c.g + g] = D'.proved[o'.val * st'.c.g + g'] →
    outBytes (Hm.blockBit (L.model regs) z o) st.c g j ≠ outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j →
    Binding.Collides (commitPair
      ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L.model regs) z o) st.c g j)
      (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g (HmOut.nIn st.c + j)))
      ((HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default)).data.toList ++
        outBytes (Hm.blockBit (L'.model regs') z' o') st'.c g' j)
      (Hm.bytesOf (Hm.blockBit (L'.model regs') z' o') 192 (Hm.saltCol st'.c g' (HmOut.nIn st'.c + j))))


namespace Any

section Cols

variable {c : Circuit} (R : RowRanges c)
include R

/-- **Row `p`'s message bit `t` of byte `b`** is at its compression `b / 128`'s `sha512x3` slot, past the `cv`, below
the net's self rows. -/
theorem deltaMsgCol_at {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hsg : (c.ports.getD p default).segs = none)
    (hb : b < 128 * c.ports[p]!.chunks)
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
  rw [deltaMsgCol_eq _ _ _ _ _ hsg hp hk, (shaIn_eq R.rowNets _ hsub).2, shaBase, Circuit.slotCol]
  omega

theorem deltaMsgCol_lt {g p b t : ℕ} (hg : g < c.g) (hp : p < c.ports.size) (hsg : (c.ports.getD p default).segs = none)
    (hb : b < 128 * c.ports[p]!.chunks)
    (ht : t < 8) : deltaMsgCol c g p b t < 2 ^ c.kLog :=
  (deltaMsgCol_at R hg hp hsg hb ht).1.lt R.bound (R.range R.shaNet).1

/-- **Distinct message bits are distinct columns**: template instance, row, byte and bit. -/
theorem deltaMsgCol_inj {g p b t g' p' b' t' : ℕ} (hg : g < c.g) (hp : p < c.ports.size)
    (hsg : (c.ports.getD p default).segs = none) (hb : b < 128 * c.ports[p]!.chunks) (ht : t < 8) (hg' : g' < c.g) (hp' : p' < c.ports.size)
    (hsg' : (c.ports.getD p' default).segs = none) (hb' : b' < 128 * c.ports[p']!.chunks) (ht' : t' < 8) (h : deltaMsgCol c g p b t = deltaMsgCol c g' p' b' t') :
    g = g' ∧ p = p' ∧ b = b' ∧ t = t' := by
  have hn := R.range R.shaNet
  have hk : b / 128 < c.ports[p]!.chunks := by omega
  have hk' : b' / 128 < c.ports[p']!.chunks := by omega
  obtain ⟨-, hs, ho⟩ := (deltaMsgCol_at R hg hp hsg hb ht).1.inj R.disj hn.1 hn.2.1 hn.1 hn.2.1
    (by rw [h]; exact (deltaMsgCol_at R hg' hp' hsg' hb' ht').1)
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

end Cols

/-- `OutRows` with only the output rows unsegmented: one output row per output port, each after the input rows and of
`2 · words = 128 · chunks` bytes. -/
def OutRows (c : Circuit) : Prop :=
  (∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).segs = none) ∧ c.outRowPorts = Array.replicate c.outPorts.size 1 ∧
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
    obtain ⟨h1, h2, h3, h4⟩ := deltaMsgCol_inj R hg' hp' (hrows.1 j' hj') hb' ht'' hg hp (hrows.1 j hj) hb ht''' he
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

theorem out_rows_whole {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hu : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).segs = none) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (h64 : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0) :
    ∀ j < c.outPorts.size, HmOut.nIn c + j < c.ports.size ∧
      2 * (c.ports.getD (HmOut.nIn c + j) default).words = 128 * (c.ports.getD (HmOut.nIn c + j) default).chunks ∧
      (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words := by
  have P := parse_facts hp
  obtain ⟨⟨mj, typed, hrp⟩, -⟩ := parse_rowPorts hp
  intro j hj
  have hj' : j < c.outRowPorts.size := by rw [hr, Array.size_replicate]; exact hj
  have h6 := h64 j hj
  obtain ⟨hlt, hrow⟩ := checkRows_out hc hj'
  have hmem := getD_mem_toList (d := default) hlt
  have hsj := hu j hj
  generalize c.ports.getD (HmOut.nIn c + j) default = p at hmem hrow h6 hsj ⊢
  have hb : p.bits = 16 * p.words := by
    rcases hrow with ⟨-, -, hb, -⟩ | ⟨hs, hb⟩
    · exact hb
    · have hv : p.v2 = true := by
        rcases hs with hs | hs
        · rw [hsj] at hs; cases hs
        · exact hs
      rw [(rowPorts_v2 hrp p hmem hv).1, hb]
      omega
  refine ⟨hlt, ?_, hb⟩
  cases hv : p.v2
  · exact P.portWords p hmem hv
  · rw [(rowPorts_v2 hrp p hmem hv).2.1 hsj, hb]
    omega


/-- `OutRowsW` with only the output rows unsegmented (`out_rows_whole`). -/
def OutRowsW (c : Circuit) : Prop :=
  OutRows c ∧ ∀ j < c.outPorts.size,
    (c.ports.getD (HmOut.nIn c + j) default).bits = 16 * (c.ports.getD (HmOut.nIn c + j) default).words

theorem OutRowsW.rows {c : Circuit} (h : OutRowsW c) : OutRows c := h.1

theorem OutRowsW.kb {c : Circuit} (h : OutRowsW c) {j : ℕ} (hj : j < c.outPorts.size) :
    c.ports[HmOut.nIn c + j]!.chunks ≤ c.ports[HmOut.nIn c + j]!.bits / 1024 := by
  obtain ⟨-, hw⟩ := h.1.2.2 j hj
  have hb := h.2 j hj
  rw [getElem!_eq_getD']
  omega

theorem outRowsW_of {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Circuit}
    (hp : HmRow.parse tags cf tables = .ok c) (hc : HmOut.checkRows c = .ok ())
    (hu : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).segs = none) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (h64 : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0) : OutRowsW c :=
  have h := out_rows_whole hp hc hu hr h64
  ⟨⟨hu, hr, fun j hj => ⟨(h j hj).1, (h j hj).2.1⟩⟩, fun j hj => (h j hj).2.2⟩

/-! ## The flat Δ at an output row's message bit -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`flatPairsAll` at a message bit of a row with no constant message bit before its last block**: none. -/
theorem flat_count_msgAll (hrw : RowWires st.c) {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hs : (st.c.ports.getD p default).segs = none)
    (hkb : st.c.ports[p]!.chunks ≤ st.c.ports[p]!.bits / 1024) (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8)
    (j : ℕ) : (flatPairsAny st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hg hp hs hb ht
  rw [flatAll_count L F hrw (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; omega) j,
    msg_filterAll R (pin := st.pin) hg hp (by rw [getElem!_eq_getD']; exact hs) hkb hb ht]
  rfl

/-- **The flat Δ's row at such a message bit is a self row**, in `A` and in `B`, for any rows. -/
theorem flat_selfAll (hda : st.da = (HmRow.delta st.c st.pin).1) {g p b t : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hs : (st.c.ports.getD p default).segs = none) (hkb : st.c.ports[p]!.chunks ≤ st.c.ports[p]!.bits / 1024)
    (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hg hp hs hb ht
  have hfc := flat_count_msgAll L F F.parse.rowWires hg hp hs hkb hb ht
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
  exact inBit_rows R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ j

end Twin

/-! ## The hidden statement and its flat twin, for any rows -/

section Hidden

variable {tags : Tags} {st : Stmt} (L : Layout st) (L0 : Layout (flatTwin st)) (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRowsW st.c)
include L L0 F0 hda hrows

/-- **An output row's message bit holds its copy** in every block of a satisfying witness of the hidden statement, for
any rows: at template instance `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copyAll {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (outSrc st.c g j k t) := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  obtain ⟨hp, hb, ht8⟩ := outRow_bound hrows.rows hj hk ht
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hg hp (hrows.1.1 j hj) hb ht8
  have hx := deltaMsgCol_lt R hg hp (hrows.1.1 j hj) hb ht8
  have hfc := flat_count_msgAll L0 F0 F0.parse.rowWires hg hp (hrows.1.1 j hj) (hrows.kb hj) hb ht8
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
        delta_flat_any _ _ F0.parse.typed, hfc, Nat.zero_add, outPairs_count R hrows.rows hg hj hk ht]
    exact ⟨e, by rw [L.delta_b]; exact e⟩
  exact block_of_rows hz o hx fun i =>
    inBit_rows (st := st) R (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (shaLaid_self hoff) hx hδ i

/-- **A satisfying witness of the hidden statement satisfies its flat twin**, for any rows: the rows agree past
`outPairs`' message bits (`deltaEntry_append_out`), and there the twin's row is a self row (`flat_selfAll`). -/
theorem sat_flatAll {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) : (L0.model regs).Satisfies z := by
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  refine ⟨fun o i => ?_, hz.2.1, hz.2.2⟩
  by_cases hi : ∃ q ∈ outPairs st.c, q.1 = i.val
  · obtain ⟨q, hq, hqi⟩ := hi
    simp only [outPairs, List.mem_flatMap, List.mem_range, List.mem_cons, List.not_mem_nil, or_false] at hq
    obtain ⟨g, hg, j, hj, k, hk, t, ht, hq⟩ := hq
    have hq1 : q.1 = deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8) := by
      rcases hq with rfl | rfl <;> rfl
    rw [hrows.1.size] at hj
    obtain ⟨hp, hb, ht8⟩ := outRow_bound hrows.rows hj hk ht
    have hx := deltaMsgCol_lt R hg hp (hrows.1.1 j hj) hb ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfAll L0 F0 rfl hg hp (hrows.1.1 j hj) (hrows.kb hj) hb ht8 hx i'
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

/-- **An output row's message bytes are the output words its template instance computes** (`outBytes`), for any rows, in
every block of a satisfying witness of the hidden statement whose forced-zero row carries 0. -/
theorem msg_bytesAll {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    (hzero : blockBit (L.model regs) z o (zeroCol st.c) = false) {g j : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) :
    bytesOf (blockBit (L.model regs) z o) (2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words)
        (Typed.msgCol st.c g (HmOut.nIn st.c + j)) =
      outBytes (blockBit (L.model regs) z o) st.c g j := by
  have hG := outGroup_single hrows.1.2.1 hj
  unfold bytesOf outBytes
  congr 1
  funext b
  have hb := b.isLt
  have hcopy : ∀ t : Fin 8,
      blockBit (L.model regs) z o (Typed.msgCol st.c g (HmOut.nIn st.c + j) b.val t.val) =
        blockBit (L.model regs) z o (outSrc st.c g j (b.val / 2) (8 * (b.val % 2) + t.val)) := fun t => by
    have ht := t.isLt
    have e1 : 2 * (b.val / 2) + (8 * (b.val % 2) + t.val) / 8 = b.val := by omega
    have e2 : (8 * (b.val % 2) + t.val) % 8 = t.val := by omega
    have := msg_copyAll L L0 F0 hda hrows hz o hg hj (k := b.val / 2) (t := 8 * (b.val % 2) + t.val) (by omega)
      (by omega)
    rw [e1, e2] at this
    rw [msgCol_delta]
    exact this
  by_cases hk : b.val / 2 < (st.c.outPorts.getD j default).2
  · simp only [hk, ↓reduceIte]
    congr 1
    funext t
    rw [hcopy t, outSrc, hG]
    simp only [hk, ↓reduceIte, Nat.add_assoc]
  · simp only [hk, ↓reduceIte]
    have h0 : (fun t : Fin 8 => blockBit (L.model regs) z o (Typed.msgCol st.c g (HmOut.nIn st.c + j) b.val t.val)) =
        fun _ => false := by
      funext t
      rw [hcopy t, outSrc, hG]
      simp only [hk, ↓reduceIte, hzero]
    rw [h0]
    simp [Binding.byteOf]

end Hidden

theorem outputs_committed_seg' : OutputsCommittedSeg := by
  intro tags cf pf coins tables d partition program st ht h pop hpop hpt D hD hlt hupv hu hor h64 L regs hregs z hz o g
    j hg hg2 hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := parse_untyped ht s.parse
  have P := parse_facts hp
  have L0 : FlockSoundness.Layout (flatTwin st) := layout_of (st := flatTwin st) P s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (flatTwin st) := unitFacts_of_parse (st := flatTwin st) hp rfl
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := by rw [s.da, hidden_delta']
  have hrows : OutRowsW st.c := outRowsW_of hp hcr hu hor h64
  have hz0 := sat_flatAll L L0 F0 hda hrows hz
  have hzero : blockBit (L.model regs) z o (zeroCol st.c) = false :=
    zero_bit L0 F0 hupv (show 0 < st.c.g by omega) hz0 o
  obtain ⟨hpl, hw⟩ := hrows.1.2.2 j hj
  have hb := hrows.2 j hj
  have hbc : bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g (HmOut.nIn st.c + j)) =
      Binding.commitString
        ((HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default)).data.toList ++
          bytesOf (blockBit (L.model regs) z o) (2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words)
            (Typed.msgCol st.c g (HmOut.nIn st.c + j)))
        (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g (HmOut.nIn st.c + j))) := by
    have h2 : 2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words + 17 ≤
        128 * ((st.c.ports.getD (HmOut.nIn st.c + j) default).chunks + 1) := by omega
    have := row_bc_tgt L0 F0 rfl hupv hz0 o hg hpl (by rw [getElem!_eq_getD']; exact hb)
      (by rw [getElem!_eq_getD']; exact hw.ge) (by rw [getElem!_eq_getD']; exact h2)
      (by rw [getElem!_eq_getD']; exact hu j hj)
    rw [getElem!_eq_getD'] at this
    exact this
  rw [← msg_bytesAll L L0 F0 hda hrows hz o hzero hg hj, ← hbc, ← bytesAt_bc L R regs z o hg hpl]
  exact (bc_drawnH h hpop hpt hD hlt L regs hregs hz o hg hg2 hpl hr).symm

/-- **`OutputsBindingSeg` from `OutputsCommittedSeg`**. -/
theorem outputs_binding_seg_of (hC : OutputsCommittedSeg) : OutputsBindingSeg := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hs hs' pop hpop hpt D D' hD
    hDp hD' hDp' hupv hu hor h64 L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg2 hg' hg2' hj hr hr' hidx hne
  have hcc : st.c = st'.c := setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have hupv' : 0 < st'.c.upv := by rw [← hcc]; exact hupv
  have hj' : j < st'.c.outPorts.size := by rw [← hcc]; exact hj
  have hu' : ∀ j < st'.c.outPorts.size, (st'.c.ports.getD (HmOut.nIn st'.c + j) default).segs = none := by
    rw [← hcc]; exact hu
  have hor' : st'.c.outRowPorts = Array.replicate st'.c.outPorts.size 1 := by rw [← hcc]; exact hor
  have h64' : ∀ j < st'.c.outPorts.size, (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words % 64 = 0 := by
    rw [← hcc]; exact h64
  have e1 := hC ht hs hpop hpt hD hDp hupv hu hor h64 L regs hregs hz o hg hg2 hj hr
  have e2 := hC ht hs' hpop' hpt hD' hDp' hupv' hu' hor' h64' L' regs' hregs' hz' o' hg' hg2' hj' hr'
  have hp : HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default) =
      HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default) := by rw [hcc]
  refine commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hp] at e
    exact hne (List.append_cancel_left e)
  · rw [hidx, hcc]


end Any

end FlockSoundness.Discharge.Hidden
