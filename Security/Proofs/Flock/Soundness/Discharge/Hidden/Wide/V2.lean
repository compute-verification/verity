import Proofs.Flock.Soundness.Discharge.Hidden.Wide.RowBc
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Tail

/-!
# `sha512/row/v1` or `sha512/row/v2` rows (`OutputsCommittedV2`, `OutputsBindingV2`)

`outputs_committed_tail'`'s argument, for any rows: the flat twin's Δ is `flatPairsAll` (`delta_flat_all`), whose copies
at an output row's message bit are none (`msg_filterAll`: an output row's message has no constant bit before its last
block), so a satisfying witness of the hidden statement satisfies its flat twin (`sat_flatAll`) and holds each output
row's message bit's copy (`msg_copyAll`); the row's `b ‖ c` is under its own prefix (`row_bc_tgt`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Hm Compose Flock.DeriveAll Types.Dag

/-- Unsegmented rows, one output row per output port, each after the input rows, of `2 · words = 128 · chunks` bytes
and `16 · words` bits (`out_rows_whole`). -/
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
    (hu : ∀ p ∈ c.ports.toList, p.segs = none) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (h64 : ∀ j < c.outPorts.size, (c.ports.getD (HmOut.nIn c + j) default).words % 64 = 0) : OutRowsW c :=
  have h := out_rows_whole hp hc hu hr h64
  ⟨⟨hu, hr, fun j hj => ⟨(h j hj).1, (h j hj).2.1⟩⟩, fun j hj => (h j hj).2.2⟩

/-! ## The flat Δ at an output row's message bit -/

section Twin

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
include L F

/-- **`flatPairsAll` at a message bit of a row with no constant message bit before its last block**: none. -/
theorem flat_count_msgAll (hrw : RowWires st.c) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hkb : st.c.ports[p]!.chunks ≤ st.c.ports[p]!.bits / 1024) (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8)
    (j : ℕ) : (flatPairsAll st.c st.pin).count (deltaMsgCol st.c g p b t, j) = 0 := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hb ht
  rw [flatAll_count L F hrw (Or.inl rfl) hat (shaNet_ofRows R.rowNets) (by rw [shaNet_inRows R.rowNets]; omega) j,
    msg_filterAll R hu hg hp hkb hb ht]
  rfl

/-- **The flat Δ's row at such a message bit is a self row**, in `A` and in `B`, for any rows. -/
theorem flat_selfAll (hda : st.da = (HmRow.delta st.c st.pin).1) (hu : ∀ p ∈ st.c.ports.toList, p.segs = none) {g p b t : ℕ} (hg : g < st.c.g)
    (hp : p < st.c.ports.size) (hkb : st.c.ports[p]!.chunks ≤ st.c.ports[p]!.bits / 1024)
    (hb : b < 128 * st.c.ports[p]!.chunks) (ht : t < 8) (hx : deltaMsgCol st.c g p b t < 2 ^ st.c.kLog)
    (j : Fin (2 ^ st.c.kLog)) :
    placedA st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) ∧
      placedB st ⟨deltaMsgCol st.c g p b t, hx⟩ j = (if j.val = deltaMsgCol st.c g p b t then 1 else 0) := by
  have R := RowRanges.ofL L F.parse F.check
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hu hg hp hb ht
  have hfc := flat_count_msgAll L F F.parse.rowWires hu hg hp hkb hb ht
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
  obtain ⟨hat, hoff⟩ := deltaMsgCol_at R hrows.1.1 hg hp hb ht8
  have hx := deltaMsgCol_lt R hrows.1.1 hg hp hb ht8
  have hfc := flat_count_msgAll L0 F0 F0.parse.rowWires hrows.1.1 hg hp (hrows.kb hj) hb ht8
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
        delta_flat_all _ _ F0.parse.typed hrows.1.1, hfc, Nat.zero_add, outPairs_count R hrows.rows hg hj hk ht]
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
    have hx := deltaMsgCol_lt R hrows.1.1 hg hp hb ht8
    have hie : i = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ :=
      Fin.ext (hqi.symm.trans hq1)
    subst hie
    have hrow : ∀ i' : Fin (2 ^ st.c.kLog),
        (L0.model regs).A₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) ∧
        (L0.model regs).B₀ ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ i' =
          (if i' = ⟨deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8), hx⟩ then 1 else 0) := by
      intro i'
      have hs := flat_selfAll L0 F0 rfl hrows.1.1 hg hp (hrows.kb hj) hb ht8 hx i'
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

theorem outputs_committed_v2' : OutputsCommittedV2 := by
  intro tags cf pf coins tables d partition program st ht hi h pop hpop hpt D hD hlt hupv hu hor h64 L regs hregs z hz o g
    j hg hg2 hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := parse_untyped ht (s.parseF hi)
  have P := parse_facts hp
  have L0 : FlockSoundness.Layout (flatTwin st) := layout_of (st := flatTwin st) P s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (flatTwin st) := unitFacts_of_parse (st := flatTwin st) hp rfl
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := by
    rw [s.da, hidden_delta_unpinned (unpinned_of_rep hp hcr hor)]
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
    have := row_bc_tgt L0 F0 rfl hupv (hz0.ex _) o hg hpl (by rw [getElem!_eq_getD']; exact hb)
      (by rw [getElem!_eq_getD']; exact hw.ge) (by rw [getElem!_eq_getD']; exact h2) hu
    rw [getElem!_eq_getD'] at this
    exact this
  rw [← msg_bytesAll L L0 F0 hda hrows hz o hzero hg hj, ← hbc, ← bytesAt_bc L R regs z o hg hpl]
  exact (bc_drawnH h hpop hpt hD hlt L regs hregs hz o hg hg2 hpl hr).symm

/-- **`OutputsBindingV2` from `OutputsCommittedV2`**. -/
theorem outputs_binding_v2_of (hC : OutputsCommittedV2) : OutputsBindingV2 := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hi hs hs' pop hpop hpt D D' hD
    hDp hD' hDp' hupv hu hor h64 L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg2 hg' hg2' hj hr hr' hidx hne
  have hcc : st.c = st'.c := setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have hupv' : 0 < st'.c.upv := by rw [← hcc]; exact hupv
  have hj' : j < st'.c.outPorts.size := by rw [← hcc]; exact hj
  have hu' : ∀ p ∈ st'.c.ports.toList, p.segs = none := by rw [← hcc]; exact hu
  have hor' : st'.c.outRowPorts = Array.replicate st'.c.outPorts.size 1 := by rw [← hcc]; exact hor
  have h64' : ∀ j < st'.c.outPorts.size, (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words % 64 = 0 := by
    rw [← hcc]; exact h64
  have e1 := hC ht hi hs hpop hpt hD hDp hupv hu hor h64 L regs hregs hz o hg hg2 hj hr
  have e2 := hC ht hi hs' hpop' hpt hD' hDp' hupv' hu' hor' h64' L' regs' hregs' hz' o' hg' hg2' hj' hr'
  have hp : HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default) =
      HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default) := by rw [hcc]
  refine commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hp] at e
    exact hne (List.append_cancel_left e)
  · rw [hidx, hcc]

end FlockSoundness.Discharge.Hidden
