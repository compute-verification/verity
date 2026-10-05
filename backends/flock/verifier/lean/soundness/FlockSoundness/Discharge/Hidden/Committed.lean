import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Discharge.Hidden.Public
import FlockSoundness.Discharge.Hm.Bc
import FlockSoundness.Discharge.Hm.Site
import FlockSoundness.Discharge.Layout.Drawn
import FlockSoundness.Protocol.Discharge.Hidden.Committed
import FlockSoundness.Discharge.Hidden.Delta
import FlockSoundness.Discharge.Hidden.Rows
import FlockSoundness.Discharge.Layout

/-!
# An output row's registered `b ‖ c` (`OutputsCommitted`)

In a satisfying witness of an untyped hidden-output statement, at block `o`, VU `g` and output row `j`:
* the 128 bytes at the row's `hm96` output are the population's `b ‖ c` of the drawn unit there: `HmOut.regions`' region
  `p` pins them (`bc_pinned'`, `Layout.bc_pinned` at the region `regions_get` gives), and `HmOut.drawn` is `HmRow.drawn`
  on the circuit without output ports (`Layout.bcChunk_drawn`): `bc_drawnH`, at every row;
* they are the commit string of the row prefix and the row's message bytes under its salt: the witness satisfies the flat
  twin (`sat_flat`), where `Hm.row_bc_flat` holds;
* the row's message bytes are the output words the VU computes (`msg_bytes`): each message bit is a copy of `outSrc`
  (`msg_copy`), the output word's bit below the port's words and the forced-zero row past them (`Hm.zero_bit`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

section Pinned

open Model Refine Binding FlockSoundness.Discharge.Layout

end Pinned

open Hm Compose Flock.DeriveAll Types.Dag

/-- The bytes at the address region "digest p" pins (`Layout.bcAddr`) are the block's at `bcCol`. -/
theorem bytesAt_bc {st : Stmt} (L : FlockSoundness.Layout st) (R : RowRanges st.c)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) (z : Model.Witness (L.model regs).m)
    (o : Fin (2 ^ (L.model regs).nbl)) {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g p) := by
  unfold Binding.bytesAt Binding.bitsAt bytesOf
  congr 1
  funext j
  congr 1
  funext t
  have hy := bcCol_lt R hg hp j.isLt t.isLt
  have he : Layout.bcAddr st.c g p j.val t.val = bcCol st.c g p j.val t.val :=
    Layout.bcAddr_eq_slotCol _ _ _ _ _ (perVu_hm_pow2 R)
  have hin : Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val) = ⟨bcCol st.c g p j.val t.val, hy⟩ := by
    rw [he]
    simp only [Layout.inBlockOr, hy, ↓reduceDIte]
  show FlockSoundness.bit ((L.model regs).block z o
    (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) = _
  rw [hin]
  exact (blockBit_of_lt _ z o hy).symm

/-- **The `b ‖ c` a satisfying witness of a drawn hidden-output statement holds at a proved unit's `hm96` slot** is the
unit's `b ‖ c` in the registered population's public file, at every row `p`, input or output (`Layout.bc_drawn` at
`Stmt.setupHidden`'s statement): region `p` of `HmOut.regions` pins it (`regions_get`, `bc_pinned'`), and
`HmOut.drawn` is `HmRow.drawn` on the circuit without output ports. -/
theorem bc_drawnH {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) {pop : Public}
    (hpop : HmOut.loadPublic tags st.c pf = .ok pop) (hpt : pop.tables = none) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population) (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hr : o.val * st.c.g + g < D.proved.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList := by
  have s := setupHidden_spec' h
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hpop] at hpop'
  cases hpop'
  have hdrawn : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hdr d rfl
  have hload := loadPublic_nil hpop
  have hsz := (Layout.loadPublic_digests hload hpt).1
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := Layout.drawn_spec hdrawn
    rw [hD] at hD'
    cases hD'
    exact hpn
  have hchunk : ∀ g' (hr' : o.val * st.c.g + g' < D.proved.size),
      Layout.bcChunk st.c st.pub o.val g' p = pop.digest D.proved[o.val * st.c.g + g'] p st.c.ports.size :=
    fun g' hr' => Layout.bcChunk_drawn hdrawn hD hpt hsz hlt hr' hp
  have hsz128 : ∀ g' ≤ g, (Layout.bcChunk st.c st.pub o.val g' p).size = 128 := fun g' hg' => by
    have hr' : o.val * st.c.g + g' < D.proved.size := by omega
    rw [hchunk g' hr']
    exact Layout.pop_digest_size hload hpt (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp
  obtain ⟨hrsz, hall⟩ := regions_get s.regions
  have hpr : p < st.regions.size := hrsz ▸ hp
  rw [← hchunk g hr]
  exact bc_pinned' L regs hregs hz o hg hg2 (Array.getElem_mem_toList hpr) (hall _ hpr) hsz128

section Msg

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (L0 : FlockSoundness.Layout (flatTwin st))
  (F0 : Aliased.UnitFacts tags (flatTwin st))
  (hv1 : ∀ p ∈ st.c.ports.toList, p.v2 = false ∧ p.bits = 16 * p.words ∧ 0 < p.chunks)
  (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray) (hrows : OutRows st.c)
include L L0 F0 hv1 hda hrows

/-- **An output row's message bytes are the output words its VU computes** (`outBytes`), in every block of a satisfying
witness of the hidden statement whose forced-zero row carries 0. -/
theorem msg_bytes {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    (hzero : blockBit (L.model regs) z o (zeroCol st.c) = false) {g j : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outPorts.size) :
    bytesOf (blockBit (L.model regs) z o) (2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words)
        (Typed.msgCol st.c g (HmOut.nIn st.c + j)) =
      outBytes (blockBit (L.model regs) z o) st.c g j := by
  have hG := outGroup_single hrows.2.1 hj
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
    have := msg_copy L L0 F0 hv1 hda hrows hz o hg hj (k := b.val / 2) (t := 8 * (b.val % 2) + t.val) (by omega)
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

end Msg

theorem outputs_committed' : OutputsCommitted := by
  intro tags cf pf coins tables d partition program st ht h pop hpop hpt D hD hlt ho hv1 L regs hregs z hz o g j hg hg2
    hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := parse_untyped ht s.parse
  have P := parse_facts hp
  have L0 : FlockSoundness.Layout (flatTwin st) := layout_of (st := flatTwin st) P s.pin rfl rfl s.m s.kLog
  have F0 : Aliased.UnitFacts tags (flatTwin st) := unitFacts_of_parse (st := flatTwin st) hp rfl
  have R : RowRanges st.c := RowRanges.ofL (st := flatTwin st) L0 F0.parse F0.check
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (outPairs st.c).toArray := by rw [s.da, hidden_delta']
  have hrows : OutRows st.c := out_rows_v1 hp hcr fun p hp => (hv1 p hp).1
  have hz0 := sat_flat L L0 F0 hv1 hda hrows hz
  have hupv := upv_pos F0.check ho
  have hzero : blockBit (L.model regs) z o (zeroCol st.c) = false := zero_bit L0 F0 hupv (show 0 < st.c.g by omega) hz0 o
  obtain ⟨hpl, -⟩ := hrows.2.2 j hj
  have hbc : bytesOf (blockBit (L.model regs) z o) 128 (bcCol st.c g (HmOut.nIn st.c + j)) =
      Binding.commitString
        ((HmRow.rowPrefix (st.c.ports.getD (HmOut.nIn st.c + j) default).words).data.toList ++
          bytesOf (blockBit (L.model regs) z o) (2 * (st.c.ports.getD (HmOut.nIn st.c + j) default).words)
            (Typed.msgCol st.c g (HmOut.nIn st.c + j)))
        (bytesOf (blockBit (L.model regs) z o) 192 (saltCol st.c g (HmOut.nIn st.c + j))) := by
    have := row_bc_flat L0 F0 rfl hupv hv1 hz0 o hg hpl
    rw [getElem!_eq_getD'] at this
    exact this
  rw [← msg_bytes L L0 F0 hv1 hda hrows hz o hzero hg hj, ← hbc, ← bytesAt_bc L R regs z o hg hpl]
  exact (bc_drawnH h hpop hpt hD hlt L regs hregs hz o hg hg2 hpl hr).symm

end FlockSoundness.Discharge.Hidden
