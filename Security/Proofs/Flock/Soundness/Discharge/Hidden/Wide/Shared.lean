import Proofs.Flock.Soundness.Discharge.Hidden.Wide.V2
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows

/-!
# Shared rows (`OutputsCommittedWide`, `OutputsBindingWide`)

A shared-row population's drawn `b ‖ c` are its proved units' (`Layout.drawn_digestW`, `Layout.bcChunk_drawnW` and
`Layout.pop_digest_sizeW`, in `Layout/SharedRows.lean`). `bc_drawnW` is `bc_drawnH` without `pop.tables = none`, and
`outputs_committed_wide'` is `outputs_committed_v2'`'s argument through it.
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open FlockSoundness.Discharge.Layout

/-! ## The registered `b ‖ c` at a drawn unit -/

/-- **`bc_drawnH` with or without shared rows**: the `b ‖ c` a satisfying witness of a drawn hidden-output statement
holds at a proved unit's `hm96` slot is the unit's `b ‖ c` in the registered population's public file, at every row. -/
theorem bc_drawnW {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st) {pop : Public}
    (hpop : HmOut.loadPublic tags st.c pf = .ok pop) {D : Draw.UnitDraw}
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
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := Layout.drawn_spec hdrawn
    rw [hD] at hD'
    cases hD'
    exact hpn
  have hchunk : ∀ g' (hr' : o.val * st.c.g + g' < D.proved.size),
      Layout.bcChunk st.c st.pub o.val g' p = pop.digest D.proved[o.val * st.c.g + g'] p st.c.ports.size :=
    fun g' hr' => bcChunk_drawnW hload hdrawn hD hlt hr' hp
  have hsz128 : ∀ g' ≤ g, (Layout.bcChunk st.c st.pub o.val g' p).size = 128 := fun g' hg' => by
    have hr' : o.val * st.c.g + g' < D.proved.size := by omega
    rw [hchunk g' hr']
    exact pop_digest_sizeW hload (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp
  obtain ⟨hrsz, hall⟩ := regions_get s.regions
  have hpr : p < st.regions.size := hrsz ▸ hp
  rw [← hchunk g hr]
  exact bc_pinned' L regs hregs hz o hg hg2 (Array.getElem_mem_toList hpr) (hall _ hpr) hsz128

open Hm Compose Flock.DeriveAll Types.Dag

theorem outputs_committed_wide' : OutputsCommittedWide := by
  intro tags cf pf coins tables d partition program st ht h pop hpop D hD hlt hupv hu hor h64 L regs hregs z hz o g j hg
    hg2 hj hr
  have s := setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := parse_untyped ht s.parse
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
    have := row_bc_tgt L0 F0 rfl hupv hz0 o hg hpl (by rw [getElem!_eq_getD']; exact hb)
      (by rw [getElem!_eq_getD']; exact hw.ge) (by rw [getElem!_eq_getD']; exact h2) hu
    rw [getElem!_eq_getD'] at this
    exact this
  rw [← msg_bytesAll L L0 F0 hda hrows hz o hzero hg hj, ← hbc, ← bytesAt_bc L R regs z o hg hpl]
  exact (bc_drawnW h hpop hD hlt L regs hregs hz o hg hg2 hpl hr).symm

/-- **`OutputsBindingWide` from `OutputsCommittedWide`**. -/
theorem outputs_binding_wide_of (hC : OutputsCommittedWide) : OutputsBindingWide := by
  intro tags cf pf coins coins' tables d d' partition partition' program program' st st' ht hs hs' pop hpop D D' hD
    hDp hD' hDp' hupv hu hor h64 L regs hregs L' regs' hregs' z hz z' hz' o o' g g' j hg hg2 hg' hg2' hj hr hr' hidx hne
  have hcc : st.c = st'.c := setupHidden_c hs hs'
  have hpop' : HmOut.loadPublic tags st'.c pf = .ok pop := by rw [← hcc]; exact hpop
  have hupv' : 0 < st'.c.upv := by rw [← hcc]; exact hupv
  have hj' : j < st'.c.outPorts.size := by rw [← hcc]; exact hj
  have hu' : ∀ p ∈ st'.c.ports.toList, p.segs = none := by rw [← hcc]; exact hu
  have hor' : st'.c.outRowPorts = Array.replicate st'.c.outPorts.size 1 := by rw [← hcc]; exact hor
  have h64' : ∀ j < st'.c.outPorts.size, (st'.c.ports.getD (HmOut.nIn st'.c + j) default).words % 64 = 0 := by
    rw [← hcc]; exact h64
  have e1 := hC ht hs hpop hD hDp hupv hu hor h64 L regs hregs hz o hg hg2 hj hr
  have e2 := hC ht hs' hpop' hD' hDp' hupv' hu' hor' h64' L' regs' hregs' hz' o' hg' hg2' hj' hr'
  have hp : HmRow.prefixOf (st.c.ports.getD (HmOut.nIn st.c + j) default) =
      HmRow.prefixOf (st'.c.ports.getD (HmOut.nIn st'.c + j) default) := by rw [hcc]
  refine commitPair_spec' _ _ _ _ (fun e => ?_) (e1.symm.trans (Eq.trans ?_ e2))
  · rw [hp] at e
    exact hne (List.append_cancel_left e)
  · rw [hidx, hcc]

end FlockSoundness.Discharge.Hidden
