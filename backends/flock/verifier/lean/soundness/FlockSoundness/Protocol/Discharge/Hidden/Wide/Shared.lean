import FlockSoundness.Discharge.Layout.SharedRows
import FlockSoundness.Protocol.Discharge.Hidden.Wide.RowBc
import FlockSoundness.Protocol.Discharge.Hidden.Committed

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

open Hm Compose Flock.DeriveAll

end FlockSoundness.Discharge.Hidden
