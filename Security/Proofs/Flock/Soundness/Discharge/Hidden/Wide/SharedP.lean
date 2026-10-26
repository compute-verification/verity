import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Shared
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP

/-!
# The registered `b ‖ c` at a drawn unit, at public inputs

`HmIn.loadPublic` and `HmIn.drawn` are `HmRow`'s on the circuit whose output ports are the public input words
(`inRows`), so the `b ‖ c` a drawn statement's regions pin at a proved unit's `hm96` slot is the unit's in the
population's public file (`bc_drawnWP`, `bc_drawnW` at `Hidden.SpecP`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open FlockSoundness.Discharge.Layout Refine

/-- The circuit `HmIn.loadPublic` and `HmIn.drawn` read the public file under: its output ports the public input words. -/
abbrev inRows (c : Circuit) (pi : HmIn.Ports) : Circuit :=
  { c with outPorts := if pi.ports.isEmpty then #[] else #[("public inputs", pi.words)] }

/-- `HmIn.loadPublic` is `HmRow.loadPublic` on `inRows`, then checks. -/
theorem loadPublic_inRows {tags : Tags} {c : Circuit} {pi : HmIn.Ports} {pf : ByteArray} {pop : Public}
    (h : HmIn.loadPublic tags c pi pf = .ok pop) : HmRow.loadPublic tags (inRows c pi) pf = .ok pop := by
  unfold HmIn.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

/-- **`bc_drawnW` at public inputs**: the `b ‖ c` the regions pin at a proved unit's `hm96` slot, in a satisfying
witness of a drawn hidden-output statement at public inputs, is the unit's `b ‖ c` in the population's public file, at
every row. -/
theorem bc_drawnWP {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {own : Option ByteArray} {pi : HmIn.Ports} {st : Stmt} (s : SpecP tags cf pf coins tables (some d) own pi st)
    {pop : Public} (hpop : HmIn.loadPublic tags st.c pi pf = .ok pop) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population) (L : FlockSoundness.Layout st)
    (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hr : o.val * st.c.g + g < D.proved.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (Layout.inBlockOr st.c.kLog (Layout.bcAddr st.c g p j.val t.val))) =
      (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList := by
  obtain ⟨pop', hpop', -, -, hdr⟩ := s.pub
  rw [hpop] at hpop'
  cases hpop'
  have hdrawn : HmRow.drawn (inRows st.c pi) pop d = .ok st.pub := hdr d rfl
  have hload := loadPublic_inRows hpop
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
  obtain ⟨ro, ri, hro, -, hst⟩ := s.regions
  obtain ⟨hrsz, hall⟩ := regions_get hro
  have hpr : p < ro.size := hrsz ▸ hp
  have hmem : ro[p] ∈ st.regions.toList := by
    rw [hst, Array.toList_append]
    exact List.mem_append_left _ (Array.getElem_mem_toList hpr)
  rw [← hchunk g hr]
  exact bc_pinned' L regs hregs hz o hg hg2 hmem (hall _ hpr) hsz128

end FlockSoundness.Discharge.Hidden
