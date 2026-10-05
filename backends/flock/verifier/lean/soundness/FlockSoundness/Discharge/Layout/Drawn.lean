import FlockSoundness.Discharge.Layout.HmRegion
import FlockSoundness.Protocol.Discharge.Layout.Drawn

set_option Elab.async false

/-!
# The drawn units' `b ‖ c`: the population's

A drawn statement's public file is `HmRow.drawn` of the registered population's (`setupH_drawn`): its instances are the
draw's proved units, and instance `r`'s `b ‖ c` is proved unit `D.proved[r]`'s in the population (`drawn_digest`, for a
file without shared rows, whose `b ‖ c` are `n · np` pieces of 128 bytes, `loadPublic_digests`; the draw's units are in
the population, `ofJson_lt`). So at a proved unit's `hm96` slot, block `o` and VU `g` with `o · c.g + g = r`, a satisfying
witness holds the population's `b ‖ c` of unit `D.proved[r]` (`bc_drawn`): the registration's, fixed by the public file
before the draw.
-/

namespace FlockSoundness.Discharge.Layout

open Flock Refine

/-- A draw without a closure proves its drawn units, which are in its population. -/
theorem proved_lt {d : Lean.Json} {D : Draw.UnitDraw} (h : Draw.UnitDraw.ofJson d = .ok D) (hcl : D.closure = #[]) :
    ∀ u ∈ D.proved, u < D.population := by
  unfold Draw.UnitDraw.proved
  rw [hcl]
  exact (ofJson_lt h).1

/-- **A drawn statement's `hm96` chunk at block `o`, VU `g`** is, at a proved unit, that unit's population `b ‖ c`. -/
theorem bcChunk_drawn {c : Circuit} {pop pub : Public} {d : Lean.Json} {D : Draw.UnitDraw}
    (h : HmRow.drawn c pop d = .ok pub) (hD : Draw.UnitDraw.ofJson d = .ok D) (hpt : pop.tables = none)
    (hsz : pop.digests.size = pop.n * c.ports.size) (hlt : ∀ u ∈ D.proved, u < D.population)
    {o g p : ℕ} (hr : o * c.g + g < D.proved.size) (hp : p < c.ports.size) :
    bcChunk c pub o g p = pop.digest D.proved[o * c.g + g] p c.ports.size := by
  obtain ⟨D', hD', -, hn, -, -⟩ := drawn_spec h
  rw [hD] at hD'
  cases hD'
  unfold bcChunk Circuit.instOf
  rw [hn]
  simp only [hr, ↓reduceIte]
  exact drawn_digest h hD hpt hsz hlt hr hp

/-- **The `b ‖ c` a satisfying witness of a drawn statement holds at a proved unit's `hm96` slot** is the unit's `b ‖ c`
in the registered population's public file: at block `o`, VU `g` (below `2^⌊log₂ c.g⌋`), input port `p`, the instance
`o · c.g + g` is proved unit `D.proved[o · c.g + g]`, and the 128 bytes at `bcAddr` are its population `b ‖ c`. It holds
in the model's statement at setupH's layout whose regions contain the executable's (`hregs`), for a population file
without shared rows (`hpt`) and a draw whose proved units are in the population (`hlt`; `proved_lt` without a
closure). -/
theorem bc_drawn {tags : Tags} {circuitFile publicFile : ByteArray} {coins : Coins}
    {tables : List (String × ByteArray)} {d : Lean.Json} {partition : Option Lean.Json} {program : Option ByteArray}
    {st : Stmt} (h : Stmt.setupH tags circuitFile publicFile coins tables (some d) partition program = .ok st)
    {pop : Public} (hpop : HmRow.loadPublic tags st.c publicFile = .ok pop) (hpt : pop.tables = none)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (hregs : ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hg2 : g < 2 ^ st.c.g.log2) (hp : p < st.c.ports.size)
    (hr : o.val * st.c.g + g < D.proved.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) =>
        (L.model regs).addr o (inBlockOr st.c.kLog (bcAddr st.c g p j.val t.val))) =
      (pop.digest D.proved[o.val * st.c.g + g] p st.c.ports.size).data.toList := by
  obtain ⟨pop', hpop', hdr⟩ := setupH_drawn h
  rw [hpop] at hpop'
  cases hpop'
  have hsz := (loadPublic_digests hpop hpt).1
  have hpopn : D.population = pop.n := by
    obtain ⟨D', hD', hpn, -⟩ := drawn_spec hdr
    rw [hD] at hD'
    cases hD'
    exact hpn
  rw [← bcChunk_drawn hdr hD hpt hsz hlt hr hp]
  refine bc_pinned L regs hregs (setupH_regions h) hz o hg hg2 hp fun g' hg' => ?_
  have hr' : o.val * st.c.g + g' < D.proved.size := by omega
  rw [bcChunk_drawn hdr hD hpt hsz hlt hr' hp]
  exact pop_digest_size hpop hpt (by rw [← hpopn]; exact hlt _ (Array.getElem_mem hr')) hp

end FlockSoundness.Discharge.Layout
