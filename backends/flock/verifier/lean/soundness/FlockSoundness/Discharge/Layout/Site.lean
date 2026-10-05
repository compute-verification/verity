import FlockSoundness.Discharge.Layout.SharedRows
import FlockSoundness.Discharge.Placed.Keyed
import FlockSoundness.Protocol.Discharge.Layout.Site
import FlockSoundness.Discharge.Aliased.Keys
import FlockSoundness.Discharge.Const.Inst
import FlockSoundness.Discharge.Layout.Drawn
import FlockSoundness.Discharge.Placed.Wiring

set_option Elab.async false

/-!
# A drawn unit's site in its table's accepted statement

`Placed.Accepted T` is a run of `Stmt.setupH` that accepted, whose statement is `T`. At a VU `g` of a block `o` of it:

* `siteAddr a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site`: in a satisfying witness, the 128 bytes at port `p`'s `hm96` output (`bcAddr`) are the population's `b ‖ c`
  of the proved unit at rank `o · c.g + g` (`bc_drawn`, carried to `T`; `bc_siteS` with shared rows, `bc_drawnS`);
  `BcSite` bundles what `bc_siteS` reads, whose regions condition holds at `Refine.regionsOf st` (`regionsOf_mem`);
* `copy_site`: in a satisfying witness, each of the VU's stacked unit inputs whose Δ key is message bit `bit` of byte
  `byte` of row port `q` (`Aliased.srcKey`) holds the witness's bit at that message column (`Typed.msgCol`), when the
  column is in the block (`Aliased.copyPos_block`, `Aliased.copyPos_key`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model

variable {T : Model.Statement}

/-- **The `b ‖ c` at a drawn unit's site is the population's**: in a satisfying witness of an accepted drawn statement
whose regions contain the executable's (`hregs`), at block `o`, VU `g` (below `2^⌊log₂ c.g⌋`) and port `p`, the 128 bytes
at the `hm96` slot's output are the `b ‖ c` the population's public file (`pop`, without shared rows) holds for the
proved unit `u` at rank `o · c.g + g`. -/
theorem bc_site (a : Placed.Accepted T) {d : Lean.Json} (hd : a.draw = some d)
    (hregs : ∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs)
    {pop : Public} (hpop : HmRow.loadPublic a.tags a.st.c a.pf = .ok pop) (hpt : pop.tables = none)
    {D : Draw.UnitDraw} (hD : Draw.UnitDraw.ofJson d = .ok D) (hlt : ∀ u ∈ D.proved, u < D.population)
    {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g p u : ℕ} (hg : g < a.st.c.g)
    (hg2 : g < 2 ^ a.st.c.g.log2) (hp : p < a.st.c.ports.size) (hu : D.proved[o.val * a.st.c.g + g]? = some u) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddr a o (bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  dsimp only at hd hregs hpop hu hg hg2 hp ⊢
  subst hd
  obtain ⟨hr, rfl⟩ := Array.getElem?_eq_some_iff.1 hu
  exact bc_drawn h hpop hpt hD hlt (setupH_layout ht h) regs hregs hz o hg hg2 hp hr

/-- **`BcSite`'s regions at the executable's**: the model statement at `Refine.regionsOf st` (as `Refine.stmtOf` places
it) has every region of `st`. -/
theorem regionsOf_mem (st : Stmt) :
    ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ Refine.regionsOf st :=
  fun _ hr => List.mem_map_of_mem hr

end FlockSoundness.Discharge.Layout
