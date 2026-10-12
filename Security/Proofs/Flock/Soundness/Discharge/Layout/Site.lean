import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Placed.Keyed

set_option Elab.async false

/-!
# A drawn unit's site in its table's accepted statement

`Placed.Accepted T` is a run of `Stmt.setupH` that accepted, whose statement is `T`. At a template instance `g` of a
block `o` of it:

* `siteAddr a o x`: in-block column `x` of block `o`, as an address of `T`;
* `bc_site`: in a satisfying witness, the 128 bytes at port `p`'s `hm96` output (`bcAddr`) are the population's `b ‖ c`
  of the proved unit at rank `o · c.g + g` (`bc_drawn`, carried to `T`; `bc_siteS` with shared rows, `bc_drawnS`);
  `BcSite` bundles what `bc_siteS` reads, whose regions condition holds at `Refine.regionsOf st` (`regionsOf_mem`);
* `copy_site`: in a satisfying witness, each of the template instance's stacked unit inputs whose Δ key is message bit
  `bit` of byte `byte` of row port `q` (`Aliased.srcKey`) holds the witness's bit at that message column
  (`Typed.msgCol`), when the column is in the block (`Aliased.copyPos_block`, `Aliased.copyPos_key`).
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Model

variable {T : Model.Statement}

/-- **In-block column `x` of block `o`** of an accepted statement, as an address of its statement `T` (column 0 past
the block). -/
def siteAddr (a : Placed.Accepted T) (o : Fin (2 ^ T.nbl)) (x : ℕ) : Fin (2 ^ T.m) :=
  T.addr o (Fin.cast a.kLog_eq (inBlockOr a.st.c.kLog x))

/-- **The `b ‖ c` at a drawn unit's site is the population's**: in a satisfying witness of an accepted drawn statement
whose regions contain the executable's (`hregs`), at block `o`, template instance `g` (below `2^⌊log₂ c.g⌋`) and port `p`, the 128 bytes
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

/-- **`bc_site` with or without shared rows** (`bc_drawnS`). -/
theorem bc_siteS (a : Placed.Accepted T) {d : Lean.Json} (hd : a.draw = some d)
    (hregs : ∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs)
    {pop : Public} (hpop : HmRow.loadPublic a.tags a.st.c a.pf = .ok pop)
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
  exact bc_drawnS h hpop hD hlt (setupH_layout ht h) regs hregs hz o hg hg2 hp hr

/-- **What `bc_site` reads at a site**: block `o` and template instance `g` of an accepted statement whose draw `d` decodes to `D`, whose
proved units are in its population; whose model statement's regions contain the executable's; whose public file loads to
`pop`, with or without shared rows; with `g` below `2^⌊log₂ c.g⌋`; and whose proved unit at rank `o · c.g + g` is `u`. -/
def BcSite (a : Placed.Accepted T) (pop : Public) (o : Fin (2 ^ T.nbl)) (g u : ℕ) : Prop :=
  ∃ d D, a.draw = some d ∧ Draw.UnitDraw.ofJson d = .ok D ∧ (∀ v ∈ D.proved, v < D.population) ∧
    (∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs) ∧
    HmRow.loadPublic a.tags a.st.c a.pf = .ok pop ∧ g < 2 ^ a.st.c.g.log2 ∧
    D.proved[o.val * a.st.c.g + g]? = some u

/-- `bc_siteS` at a `BcSite`. -/
theorem bc_site_of {a : Placed.Accepted T} {pop : Public} {o : Fin (2 ^ T.nbl)} {g u : ℕ} (hs : BcSite a pop o g u)
    {z : Witness T.m} (hz : T.Satisfies z) (hg : g < a.st.c.g) {p : ℕ} (hp : p < a.st.c.ports.size) :
    Binding.bytesAt z (fun (j : Fin 128) (t : Fin 8) => siteAddr a o (bcAddr a.st.c g p j.val t.val)) =
      (pop.digest u p a.st.c.ports.size).data.toList := by
  obtain ⟨d, D, hd, hD, hlt, hregs, hpop, hg2, hu⟩ := hs
  exact bc_siteS a hd hregs hpop hD hlt hz o hg hg2 hp hu

/-- **`BcSite`'s regions at the executable's**: the model statement at `Refine.regionsOf st` (as `Refine.stmtOf` places
it) has every region of `st`. -/
theorem regionsOf_mem (st : Stmt) :
    ∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ Refine.regionsOf st :=
  fun _ hr => List.mem_map_of_mem hr

/-- **A drawn unit's message-keyed input holds its row's bit**: in a satisfying witness of an accepted statement, at
block `o` and template instance `g`, the template instance's stacked unit input `x` whose Δ key is message bit `bit` of
byte `byte` of row port `q` holds the witness's bit at that message column (`Typed.msgCol`), when the column is in the
block. -/
theorem copy_site (a : Placed.Accepted T) {z : Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl)) {g : ℕ}
    (hg : g < a.st.c.g) (x : Fin a.rows.w) (hx : x.val < a.rows.nIn) {q byte bit : ℕ}
    (hk : Aliased.srcKey a.st.c x.val = .msg q byte bit) (hin : Typed.msgCol a.st.c g q byte bit < 2 ^ a.st.c.kLog) :
    T.block z o (a.col hg x) = z (siteAddr a o (Typed.msgCol a.st.c g q byte bit)) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  dsimp only at hg x hx hk hin ⊢
  have F := Aliased.setupH_unitFacts ht h
  have hb := Aliased.copyPos_block (setupH_layout ht h) regs F hg x hx hz o
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    rw [← (setupH_layout ht h).unitRows_nIn]; exact hx
  rcases Aliased.copyPos_key (setupH_layout ht h) regs F hg hx' (by simp [hk]) (fun k off => by simp [hk]) with
    he | ⟨hn, -⟩
  · simp only [Placed.Accepted.col, siteAddr, Fin.cast_eq_self]
    rw [hb]
    show z _ = z _
    congr 2
    apply Fin.ext
    rw [he, hk]
    simp [Aliased.realize, inBlockOr, hin]
    rfl
  · exact absurd (by rw [hk]; exact hin) hn

end FlockSoundness.Discharge.Layout
