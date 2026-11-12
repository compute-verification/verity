module

public import Proofs.Properties.Nci.Record
public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordTakes

/-!
# The circuit's θ and blocks

`slot_of_unit`: on a record `read` accepts, every unit has a slot below the record's number of slots. A unit of a
template is below the units of the text's template of that id (`unitAt_iff`), the text's template ids are distinct
(`templatesOf_ranges`), so one of the template's grids holds the unit (`gridsOf_covers`), and `slotIn` reads the slot
off the first grid that does, a remainder by the slots, of which there is at least one (`gridsOf_ok`).

`names_divides`: the two things the registered circuit takes beside the wired circuit `D`. θ is the gates of `D` that
are leaves of a row `held` holds; each is an input gate, since a leaf of a reader row is one (`OwnRows`) and `D`'s
gates have the Program's kinds (`LaysOut`). The blocks give an input gate none and a computed gate the slot of a unit
it lies in: a computed gate lies in a unit (`GatesOwn`), and every unit has a slot below `n`. The division takes, by
choice, a unit the gate lies in; that no gate lies in two units is not used.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row)
open NciCheck.Record.SecurityProofs (read_front finish_text front_text gridsOf_covers gridsOf_ok)
open Lean (Json)

/-- **Every unit has a slot**, below the record's number of slots, on a record `read` accepts. -/
theorem slot_of_unit {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c)
    {ts : Array String} {rs : Array TRange}
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg c.f.P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt c.f.P rs tid u sc) : ∃ s, c.slot tid u = some s ∧ s < c.S := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, hg, -⟩ := finish_text hc
  have hts := front_text hf
  obtain ⟨hnd, hmap⟩ := templatesOf_ranges hts hrs
  obtain ⟨i, t, -, -, ht, htt, -, hut, -⟩ := (unitAt_iff hts hrs tid u sc).1 hu
  have hd : (c.f.ts.map (·.tid)).Nodup := by
    have e : c.f.ts.map (·.tid) = rs.toList.map (·.template) := by
      have := congrArg (List.map Prod.fst) hmap
      simpa [List.map_map, Function.comp_def] using this
    rw [e]
    exact hnd
  obtain ⟨g, hgm, hlo, hhi⟩ := gridsOf_covers hg hd (List.mem_of_getElem? ht) hut
  rw [htt] at hgm
  have hS := (gridsOf_ok hg).1
  have hex : ∃ g', ((c.grids.lookup tid).getD []).find? (fun g => g.start ≤ u && u < g.start + g.units) = some g' :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨g, hgm, by simp [hlo, hhi]⟩)
  obtain ⟨g', hfind⟩ := hex
  refine ⟨(u - g'.start) / g'.q % c.S, ?_, Nat.mod_lt _ (by omega)⟩
  simp only [NciCheck.Record.Case.slot, NciCheck.RecordText.slotIn, hfind, Option.map_some]

/-- **θ and the blocks of a wired circuit.** Where `D` is the wired circuit (`LaysOut`), every unit's rows are its own
input gates (`OwnRows`), every computed gate lies in a unit (`GatesOwn`) and every unit has a slot below `n`: some θ
of input gates is the gates of the rows `held` holds (`Names`), and some division of `D`'s computed gates into `n`
blocks gives each gate its unit's slot (`Divides`). -/
theorem names_divides {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}
    {D : Verity.Circuit} {at_ : Fin D.N → ℕ} (hL : LaysOut P rs W S D at_) (hO : OwnRows P rs W)
    (hG : GatesOwn P rs) (held : Row → Bool) (slot : String → ℕ → Option ℕ) (n : ℕ)
    (hslot : ∀ tid u sc, UnitAt P rs tid u sc → ∃ s, slot tid u = some s ∧ s < n) :
    ∃ θ : Finset (Fin D.N), (∀ g ∈ θ, D.isInput g = true) ∧ ∃ U : Verity.Partition D n,
      Names P rs W held at_ θ ∧ Divides P rs slot at_ U := by
  classical
  obtain ⟨-, hgates, hkind⟩ := hL
  obtain ⟨hown, -⟩ := hO
  have hin : ∀ x p ops sc, P.gate (at_ x) = .ok (p, ops, sc) →
      D.isInput x = Flock.Program.isInputId (P.defs[p]!).key := by
    intro x p ops sc hp
    obtain ⟨p', ops', sc', hp', -, hk, -⟩ := hkind x
    rw [hp] at hp'
    cases hp'
    exact hk
  refine ⟨Finset.univ.filter fun x => ∃ r l lo hi, held r = true ∧ RowLeaf P rs W r l (at_ x) lo hi, ?_, ?_⟩
  · intro g hg
    obtain ⟨r, l, lo, hi, -, hrow⟩ := (Finset.mem_filter.1 hg).2
    obtain ⟨p, ops, sc, hp, hk⟩ := hown r l (at_ g) lo hi hrow
    rw [hin g p ops sc hp, hk]
  · have key : ∀ x : Fin D.N, ∃ o : Option (Fin n), (o = none ↔ D.isInput x = true) ∧
        (D.isInput x = false → ∃ tid u sc s, UnitAt P rs tid u sc ∧ sc.off ≤ at_ x ∧
          at_ x < sc.off + (P.defs[sc.d]!).gates ∧ slot tid u = some s ∧ o.map Fin.val = some s) := by
      intro x
      cases hx : D.isInput x with
      | true => exact ⟨none, ⟨fun _ => rfl, fun _ => rfl⟩, fun h => by cases h⟩
      | false =>
        obtain ⟨p, ops, sc, w, hp, -, -, hlies⟩ := hG (at_ x) ((hgates (at_ x)).1 ⟨x, rfl⟩).1
        obtain ⟨tid, u, sc', hu, hlo, hhi⟩ := hlies (by rw [← hin x p ops sc hp]; exact hx)
        obtain ⟨s, hs, hsn⟩ := hslot tid u sc' hu
        exact ⟨some ⟨s, hsn⟩, ⟨(fun h => by cases h), (fun h => by cases h)⟩,
          fun _ => ⟨tid, u, sc', s, hu, hlo, hhi, hs, rfl⟩⟩
    obtain ⟨block, hblock⟩ := Classical.axiomOfChoice key
    exact ⟨⟨block, fun g => (hblock g).1⟩,
      fun x => ⟨fun h => (Finset.mem_filter.1 h).2, fun h => Finset.mem_filter.2 ⟨Finset.mem_univ x, h⟩⟩,
      fun x hx => (hblock x).2 hx⟩

end NciCheck.Properties.Record.SecurityProofs
