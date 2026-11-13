module

public import Proofs.Properties.Nci.RecordBlocks
public import Proofs.Properties.Nci.RecordGrid
public import Proofs.Properties.Nci.RecordInputs
public import Proofs.Properties.Nci.RecordSameSlot
public import Proofs.Properties.Nci.RecordWire
public import Proofs.Properties.Nci.RecordWireRead

/-!
# What a row's kind says of the row

The record check classes every reader row by what fills it (`Record.Case.rowKind`). Two kinds are not fresh, and
each says something of the row, on a record the check reads:

* `theta_held`: a row of kind `theta` is held. The check gives the reason it counts as θ (`Record.Case.heldAt`).
* `inside_writer`: a row of kind `inside` has a writer, and the writer's unit is in the row's slot.

Both read what fills the row off the declarations. No two declarations fill one row (`inputsOf_decls`), so what
fills a row is the fill of any declaration that covers it (`fill_of_cover`, by `cover_apart`: two disjoint wires do
not both cover a row). A row of kind `inside` is filled by a wire's declaration whose wire the check found to stay
inside a slot (`Record.sameSlot`); the row's writer is that wire's, at the row's index, and the test is sound there
(`sameSlot_sound`) for a wire with a step for each digit and no digit of radix zero (`text_digits`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.Record (Case Front read finish sameSlot openingHeld Kind kindOf)
open NciCheck.Record.SecurityProofs (read_front finish_text slotOf OneGrid slotIn_eq gridsOf_lookup sameSlot_sound
  text_digits)
open NciCheck.RecordText (Row Decl Fill Inputs fillAt)
open Lean (Json)

/-! ## One declaration fills a row -/

/-- **Two disjoint wires do not both cover a row.** -/
theorem cover_apart {a b : Spot.Wires.Wire} (h : Spot.Wires.disjoint a b = true) {t : String} {u : ℕ} {p : String}
    {row : ℕ} (ha : (a.indexOf t u p row).isSome = true) (hb : (b.indexOf t u p row).isSome = true) : False := by
  obtain ⟨i, hi⟩ := Option.isSome_iff_exists.1 ha
  obtain ⟨i', hi'⟩ := Option.isSome_iff_exists.1 hb
  obtain ⟨h1, h2, h3, h4, h5, h6, -⟩ := indexOf_some hi
  obtain ⟨h1', h2', h3', h4', h5', h6', -⟩ := indexOf_some hi'
  simp only [Spot.Wires.disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact h (h1.trans h1'.symm)
  · exact h (h2.trans h2'.symm)
  all_goals omega

/-- **What fills a row is the fill of any declaration that covers it**, where no two declarations fill one row. -/
theorem fill_of_cover : ∀ {ds : List Decl}, Spot.Wires.pairwiseDisjoint (ds.map (·.w)) = true → ∀ {d : Decl},
    d ∈ ds → ∀ {r : Row}, (d.w.indexOf r.tid r.unit r.port r.row).isSome = true → fillAt ds r = some d.fill
  | [], _, _, hm, _, _ => by cases hm
  | a :: ds, hd, d, hm, r, hc => by
    simp only [List.map_cons, Spot.Wires.pairwiseDisjoint, Bool.and_eq_true, List.all_eq_true] at hd
    obtain ⟨hall, hrest⟩ := hd
    show ((a :: ds).find? fun d => (d.w.indexOf r.tid r.unit r.port r.row).isSome).map (·.fill) = some d.fill
    rw [List.find?_cons]
    rcases List.mem_cons.1 hm with hda | hm'
    · subst hda
      simp only [hc]
      rfl
    · have hna : (a.w.indexOf r.tid r.unit r.port r.row).isSome = false := by
        cases hx : (a.w.indexOf r.tid r.unit r.port r.row).isSome with
        | false => rfl
        | true => exact (cover_apart (hall d.w (List.mem_map.2 ⟨d, hm', rfl⟩)) hx hc).elim
      simp only [hna]
      exact fill_of_cover hrest hm' hc

/-! ## A row of kind theta is held -/

/-- **A row of kind `theta` is held**: the check gives a reason it counts as θ (`Case.heldAt`). A row a registered
read or a table read fills is of kind `theta` where its parameter is θ, which is when `heldAt` gives the reason; a
row an opening fills is of kind `theta` where the opening is held, which `openingHeld` decides by the opening's
reader being pinned and its table lying in a θ parameter, the two things `heldAt` asks beside. -/
theorem theta_held {f : Front} {pin : String → Option String} {c : Case} (hc : finish f pin = .ok c) (r : Row)
    (hk : c.rowKind r = Kind.theta) : (c.heldAt r).isSome = true := by
  obtain ⟨-, -, -, -, -, hheld⟩ := finish_case hc
  unfold Case.rowKind Case.kind at hk
  unfold Case.heldAt
  cases hfill : fillAt c.x.decls r with
  | none =>
    rw [hfill] at hk
    cases hk
  | some fill =>
    rw [hfill] at hk
    cases fill with
    | wire i =>
      simp only [kindOf] at hk
      split at hk <;> cases hk
    | read p =>
      simp only [kindOf] at hk
      split at hk
      · rename_i hθ
        simp only [Inputs.declares, hθ, ite_true]
        rfl
      · cases hk
    | table name root =>
      simp only [kindOf] at hk
      split at hk
      · rename_i hθ
        cases ht : c.x.tables.find? (fun t => t.name == name) with
        | none =>
          rw [ht] at hθ
          cases hθ
        | some t =>
          rw [ht] at hθ
          have hθ' : c.isTheta t.param = true := hθ
          simp only [Inputs.declares, ht, Option.map_some, hθ', ite_true]
          rfl
      · cases hk
    | opening i =>
      simp only [kindOf] at hk
      split at hk
      · rename_i hi
        have hi0 := hi
        rw [hheld, List.getElem?_toArray, List.getElem?_map] at hi
        cases ho : c.x.openings[i]? with
        | none =>
          rw [ho] at hi
          cases hi
        | some o =>
          rw [ho] at hi
          have hoh : openingHeld (fun tid => (c.pins.lookup tid).isSome) c.isTheta c.x o = true := hi
          unfold openingHeld at hoh
          simp only [Bool.and_eq_true] at hoh
          obtain ⟨⟨⟨hpin, htab⟩, -⟩, -⟩ := hoh
          cases ht : c.x.tables.find? (fun t => t.name == o.table) with
          | none =>
            rw [ht] at htab
            cases htab
          | some t =>
            rw [ht] at htab
            have hθ' : c.isTheta t.param = true := htab
            obtain ⟨pn, hpn⟩ := Option.isSome_iff_exists.1 hpin
            simp only [Inputs.declares, ho, ht, Option.bind_some, Option.map_some, hi0, hθ', Bool.and_self, ite_true,
              hpn]
            rfl
      · cases hk
    | ingress =>
      simp only [kindOf] at hk
      cases hk

/-! ## A row of kind inside has its writer in its slot -/

/-- **A row of kind `inside` has a writer, in the row's slot.** The row is filled by a wire's declaration, the wire
one the check found to stay inside a slot (`Record.sameSlot`); the row's writer is that wire's (no two declarations
fill one row), at the row's index, where `sameSlot` is sound (`sameSlot_sound`). -/
theorem inside_writer {j tr : Json} {c : Case} (h : read j tr = .ok c) (r : Row)
    (hk : c.rowKind r = Kind.inside) :
    (c.wired.writerOf r.tid r.unit r.port r.row).isSome = true ∧
      ∀ wt wu out s s', c.wired.writerOf r.tid r.unit r.port r.row = some (wt, wu, out) →
        c.slot r.tid r.unit = some s → c.slot wt wu.toNat = some s' → s' = s := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, hg, text, htext, hx, hcuts, hwires⟩ := finish_text hc
  obtain ⟨hchk, -, -, hinside, -, -⟩ := finish_case hc
  obtain ⟨hdisj, rest, hdecls, hrest⟩ := inputsOf_decls hx
  have hS := (Spot.Wires.check_sound hchk).1
  unfold Case.rowKind Case.kind at hk
  cases hfill : fillAt c.x.decls r with
  | none =>
    rw [hfill] at hk
    cases hk
  | some fill =>
    rw [hfill] at hk
    cases fill with
    | wire i =>
      have hin : c.inside[i]?.getD false = true := by
        simp only [kindOf] at hk
        split at hk
        · assumption
        · cases hk
      -- the declaration that fills the row is wire `i`'s, and the wire covers the row
      have hfill0 := hfill
      unfold fillAt at hfill
      obtain ⟨d, hd, hdf⟩ := Option.map_eq_some_iff.1 hfill
      have hdm := List.mem_of_find?_eq_some hd
      have hcov0 := List.find?_some hd
      have hcov : (d.w.indexOf r.tid r.unit r.port r.row).isSome = true := hcov0
      rw [hdecls] at hdm
      rcases List.mem_append.1 hdm with hdm | hdm
      · obtain ⟨⟨w, i'⟩, hwi, rfl⟩ := List.mem_map.1 hdm
        have hi : i' = i := by
          injection hdf
        subst hi
        have hget : text.wires[i']? = some w := List.mem_zipIdx_iff_getElem?.1 hwi
        have hwm : w ∈ c.W.wires := by
          rw [hwires]
          exact List.mem_of_getElem? hget
        have hss : sameSlot c.S (c.gridsAt w.reader) (c.gridsAt w.writer) w = true := by
          rw [hinside, hS, hwires, List.getElem?_toArray, List.getElem?_map, hget] at hin
          exact hin
        obtain ⟨idx, hidx⟩ := Option.isSome_iff_exists.1 hcov
        have hidx' : w.indexOf r.tid r.unit r.port r.row = some idx := hidx
        constructor
        · unfold Spot.Wires.Summary.writerOf
          rw [List.findSome?_isSome_iff]
          refine ⟨w, by rw [hS]; exact hwm, ?_⟩
          rw [hidx']
          rfl
        · intro wt wu out s s' hw hs hs'
          obtain ⟨w', hw'm, idx', hidx'', rfl, rfl, rfl⟩ := writerOf_wire hchk hw
          -- the writer's wire is wire `i`
          have hw'e : w' = w := by
            rw [hwires] at hw'm
            obtain ⟨i'', hi''⟩ := List.mem_iff_getElem?.1 hw'm
            have hd'm : ({ w := w', bits := 16 * NciCheck.RecordText.rowWords c.f.ts text w', fill := .wire i'' } : Decl)
                ∈ c.x.decls := by
              rw [hdecls]
              exact List.mem_append_left _ (List.mem_map.2 ⟨(w', i''), List.mem_zipIdx_iff_getElem?.2 hi'', rfl⟩)
            have hf' := fill_of_cover hdisj hd'm (r := r) (by
              show (w'.indexOf r.tid r.unit r.port r.row).isSome = true
              rw [hidx'']
              rfl)
            rw [hfill0] at hf'
            have hii : i' = i'' := by
              injection hf' with hf'
              injection hf'
            rw [← hii, hget] at hi''
            exact (Option.some.inj hi'').symm
          subst hw'e
          have hie : idx' = idx := Option.some.inj (hidx''.symm.trans hidx')
          subst hie
          obtain ⟨rt, wtm, hsh, -⟩ := wire_shape hchk w' hw'm
          obtain ⟨-, -, p, ow, -, -, hc1, -, hlast, hmod, -⟩ := shape_spec hsh
          obtain ⟨hlen, hrad⟩ := text_digits htext w' (by rw [← hwires]; exact hw'm)
          obtain ⟨hrd, hpt, hu1, hu2, hr1, hr2, hie⟩ := indexOf_some hidx'
          have hlt := Spot.Wires.indexOf_lt hmod hidx'
          have hgr : ∀ tid, OneGrid (c.gridsAt tid) := fun tid =>
            (gridsOf_lookup hg tid).1.imp fun hab => Or.inl hab
          have hunit : (w'.readerAt idx').1 = r.unit := by
            show w'.start + idx' / w'.cuts = r.unit
            rw [hie, Nat.add_comm ((r.unit - w'.start) * w'.cuts), Nat.add_mul_div_right _ _ (by omega),
              Nat.div_eq_of_lt (by omega)]
            omega
          refine (sameSlot_sound c.S (c.gridsAt w'.reader) (c.gridsAt w'.writer) w' (hgr _) (hgr _) hlen hrad hc1
            hlast hss idx' hlt s s' ?_ ?_).symm
          · rw [hunit, hrd]
            exact hs
          · exact hs'
      · exact absurd hdf ((hrest d hdm).1 i)
    | read p =>
      simp only [kindOf] at hk
      split at hk <;> cases hk
    | table name root =>
      simp only [kindOf] at hk
      split at hk <;> cases hk
    | opening i =>
      simp only [kindOf] at hk
      split at hk <;> cases hk
    | ingress =>
      simp only [kindOf] at hk
      cases hk

end NciCheck.Properties.Record.SecurityProofs
