module

public import Proofs.Properties.Nci.RecordRead
public import Spot.WiresSound

/-!
# The record's declarations and its wires

`RecordText.inputsOf` reads what the record declares to fill its units' rows. Its declarations begin with the
wires' own, one for each wire in the wires' order, and every later one is appended: `inputsOf_wires`. The walk is
down `inputsOf`'s stages, and through each of its loops by one invariant: a `for` loop in `Except` whose body never
breaks keeps what its body keeps (`forIn_keeps`).

On a record `Record.read` accepts:

* `wired_of_read`: the summary the check holds is what `Spot.Wires.check` gives of the wiring it holds, read off
  `Record.finish`.
* `held_unwired`: a row the check holds in θ has no writer. What fills a row is the first declaration that covers it
  (`RecordText.fillAt`), the wires' declarations come first, and a wire's declaration declares the row to hold
  nothing (`RecordText.Inputs.declares`); so a row a wire covers is filled by a wire's declaration and is not held.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open NciCheck.Record (Case Front read finish unwhole rolesOf sameSlot isTheta openingHeld)
open NciCheck.Record.SecurityProofs (read_front finish_text)
open NciCheck.RecordText (Row Tmpl Decl Fill Inputs inputsOf rowWords fillAt)
open Lean (Json)

/-! ## The declarations begin with the wires' -/

/-- The array `a` begins with the list `D`. -/
private def Begins (D : List Decl) (a : Array Decl) : Prop := ∃ rest, a.toList = D ++ rest

private theorem begins_push {D : List Decl} {a : Array Decl} (d : Decl) (h : Begins D a) : Begins D (a.push d) := by
  obtain ⟨rest, hr⟩ := h
  exact ⟨rest ++ [d], by rw [Array.toList_push, hr, List.append_assoc]⟩

/-- A `for` loop in `Except` whose body never breaks keeps what its body keeps. -/
private theorem forIn_keeps {α β : Type} (I : β → Prop) (f : α → β → Except String (ForInStep β))
    (hf : ∀ a b s, I b → f a b = .ok s → ∃ b', s = .yield b' ∧ I b') :
    ∀ (l : List α) (b r : β), I b → forIn l b f = .ok r → I r
  | [], b, r, hb, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    exact h ▸ hb
  | a :: l, b, r, hb, h => by
    rw [List.forIn_cons] at h
    simp only [bind, Except.bind] at h
    split at h
    · cases h
    · rename_i s hs
      obtain ⟨b', rfl, hb'⟩ := hf a b s hb hs
      exact forIn_keeps I f hf l b' r hb' h

/-- The same, over an array. -/
private theorem forIn_keeps_array {α β : Type} (I : β → Prop) (f : α → β → Except String (ForInStep β))
    (hf : ∀ a b s, I b → f a b = .ok s → ∃ b', s = .yield b' ∧ I b') (as : Array α) (b r : β) (hb : I b)
    (h : forIn as b f = .ok r) : I r := by
  rw [← Array.forIn_toList] at h
  exact forIn_keeps I f hf as.toList b r hb h

/-- **The declarations begin with the wires' own, in the wires' order.** -/
theorem inputsOf_wires {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) :
    ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
      ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) ++ rest := by
  obtain ⟨D, hD⟩ : ∃ D : List Decl, D = (W.wires.zipIdx.map fun (w, i) =>
      ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) := ⟨_, rfl⟩
  rw [← hD]
  unfold inputsOf at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h
  · split at h
    · split at h
      · cases h
      · split at h
        · split at h
          · cases h
          · split at h
            · split at h
              · cases h
              · rename_i d1 h1
                have b1 : Begins D d1 := forIn_keeps_array (Begins D) _ (by
                  intro a b s hb hs
                  repeat' split at hs
                  all_goals first | (cases hs; done) | (cases hs; exact ⟨_, rfl, begins_push _ hb⟩)) _ _ _
                  ⟨[], by rw [List.append_nil, hD]⟩ h1
                split at h
                · split at h
                  · cases h
                  · rename_i d2 h2
                    have b2 : Begins D d2.1 := forIn_keeps_array (fun s => Begins D s.1) _ (by
                      intro a b s hb hs
                      repeat' split at hs
                      all_goals first | (cases hs; done) | (cases hs; exact ⟨_, rfl, begins_push _ hb⟩)) _ _ _ b1 h2
                    split at h
                    · split at h
                      · cases h
                      · rename_i d3 h3
                        have b3 : Begins D d3.1 := forIn_keeps_array (fun s => Begins D s.1) _ (by
                          intro a b s hb hs
                          repeat' split at hs
                          all_goals try (cases hs; done)
                          rename_i hin
                          cases hs
                          exact ⟨_, rfl, forIn_keeps_array (fun s => Begins D s.1) _ (by
                            intro a' b' s' hb' hs'
                            repeat' split at hs'
                            all_goals first | (cases hs'; done) | (cases hs'; exact ⟨_, rfl, begins_push _ hb'⟩))
                            _ _ _ hb hin⟩) _ _ _ b2 h3
                        split at h
                        · split at h
                          · cases h
                          · rename_i d4 h4
                            have b4 : Begins D d4 := forIn_keeps_array (Begins D) _ (by
                              intro a b s hb hs
                              repeat' split at hs
                              all_goals first | (cases hs; done) | (cases hs; exact ⟨_, rfl, begins_push _ hb⟩))
                              _ _ _ b3 h4
                            split at h
                            · cases h
                            · cases h
                              exact b4
                        · cases h
                    · cases h
                · cases h
            · cases h
        · cases h
    · cases h

/-! ## What `finish` puts in the case -/

/-- `finish` holds the summary `Spot.Wires.check` gives of the wiring it holds. -/
private theorem finish_wired {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    Spot.Wires.check c.W = .ok c.wired := by
  unfold finish at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · rename_i wired hwired
        split at h
        · cases h
        · split at h
          · cases h
          · split at h
            · cases h
            · split at h
              · cases h
              · cases h
                exact hwired
      · cases h

/-- **The summary the check holds is `Spot.Wires.check`'s of the wiring it holds**, on a record `read` accepts. -/
theorem wired_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) : Spot.Wires.check c.W = .ok c.wired := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  exact finish_wired hc

/-! ## A held row has no writer -/

/-- The first of a list that passes a test is one of its leading images, where one of them passes. -/
private theorem find?_images {α β : Type} (g : α → β) (p : β → Bool) (l : List α) (rest L : List β)
    (hL : L = l.map g ++ rest) {a : α} (ha : a ∈ l) (hp : p (g a) = true) :
    ∃ a' ∈ l, L.find? p = some (g a') := by
  rw [hL, List.find?_append]
  cases hf : (l.map g).find? p with
  | none => exact absurd hp (List.find?_eq_none.1 hf (g a) (List.mem_map.2 ⟨a, ha, rfl⟩))
  | some d =>
    obtain ⟨a', ha', rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hf)
    exact ⟨a', ha', rfl⟩

/-- **A row the check holds has no writer**, on a record `read` accepts. -/
theorem held_unwired {j tr : Json} {c : Case} (h : read j tr = .ok c) (r : Row)
    (hh : (c.heldAt r).isSome = true) : c.wired.writerOf r.tid r.unit r.port r.row = none := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  obtain ⟨rfl, -, text, -, hx, -, hwires⟩ := finish_text hc
  obtain ⟨rest, hdecls⟩ := inputsOf_wires hx
  have hS := (Spot.Wires.check_sound (wired_of_read h)).1
  cases hw : c.wired.writerOf r.tid r.unit r.port r.row with
  | none => rfl
  | some wr =>
    exfalso
    unfold Spot.Wires.Summary.writerOf at hw
    obtain ⟨w, hwmem, hwi⟩ := List.exists_of_findSome?_eq_some hw
    have hidx : (w.indexOf r.tid r.unit r.port r.row).isSome = true := by
      cases hi : w.indexOf r.tid r.unit r.port r.row with
      | none => rw [hi] at hwi; cases hwi
      | some _ => rfl
    rw [hS, hwires] at hwmem
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.1 hwmem
    obtain ⟨⟨w', i'⟩, -, hf⟩ := find?_images (a := (w, i)) _
      (fun d : Decl => (d.w.indexOf r.tid r.unit r.port r.row).isSome) _ _ _ hdecls
      (List.mem_zipIdx_iff_getElem?.2 hi) hidx
    have hfill : fillAt c.x.decls r = some (Fill.wire i') := by
      unfold fillAt
      rw [hf]
      rfl
    unfold NciCheck.Record.Case.heldAt at hh
    rw [hfill] at hh
    simp [NciCheck.RecordText.Inputs.declares] at hh

end NciCheck.Properties.Record.SecurityProofs
