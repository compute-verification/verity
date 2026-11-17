module

public import Proofs.Properties.InferenceOnly.RecordRead
public import Spot.WiresSound

/-! What `RecordText.inputsOf` returns of a record's declarations: no two fill one row, and they begin with the wires'
own. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.Record.SecurityProofs

open InferenceOnlyCheck.Record (Case Front read finish unwhole rolesOf sameSlot isTheta openingHeld)
open InferenceOnlyCheck.Record.SecurityProofs (read_front finish_text)
open Spot.RecordText (Row Tmpl Decl Fill Inputs inputsOf rowWords fillAt tmplIn rowIn rect widthOf)
open Lean (Json)

private def Later (ts : List Tmpl) (W : Spot.Wires.Wiring) (d : Decl) : Prop :=
  (∀ i, d.fill ≠ .wire i) ∧ ∃ t, ts.find? (·.tid == d.w.reader) = some t ∧
    d.w.start + d.w.readerUnits ≤ t.units ∧ d.w.cut0 + d.w.cuts ≤ W.cutsOf d.w.reader d.w.port ∧
    ((t.rows.find? (·.1 == d.w.port) = none ∧ d.bits = 0) ∨
      ∃ row, t.rows.find? (·.1 == d.w.port) = some row ∧ d.bits = 16 * row.2.1 / W.cutsOf d.w.reader d.w.port)

private def Begins (Q : Decl → Prop) (D : List Decl) (a : Array Decl) : Prop :=
  ∃ rest, a.toList = D ++ rest ∧ ∀ d ∈ rest, Q d

private theorem begins_push {Q : Decl → Prop} {D : List Decl} {a : Array Decl} (d : Decl) (hd : Q d)
    (h : Begins Q D a) : Begins Q D (a.push d) := by
  obtain ⟨rest, hr, hq⟩ := h
  refine ⟨rest ++ [d], by rw [Array.toList_push, hr, List.append_assoc], ?_⟩
  intro e he
  rcases List.mem_append.1 he with h | h
  · exact hq e h
  · rw [List.mem_singleton.1 h]
    exact hd

private theorem of_not_not {b : Bool} (h : ¬(!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

private theorem tmplIn_ok {ts : List Tmpl} {tid : String} {t : Tmpl} (h : tmplIn ts tid = .ok t) :
    ts.find? (·.tid == tid) = some t := by
  unfold tmplIn at h
  split at h
  · rename_i t' ht
    cases h
    exact ht
  · cases h

private theorem rowIn_ok {t : Tmpl} {p : String} {words : Nat} (h : rowIn t p = .ok words) :
    ∃ row, t.rows.find? (·.1 == p) = some row ∧ row.2.1 = words := by
  unfold rowIn at h
  split at h
  · rename_i nm words' outer hrow
    cases h
    exact ⟨_, hrow, rfl⟩
  · cases h

private theorem rect_units (reader port : String) (start units k : Nat) :
    (rect reader port start units k).readerUnits ≤ units := by
  show k * (units * 1) / k ≤ units
  exact Nat.div_le_of_le_mul (by rw [Nat.mul_one]; exact Nat.le_refl _)

private theorem later_rect {ts : List Tmpl} {W : Spot.Wires.Wiring} {reader port : String} {t : Tmpl}
    {words start units : Nat} (ht : tmplIn ts reader = .ok t) (hw : rowIn t port = .ok words)
    (hu : ¬(!(decide (units ≥ 1) && decide (start + units ≤ t.units))) = true) (f : Fill)
    (hf : ∀ i, f ≠ .wire i) :
    Later ts W
      { w := rect reader port start units (W.cutsOf reader port), bits := 16 * words / W.cutsOf reader port,
        fill := f } := by
  obtain ⟨row, hrow, rfl⟩ := rowIn_ok hw
  have hu' := of_not_not hu
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hu'
  refine ⟨hf, t, tmplIn_ok ht, ?_, ?_, Or.inr ⟨row, hrow, rfl⟩⟩
  · exact Nat.le_trans (Nat.add_le_add_left (rect_units reader port start units _) start) hu'.2
  · show 0 + W.cutsOf reader port ≤ W.cutsOf reader port
    rw [Nat.zero_add]
    exact Nat.le_refl _

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

private theorem forIn_keeps_array {α β : Type} (I : β → Prop) (f : α → β → Except String (ForInStep β))
    (hf : ∀ a b s, I b → f a b = .ok s → ∃ b', s = .yield b' ∧ I b') (as : Array α) (b r : β) (hb : I b)
    (h : forIn as b f = .ok r) : I r := by
  rw [← Array.forIn_toList] at h
  exact forIn_keeps I f hf as.toList b r hb h

theorem inputsOf_decls {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) :
    Spot.Wires.pairwiseDisjoint (x.decls.map (·.w)) = true ∧
      ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
          ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) ++ rest ∧
        ∀ d ∈ rest, (∀ i, d.fill ≠ .wire i) ∧ ∃ t, ts.find? (·.tid == d.w.reader) = some t ∧
          d.w.start + d.w.readerUnits ≤ t.units ∧ d.w.cut0 + d.w.cuts ≤ W.cutsOf d.w.reader d.w.port ∧
          ((t.rows.find? (·.1 == d.w.port) = none ∧ d.bits = 0) ∨
            ∃ row, t.rows.find? (·.1 == d.w.port) = some row ∧
              d.bits = 16 * row.2.1 / W.cutsOf d.w.reader d.w.port) := by
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
            · cases h
            split at h
            · split at h
              · cases h
              · rename_i d1 h1
                have b1 : Begins (Later ts W) D d1 := forIn_keeps_array (Begins (Later ts W) D) _ (by
                  intro a b s hb hs
                  split at hs
                  · cases hs
                  · rename_i w hw
                    split at hs
                    · cases hs
                    · rename_i t ht
                      split at hs
                      · rename_i nm words outer hrow
                        split at hs
                        · cases hs
                        · rename_i hcut
                          split at hs
                          · cases hs
                          · rename_i hunits
                            split at hs
                            · split at hs
                              · cases hs
                              · cases hs
                                have hc := of_not_not hcut
                                have hu := of_not_not hunits
                                simp only [Bool.and_eq_true, decide_eq_true_eq] at hc hu
                                exact ⟨_, rfl, begins_push _ ⟨(fun i e => by cases e), t, tmplIn_ok ht, hu.2,
                                  hc.1.1, Or.inr ⟨_, hrow, rfl⟩⟩ hb⟩
                            · cases hs
                      · rename_i words hnone hout
                        split at hs
                        · cases hs
                        · rename_i hcut
                          split at hs
                          · cases hs
                          · rename_i hunits
                            split at hs
                            · split at hs
                              · cases hs
                              · cases hs
                                have hc := of_not_not hcut
                                have hu := of_not_not hunits
                                simp only [Bool.and_eq_true, decide_eq_true_eq] at hc hu
                                exact ⟨_, rfl, begins_push _ ⟨(fun i e => by cases e), t, tmplIn_ok ht, hu.2,
                                  hc.1.1, Or.inl ⟨hnone, rfl⟩⟩ hb⟩
                            · cases hs
                      · cases hs) _ _ _
                  ⟨[], by rw [List.append_nil, hD], fun d hd => absurd hd List.not_mem_nil⟩ h1
                split at h
                · split at h
                  · cases h
                  · rename_i d2 h2
                    have b2 : Begins (Later ts W) D d2.1 :=
                      forIn_keeps_array (fun s => Begins (Later ts W) D s.1) _ (by
                      intro a b s hb hs
                      repeat' split at hs
                      all_goals try (cases hs; done)
                      cases hs
                      exact ⟨_, rfl, begins_push _ (later_rect (by assumption) (by assumption) (by assumption) _
                        (fun i e => by cases e)) hb⟩) _ _ _ b1 h2
                    split at h
                    · split at h
                      · cases h
                      · rename_i d3 h3
                        have b3 : Begins (Later ts W) D d3.1 :=
                          forIn_keeps_array (fun s => Begins (Later ts W) D s.1) _ (by
                          intro a b s hb hs
                          repeat' split at hs
                          all_goals try (cases hs; done)
                          rename_i hin
                          cases hs
                          exact ⟨_, rfl, forIn_keeps_array (fun s => Begins (Later ts W) D s.1) _ (by
                            intro a' b' s' hb' hs'
                            repeat' split at hs'
                            all_goals try (cases hs'; done)
                            cases hs'
                            exact ⟨_, rfl, begins_push _ (later_rect (by assumption) (by assumption)
                              (by assumption) _ (fun i e => by cases e)) hb'⟩)
                            _ _ _ hb hin⟩) _ _ _ b2 h3
                        split at h
                        · split at h
                          · cases h
                          · rename_i d4 h4
                            have b4 : Begins (Later ts W) D d4 := forIn_keeps_array (Begins (Later ts W) D) _ (by
                              intro a b s hb hs
                              repeat' split at hs
                              all_goals try (cases hs; done)
                              cases hs
                              exact ⟨_, rfl, begins_push _ (later_rect (by assumption) (by assumption)
                                (by assumption) _ (fun i e => by cases e)) hb⟩)
                              _ _ _ b3 h4
                            split at h
                            · cases h
                            · rename_i hdis
                              cases h
                              exact ⟨of_not_not hdis, b4⟩
                        · cases h
                    · cases h
                · cases h
            · cases h
        · cases h
    · cases h

theorem inputsOf_wires {W : Spot.Wires.Wiring} {ts : List Tmpl} {doc : Json} {x : Inputs}
    (h : inputsOf ts W doc = .ok x) :
    ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
      ({ w, bits := 16 * rowWords ts W w, fill := .wire i } : Decl)) ++ rest := by
  obtain ⟨-, rest, hrest, -⟩ := inputsOf_decls h
  exact ⟨rest, hrest⟩

theorem finish_case {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    Spot.Wires.check c.W = .ok c.wired ∧ unwhole f.ts f.leaves c.wired.wires = none ∧
      rolesOf f.P f.ts pin c.x = .ok c.pins ∧
      c.inside = (c.wired.wires.map fun w => sameSlot c.S (c.gridsAt w.reader) (c.gridsAt w.writer) w).toArray ∧
      c.theta = c.x.params.map (fun (p, leaves) => (p, isTheta f.T leaves (widthOf f.doc p) p)) ∧
      c.held = (c.x.openings.map (openingHeld (fun tid => (c.pins.lookup tid).isSome) c.isTheta c.x)).toArray := by
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
        · rename_i hwhole
          split at h
          · cases h
          · split at h
            · cases h
            split at h
            · cases h
            · split at h
              · cases h
              · rename_i pins hpins
                cases h
                refine ⟨?_, ?_, hpins, rfl, rfl, rfl⟩
                · exact hwired
                · exact Option.not_isSome_iff_eq_none.1 hwhole
      · cases h

theorem wired_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) : Spot.Wires.check c.W = .ok c.wired := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  exact (finish_case hc).1

private theorem find?_images {α β : Type} (g : α → β) (p : β → Bool) (l : List α) (rest L : List β)
    (hL : L = l.map g ++ rest) {a : α} (ha : a ∈ l) (hp : p (g a) = true) :
    ∃ a' ∈ l, L.find? p = some (g a') := by
  rw [hL, List.find?_append]
  cases hf : (l.map g).find? p with
  | none => exact absurd hp (List.find?_eq_none.1 hf (g a) (List.mem_map.2 ⟨a, ha, rfl⟩))
  | some d =>
    obtain ⟨a', ha', rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hf)
    exact ⟨a', ha', rfl⟩

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
    unfold InferenceOnlyCheck.Record.Case.heldAt at hh
    rw [hfill] at hh
    simp [Spot.RecordText.Inputs.declares] at hh

end InferenceOnlyCheck.Properties.Record.SecurityProofs
