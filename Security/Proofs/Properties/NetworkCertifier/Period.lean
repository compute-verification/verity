import Properties.NetworkCertifier.Period

/-!
# Proof that an accepted period accepted every link-window (`Properties.NetworkCertifier.PeriodAcceptMeans`)

`NetworkCheck.Period.check` accepts only when no finding is left (`firstFinding … = none`), so each link-window's
check of the parameters' form returned `ok`: `windowCheck` in the clock form, `rowsWindowCheck` in the rows form. Each
returns `ok` only past every refusal, which is `PeriodWindowAccepted` or `PeriodRowsWindowAccepted`.
-/

open NetworkCheck.Period

namespace Properties.NetworkCertifier

namespace Period

theorem firstFinding_none {fs : List (Option (Core.Verdict × String))} (h : firstFinding fs = none) :
    ∀ f ∈ fs, f = none := by
  unfold firstFinding at h
  split at h
  · cases h
  · exact List.findSome?_eq_none_iff.mp h

theorem findingOf_none {α : Type} {r : Found α} (h : findingOf r = none) : ∃ a, r = .ok a := by
  cases r with
  | error x => simp [findingOf] at h
  | ok u => exact ⟨u, rfl⟩

theorem answered_ok {what : String} {reg : Registration} {a : Core.Answer} {v : Core.Integrity × Lean.Json}
    (h : answered what reg a = .ok v) :
    a.protocol = oneStage ∧ (a.digest == some reg.digest) = true ∧ a.verdict = .accept ∧
      ∃ prof, a.published = some v.2 ∧ a.profile = some prof ∧ Core.Integrity.ofJson prof = .ok v.1 := by
  simp only [answered, refuse, pure, Except.pure] at h
  split at h
  · cases h
  rename_i d1
  split at h
  · cases h
  rename_i d2
  split at h
  · rename_i pub prof hv hp hpr
    split at h
    · rename_i i hi
      cases h
      refine ⟨?_, ?_, hv, prof, hp, hpr, hi⟩
      · simpa [bne_iff_ne] using d1
      · simpa [bne] using d2
    · cases h
  · cases h
  · cases h

theorem windowCheck_ok {p : Params} {ss : NetworkCertifier.SyncSet} {earlier : Array ByteArray} {w : Nat}
    {reg : Registration} {x : Window} (h : windowCheck p (clearClocks ss) earlier w reg x = .ok ()) :
    PeriodWindowAccepted p ss earlier w reg x := by
  simp only [windowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i clock hx
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        split at h
        · cases h
        · rename_i c4
          split at h
          · cases h
          · rename_i c5
            split at h
            · cases h
            · rename_i v ha
              split at h
              · rename_i c6
                obtain ⟨a1, a2, a3, prof, a4, a5, a6⟩ := answered_ok ha
                have r := (Bool.and_eq_true _ _).mp c1
                have q := (Bool.and_eq_true _ _).mp c6
                refine ⟨⟨r.1, by simpa using r.2⟩, ?_, ⟨clock, hx, c3, by simpa using c4, by simpa using c5⟩,
                  a1, a2, a3, v.2, prof, v.1, a4, a5, a6, q.1, q.2⟩
                obtain ⟨hs, hall⟩ := (Bool.and_eq_true _ _).mp c2
                unfold PeriodWindowCertified
                generalize x.certificates = cs at hs hall
                rcases cs with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨d, l⟩⟩⟩⟩
                · simp at hs
                · simp at hs
                · refine ⟨a, b, rfl, ?_⟩
                  simp [Array.zip, Array.all] at hall
                  revert hall
                  cases certifies p .egress w a <;> cases certifies p .ingress w b <;> simp [Id.run, pure]
                · simp at hs
              · cases h
      · cases h
    · cases h
  · cases h

theorem rowsCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {d : NetworkCheck.Direction} {r : Rows}
    (h : rowsCheck p s sd w d r = .ok ()) : PeriodRowsChecked p s sd w d r := by
  simp only [rowsCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        split at h
        · rename_i dom hdom
          split at h
          · rename_i c4
            split at h
            · rename_i head hhead
              split at h
              · rename_i c5
                split at h
                · rename_i c6
                  exact ⟨c1, c2, c3, dom, head, hdom, c4, hhead, c5, c6⟩
                · cases h
              · cases h
            · cases h
          · cases h
        · cases h
      · cases h
    · cases h
  · cases h

theorem factsOf_ok {what : String} {prof : Core.Integrity} {pub : Lean.Json} {f : Facts}
    (h : factsOf what prof pub = .ok f) :
    ∃ l b, prof.levels = #[l] ∧ f.units = l.size ∧
      ((∃ k, l.draw = some (.subset k)) ∨ ∃ ss, l.draw = some (.stratified ss)) ∧
      Flock.J.nat pub "bound" "public" = .ok b := by
  by_cases hl : ∃ l, prof.levels = #[l]
  · obtain ⟨l, hl⟩ := hl
    have e : factsOf what prof pub = (do
        let drawn ← match l.draw with
          | some (.subset k) => pure k
          | some (.stratified ss) => pure (ss.foldl (fun n st => n + st.k) 0)
          | _ => refuse s!"{what}: profile: not a subset or stratified draw"
        match Flock.J.nat pub "bound" "public" with
        | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b }
        | .error e => refuse s!"{what}: {e}" : Found Facts) := by
      unfold factsOf
      rw [hl]
      rfl
    rw [e] at h
    simp only [refuse, bind, Except.bind, pure, Except.pure] at h
    split at h
    · rename_i k hk
      split at h
      · rename_i b hb
        cases h
        exact ⟨l, b, hl, rfl, Or.inl ⟨k, hk⟩, hb⟩
      · cases h
    · rename_i ss hss
      split at h
      · rename_i b hb
        cases h
        exact ⟨l, b, hl, rfl, Or.inr ⟨ss, hss⟩, hb⟩
      · cases h
    · cases h
  · exfalso
    have : ∃ e, factsOf what prof pub = .error e := by
      unfold factsOf
      rcases h' : prof.levels with ⟨_ | ⟨l, _ | ⟨l', ls⟩⟩⟩
      · exact ⟨_, rfl⟩
      · exact absurd ⟨l, h'⟩ hl
      · exact ⟨_, rfl⟩
    obtain ⟨e, he⟩ := this
    rw [he] at h
    cases h

theorem rowsWindowCheck_ok {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {reg : Registration} {x : Window}
    {f : Facts} (h : rowsWindowCheck p s sd w reg x = .ok f) : PeriodRowsWindowAccepted p s sd w reg x := by
  simp only [rowsWindowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
  split at h
  case h_2 => cases h
  rename_i rs hx
  split at h
  · rename_i c1
    split at h
    · rename_i c2
      split at h
      · rename_i c3
        rcases rs with ⟨_ | ⟨ri, _ | ⟨re, _ | ⟨d, l⟩⟩⟩⟩
        · simp at c3
        · simp at c3
        · split at h
          · cases h
          · rename_i v hloop
            rw [← Array.forIn_toList] at hloop
            simp only [Array.toList_zip, List.zip_cons_cons, List.zip_nil_left, List.forIn_cons,
              List.forIn_nil] at hloop
            split at hloop
            · cases hloop
            rename_i u hi
            split at hloop
            · cases hloop
            rename_i u' he
            cases u
            cases u'
            split at h
            · rename_i c4
              split at h
              · cases h
              · rename_i v ha
                obtain ⟨a1, a2, a3, prof, a4, a5, a6⟩ := answered_ok ha
                split at h
                · cases h
                rename_i f' hf
                obtain ⟨l, b, hl, hu, hd, hb⟩ := factsOf_ok hf
                have hun : l.size = 2 * s.window := by
                  rw [← hu]
                  cases hc : (f'.units == 2 * s.window) with
                  | true => exact beq_iff_eq.mp hc
                  | false => simp [hc] at h
                have hh : holds reg ri = true ∧ holds reg re = true :=
                  ⟨Array.all_eq_true.mp c4 0 (by simp), Array.all_eq_true.mp c4 1 (by simp)⟩
                refine ⟨c1, ?_, ⟨ri, re, hx, rowsCheck_ok hi, rowsCheck_ok he, hh.1, hh.2⟩,
                  a1, a2, a3, v.2, prof, v.1, l, b, a4, a5, a6, hl, hun, hd, hb⟩
                obtain ⟨hs, hall⟩ := (Bool.and_eq_true _ _).mp c2
                unfold PeriodRowsCertified
                generalize x.certificates = cs at hs hall
                rcases cs with ⟨_ | ⟨q1, _ | ⟨q2, _ | ⟨q3, _ | ⟨q4, qs⟩⟩⟩⟩⟩
                · simp at hs
                · simp at hs
                · simp at hs
                · refine ⟨q1, q2, q3, rfl, ?_⟩
                  simp [records, Array.zip, Array.all] at hall
                  revert hall
                  cases certifiesOn p p.link .ingress w q1 <;> cases certifiesOn p p.link .egress w q2 <;>
                    cases certifiesOn p (p.link ++ "-fails") .egress w q3 <;> simp [Id.run, pure]
                · simp at hs
            · cases h
        · simp at c3
      · cases h
    · cases h
  · cases h

theorem map_ok {α β : Type} {g : α → β} {r : Found α} {b : β} (h : g <$> r = .ok b) : ∃ a, r = .ok a := by
  cases r with
  | error x => cases h
  | ok a => exact ⟨a, rfl⟩

end Period

theorem periodAcceptMeans : PeriodAcceptMeans := by
  intro p c e pub prof h
  change check p c e = _ at h
  unfold check at h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i h1 h2 h3 h4
  dsimp only at h
  split at h
  · cases h
  · cases h
  rename_i hf
  have hall := Period.firstFinding_none hf
  refine ⟨by omega, by omega, fun i reg x hc he => ?_⟩
  have hmem : ((reg, x), i) ∈ (c.windows.zip e.windows).zipIdx := by
    rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some]
    exact ⟨hc, he⟩
  rcases hform : p.form with ss | ⟨s, sd, scan⟩
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
      (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩)))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨u, hu⟩ := Period.map_ok ha
    exact Period.windowCheck_ok hu
  · simp only [hform] at hall
    have h0 := hall _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_map.2 ⟨_, Array.mem_toList_iff.2
      (Array.mem_map.2 ⟨_, hmem, rfl⟩), rfl⟩)))
    obtain ⟨_, ha⟩ := Period.findingOf_none h0
    obtain ⟨f, hu⟩ := Period.map_ok ha
    exact Period.rowsWindowCheck_ok hu

namespace Period

theorem answered_of {what : String} {reg : Registration} {a : Core.Answer} {pub prof : Lean.Json}
    {i : Core.Integrity} (h1 : a.protocol = oneStage) (h2 : (a.digest == some reg.digest) = true)
    (h3 : a.verdict = .accept) (h4 : a.published = some pub) (h5 : a.profile = some prof)
    (h6 : Core.Integrity.ofJson prof = .ok i) : answered what reg a = .ok (i, pub) := by
  simp only [answered, refuse, h1, bne, h2, Bool.not_true, Bool.false_eq_true, ↓reduceIte, h3, h4, h5, h6]
  rfl

theorem factsOf_of {prof : Core.Integrity} {pub : Lean.Json} {l : Core.Level} {b : Nat} (hl : prof.levels = #[l])
    (hd : (∃ k, l.draw = some (.subset k)) ∨ ∃ ss, l.draw = some (.stratified ss))
    (hb : Flock.J.nat pub "bound" "public" = .ok b) :
    ∃ f : Facts, f.units = l.size ∧ f.delta = prof.delta ∧ ∀ what, factsOf what prof pub = .ok f := by
  have e : ∀ what, factsOf what prof pub = (do
      let drawn ← match l.draw with
        | some (.subset k) => pure k
        | some (.stratified ss) => pure (ss.foldl (fun n st => n + st.k) 0)
        | _ => refuse s!"{what}: profile: not a subset or stratified draw"
      match Flock.J.nat pub "bound" "public" with
      | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b }
      | .error e => refuse s!"{what}: {e}" : Found Facts) := fun what => by
    unfold factsOf
    rw [hl]
    rfl
  rcases hd with ⟨k, hk⟩ | ⟨ss, hss⟩
  · exact ⟨{ units := l.size, drawn := k, delta := prof.delta, wrong := b }, rfl, rfl, fun what => by
      rw [e]
      simp only [hk, hb]
      rfl⟩
  · exact ⟨{ units := l.size, drawn := ss.foldl (fun n st => n + st.k) 0, delta := prof.delta, wrong := b }, rfl,
      rfl, fun what => by
        rw [e]
        simp only [hss, hb]
        rfl⟩

theorem auditCheck_of {n : Nat} {a : Audit} {reg : Registration} {x : Core.Answer} (h : PeriodAuditProved a reg x) :
    ∃ f, auditCheck n a reg x = .ok f := by
  obtain ⟨h0, h1, h2, h3, pub, prof, i, l, b, h4, h5, h6, hl, hu, hd, hb⟩ := h
  obtain ⟨f, hfu, -, hf⟩ := factsOf_of hl hd hb
  have hu' : (f.units == a.units) = true := by rw [hfu, hu]; exact beq_self_eq_true _
  refine ⟨f, ?_⟩
  simp only [auditCheck, h0, answered_of h1 h2 h3 h4 h5 h6, hf, hu', bind, Except.bind, pure, Except.pure,
    ↓reduceIte]

theorem windowCheck_of {p : Params} {ss : NetworkCertifier.SyncSet} {earlier : Array ByteArray} {w : Nat}
    {reg : Registration} {x : Window} (h : PeriodWindowAccepted p ss earlier w reg x) :
    windowCheck p (clearClocks ss) earlier w reg x = .ok () := by
  obtain ⟨⟨r1, r2⟩, ⟨eg, ig, hcs, heg, hig⟩, ⟨clock, hx, hcl, hclear, hearlier⟩, a1, a2, a3, pub, prof, i, a4, a5,
    a6, q1, q2⟩ := h
  have c1 : (reg.record.getObjValD "program" == Lean.Json.str p.program &&
      canonText (reg.record.getObjValD "window") == canonText (served w)) = true :=
    (Bool.and_eq_true _ _).mpr ⟨r1, by simp [r2]⟩
  have c2 : (x.certificates.size == 2 && (x.certificates.zip #[NetworkCheck.Direction.egress, .ingress]).all
      fun a => certifies p a.snd w a.fst) = true := by
    rw [hcs]
    simp [heg, hig]
  simp only [windowCheck, refuse, bind, Except.bind, pure, Except.pure]
  split
  case h_2 => rename_i hn; exact (hn clock hx).elim
  rename_i clock' hx'
  rw [hx] at hx'
  cases hx'
  simp only [c1, c2, hcl, hclear, hearlier, answered_of a1 a2 a3 a4 a5 a6, q1, q2, ↓reduceIte, Bool.false_eq_true,
    Bool.and_self]

theorem rowsCheck_of {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {d : NetworkCheck.Direction} {r : Rows}
    (h : PeriodRowsChecked p s sd w d r) : rowsCheck p s sd w d r = .ok () := by
  obtain ⟨c1, c2, c3, dom, head, hdom, c4, hhead, c5, c6⟩ := h
  simp only [rowsCheck, pure, Except.pure, c1, c2, c3, hdom, c4, hhead, c5, c6, ↓reduceIte]

/-- An accepted rows-form link-window's facts are at its answer's `δ`. -/
theorem rowsWindowCheck_of {p : Params} {s : Schedule} {sd : ByteArray} {w : Nat} {reg : Registration} {x : Window}
    (h : PeriodRowsWindowAccepted p s sd w reg x) :
    ∃ f : Facts, (∀ prof it, x.answer.profile = some prof → Core.Integrity.ofJson prof = .ok it → f.delta = it.delta) ∧
      rowsWindowCheck p s sd w reg x = .ok f := by
  obtain ⟨c1, ⟨rq, rp, fl, hcs, hrq, hrp, hfl⟩, ⟨ri, re, hx, hi, he, hhi, hhe⟩, a1, a2, a3, pub, prof, i, l, b, a4,
    a5, a6, hl, hun, hd, hb⟩ := h
  obtain ⟨f, hfu, hfd, hf⟩ := factsOf_of hl hd hb
  have c2 : (x.certificates.size == 3 &&
      (x.certificates.zip (records p.link)).all fun a => certifiesOn p a.2.fst a.2.snd w a.fst) = true := by
    rw [hcs]
    simp [records, hrq, hrp, hfl]
  have hu' : (f.units == 2 * s.window) = true := by rw [hfu, hun]; exact beq_self_eq_true _
  refine ⟨f, fun prof' it h5' h6' => ?_, ?_⟩
  · rw [a5] at h5'
    cases h5'
    rw [a6] at h6'
    cases h6'
    exact hfd
  simp only [rowsWindowCheck, refuse, bind, Except.bind, pure, Except.pure]
  split
  case h_2 => rename_i hn; exact (hn _ hx).elim
  rename_i rs hx'
  rw [hx] at hx'
  cases hx'
  have c1' : (reg.record.getObjValD "program" == Lean.Json.str p.program) = true := c1
  have c3 : (#[ri, re].size == 2) = true := rfl
  have c4 : #[ri, re].all (holds reg) = true := by simp [hhi, hhe]
  simp only [c1', c2, c3, c4, ↓reduceIte, List.zip_toArray, List.forIn_toArray, List.zip_cons_cons, List.zip_nil_left,
    List.forIn_cons, List.forIn_nil, rowsCheck_of hi, rowsCheck_of he, bind, Except.bind, pure, Except.pure,
    answered_of a1 a2 a3 a4 a5 a6, hf, hu']

theorem firstFinding_of {fs : List (Option (Core.Verdict × String))} (h : ∀ f ∈ fs, f = none) :
    firstFinding fs = none := by
  unfold firstFinding
  split
  · rename_i r hr
    obtain ⟨f, hf, hfr⟩ := List.exists_of_findSome?_eq_some hr
    rw [h f hf] at hfr
    cases hfr
  · exact List.findSome?_eq_none_iff.mpr h

theorem map_join_some {α β : Type} (g : α → Option β) (r : Found α) {f : β} (h : (g <$> r).toOption.join = some f) :
    ∃ a, r = .ok a ∧ g a = some f := by
  cases r with
  | error x => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem ratio_beq_self (r : Core.Ratio) : (r == r) = true := by
  cases r
  show (_ && _) = true
  simp

/-- The check's one-δ finding is none when every scan's facts are at one `δ`. -/
theorem oneDelta_none {β : Type} {scans : Array Facts} {δ : Core.Ratio} (h : ∀ f ∈ scans, f.delta = δ)
    (g : Facts → Facts → β) : (scans[0]?.bind fun f0 => (scans.find? (·.delta != f0.delta)).map (g f0)) = none := by
  cases h0 : scans[0]? with
  | none => rfl
  | some f0 =>
    have hf0 := h f0 (Array.mem_of_getElem? h0)
    have hn : scans.find? (·.delta != f0.delta) = none :=
      Array.find?_eq_none.2 fun f hf => by simp [h f hf, hf0, bne, ratio_beq_self]
    simp [hn]

end Period

/-- **The converse of ONE-13**: a period whose every link-window is accepted in the parameters' form, whose scans are
at one `δ` in the rows form, and whose every compute audit was proved, at the sizes the parameters name, is
accepted. -/
theorem periodAccepts_of {p : Params} {c : Commitment} {e : Evidence} (hw : PeriodWindowsAccepted p c e)
    (hδ : PeriodScansOneDelta p e) (ha : PeriodAuditsProved p c e) :
    ∃ pub prof, protocol.check p c e = .accept pub prof := by
  show ∃ pub prof, check p c e = .accept pub prof
  obtain ⟨hcw, hew, hall⟩ := hw
  obtain ⟨hca, hea, haud⟩ := ha
  have n1 : ¬ c.windows.size ≠ p.windows := fun h => h hcw
  have n2 : ¬ c.audits.size ≠ p.audits.size := fun h => h hca
  have n3 : ¬ e.windows.size ≠ c.windows.size := fun h => h (hew.trans hcw.symm)
  have n4 : ¬ e.audits.size ≠ c.audits.size := fun h => h (hea.trans hca.symm)
  unfold check
  rw [ite_eq_right n1, ite_eq_right n2, ite_eq_right n3, ite_eq_right n4]
  dsimp only
  rw [Period.firstFinding_of]
  · exact ⟨_, _, rfl⟩
  intro f hf
  rcases List.mem_append.1 hf with hf | hf
  · rcases List.mem_append.1 hf with hf | hf
    · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hf
      rcases hform : p.form with ss | ⟨s, sd, scan⟩
      · simp only [hform] at hy
        obtain ⟨⟨⟨reg, x⟩, i⟩, hmem, rfl⟩ := Array.mem_map.1 (Array.mem_toList_iff.1 hy)
        rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some] at hmem
        have hacc := hall i reg x hmem.1 hmem.2
        rw [hform] at hacc
        dsimp only
        rw [Period.windowCheck_of hacc]
        rfl
      · simp only [hform] at hy
        obtain ⟨⟨⟨reg, x⟩, i⟩, hmem, rfl⟩ := Array.mem_map.1 (Array.mem_toList_iff.1 hy)
        rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some] at hmem
        have hacc := hall i reg x hmem.1 hmem.2
        rw [hform] at hacc
        obtain ⟨g, -, hg⟩ := Period.rowsWindowCheck_of hacc
        dsimp only
        rw [hg]
        rfl
    · rw [List.mem_singleton.1 hf]
      rcases hform : p.form with ss | ⟨s, sd, scan⟩
      · dsimp only
        refine Period.oneDelta_none (δ := ⟨0, 0⟩) (fun f hf => ?_) _
        obtain ⟨y, hy, hfy⟩ := Array.mem_filterMap.1 hf
        obtain ⟨⟨⟨reg, x⟩, i⟩, -, rfl⟩ := Array.mem_map.1 hy
        obtain ⟨u, -, hu⟩ := Period.map_join_some _ _ hfy
        cases u
        cases hu
      · dsimp only
        rw [PeriodScansOneDelta, hform] at hδ
        obtain ⟨δ, hδ⟩ := hδ
        refine Period.oneDelta_none (δ := δ) (fun f hf => ?_) _
        obtain ⟨y, hy, hfy⟩ := Array.mem_filterMap.1 hf
        obtain ⟨⟨⟨reg, x⟩, i⟩, hmem, rfl⟩ := Array.mem_map.1 hy
        rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some] at hmem
        have hacc := hall i reg x hmem.1 hmem.2
        rw [hform] at hacc
        obtain ⟨g, hgd, hg⟩ := Period.rowsWindowCheck_of hacc
        obtain ⟨f', hr, hu⟩ := Period.map_join_some _ _ hfy
        rw [hg] at hr
        cases hr
        cases hu
        obtain ⟨_, _, _, _, _, _, _, prof, it, _, _, _, a5, a6, _⟩ := hacc
        rw [hgd prof it a5 a6]
        exact hδ i x prof it hmem.2 a5 a6
  · obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hf
    obtain ⟨⟨⟨⟨a, reg⟩, x⟩, i⟩, hmem, rfl⟩ := Array.mem_map.1 (Array.mem_toList_iff.1 hy)
    rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some, Array.getElem?_zip_eq_some] at hmem
    obtain ⟨g, hg⟩ := Period.auditCheck_of (n := i) (haud i a reg x hmem.1.1 hmem.1.2 hmem.2)
    dsimp only
    rw [hg]
    rfl

end Properties.NetworkCertifier
