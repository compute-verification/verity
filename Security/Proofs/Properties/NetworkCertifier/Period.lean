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

end Properties.NetworkCertifier
