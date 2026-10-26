import Properties.NetworkCertifier.Period

/-!
# Proof that an accepted period accepted every link-window (`Properties.NetworkCertifier.PeriodAcceptMeans`)

`NetworkCheck.Period.check` accepts only when no finding is left (`firstFinding … = none`), so each link-window's
`windowCheck` returned `ok`; `windowCheck` returns `ok` only past every refusal, which is `PeriodWindowAccepted`.
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

theorem findingOf_none {r : Found Unit} (h : findingOf r = none) : r = .ok () := by
  cases r with
  | error x => simp [findingOf] at h
  | ok u => rfl

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

theorem windowCheck_ok {p : Params} {earlier : Array ByteArray} {w : Nat} {reg : Registration} {x : Window}
    (h : windowCheck p (clearClocks p.syncSet) earlier w reg x = .ok ()) : PeriodWindowAccepted p earlier w reg x := by
  simp only [windowCheck, refuse, bind, Except.bind, pure, Except.pure] at h
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
                refine ⟨⟨r.1, by simpa using r.2⟩, ?_, ⟨c3, by simpa using c4, by simpa using c5⟩,
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
  refine ⟨by omega, by omega, fun i reg x hc he => Period.windowCheck_ok (Period.findingOf_none (hall _ ?_))⟩
  refine List.mem_append_left _ ?_
  rw [List.mem_map]
  refine ⟨_, ?_, rfl⟩
  rw [Array.mem_toList_iff, Array.mem_map]
  exact ⟨((reg, x), i), by rw [Array.mem_zipIdx_iff_getElem?, Array.getElem?_zip_eq_some]; exact ⟨hc, he⟩, rfl⟩

end Properties.NetworkCertifier
