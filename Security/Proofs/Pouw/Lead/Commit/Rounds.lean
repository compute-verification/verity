import Properties.Pouw.Lead.Commit
import Proofs.Properties.Pouw.Commit

/-!
# g-private's order: a window's round coins open after their roots

`roundsOpenAfterRoots` proves `Properties.Pouw.Lead.Commit.RoundsOpenAfterRoots` by unfolding `PouwWindow.roundsReason`
to `orderReason`'s loop over the openings. The loop keeps, while it yields, that the openings read so far are each
accepted after their root (`forIn_zipIdx_ok_or_done`, `orderReason_none`), and an opening `liveReason` accepts is the coin
it asks for, opened after what it was given (`liveReason_none`).
-/

namespace Pouw.SecurityProofs.Lead.Commit

open Properties.Pouw.Lead.Commit
open SampledProofs.Challenger (LiveCommitment LiveOpening liveReason orderReason liveDomain)

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

/-- A loop over `l.zipIdx` keeps `P` of the prefix it has read while it yields, and ends in `Q` where it stops. -/
theorem forIn_zipIdx_ok_or_done {α σ ε : Type} (f : α × Nat → σ → Except ε (ForInStep σ))
    (P : List α → σ → Prop) (Q : σ → Prop)
    (hstep : ∀ pre x s t, P pre s → f (x, pre.length) s = .ok t →
      (∃ s', t = .yield s' ∧ P (pre ++ [x]) s') ∨ (∃ s', t = .done s' ∧ Q s')) :
    ∀ (l pre : List α) (s r : σ), P pre s → forIn (l.zipIdx pre.length) s f = .ok r → P (pre ++ l) r ∨ Q r := by
  intro l
  induction l with
  | nil =>
    intro pre s r hP h
    simp only [List.zipIdx_nil, List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h; left; simpa using hP
  | cons x xs ih =>
    intro pre s r hP h
    rw [List.zipIdx_cons, List.forIn_cons] at h
    cases hf : f (x, pre.length) s with
    | error e => simp [hf, bind, Except.bind] at h
    | ok t =>
      rcases hstep pre x s t hP hf with ⟨s', rfl, hP'⟩ | ⟨s', rfl, hQ⟩
      · simp only [hf, bind, Except.bind] at h
        have := ih (pre ++ [x]) s' r hP' (by simpa [List.length_append] using h)
        simpa using this
      · simp only [hf, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
        subst h; exact .inr hQ

/-- An opening `liveReason` accepts is the coin it asks for, opened after what it was given. -/
theorem liveReason_none {c : LiveCommitment} {o : LiveOpening} {label : String} {index : Int} {after : ByteArray}
    (h : liveReason c o label index after = .ok none) : o.label = label ∧ o.index = index ∧ o.after = after := by
  unfold liveReason at h
  split at h
  · split at h
    · simp [pure, Except.pure] at h
    · split at h
      · simp [pure, Except.pure] at h
      · rename_i hli
        split at h
        · simp [pure, Except.pure] at h
        · cases hd : liveDomain c.scheme c.epoch c.families with
          | error e => simp [hd, bind, Except.bind] at h
          | ok dom =>
            simp only [hd, bind, Except.bind] at h
            split at h
            · split at h
              · simp [pure, Except.pure] at h
              · rename_i ha
                simp only [bne_iff_ne, ne_eq, Bool.or_eq_true, not_or, Decidable.not_not] at hli ha
                exact ⟨hli.1, hli.2, ha⟩
            · simp [pure, Except.pure] at h
  · simp [throw, throwThe, MonadExceptOf.throw] at h

/-- What `orderReason` accepts: one opening per recorded root, opening `i` accepted after root `i`. -/
theorem orderReason_none {c : LiveCommitment} {label : String} {roots : List ByteArray}
    {openings : List LiveOpening} (h : orderReason c label roots openings = .ok none) :
    openings.length = roots.length ∧
      ∀ i (hi : i < openings.length) (hr : i < roots.length),
        liveReason c openings[i] label i roots[i] = .ok none := by
  unfold orderReason at h
  split at h
  · split at h
    · simp [pure, Except.pure] at h
    · simp only at h
      rw [bind_ok] at h
      obtain ⟨s, hs, h⟩ := h
      have hI := forIn_zipIdx_ok_or_done _
        (fun (pre : List LiveOpening) (s : Option (Option String) × List (String × Int)) =>
          s.1 = none ∧ pre.length ≤ roots.length ∧
            ∀ j (hj : j < pre.length) (hr : j < roots.length), liveReason c pre[j] label j roots[j] = .ok none)
        (fun s => ∃ m, s.1 = some (some m)) ?step openings [] _ s ⟨rfl, Nat.zero_le _, fun j hj => absurd hj (by simp)⟩
        (by simpa using hs)
      · rcases hI with ⟨h1, hlen, hall⟩ | ⟨m, hm⟩
        · simp only [List.nil_append] at hlen hall
          simp only [h1] at h
          split at h
          · simp [pure, Except.pure] at h
          · rename_i hlt
            exact ⟨by omega, hall⟩
        · simp [hm, pure, Except.pure] at h
      case step =>
        intro pre x s t hP hf
        obtain ⟨h1, hlen, hall⟩ := hP
        simp only at hf
        split at hf
        · simp only [pure, Except.pure, Except.ok.injEq] at hf
          exact .inr ⟨_, hf.symm, _, rfl⟩
        · split at hf
          · rename_i root hroot
            cases hlr : liveReason c x label (↑pre.length) root with
            | error e => simp [hlr, bind, Except.bind] at hf
            | ok r =>
              simp only [hlr, bind, Except.bind] at hf
              cases r with
              | some why =>
                simp only [pure, Except.pure, Except.ok.injEq] at hf
                exact .inr ⟨_, hf.symm, _, rfl⟩
              | none =>
                simp only [pure, Except.pure, Except.ok.injEq] at hf
                have hlt : pre.length < roots.length := (List.getElem?_eq_some_iff.1 hroot).1
                have hrt : roots[pre.length] = root := (List.getElem?_eq_some_iff.1 hroot).2
                refine .inl ⟨_, hf.symm, rfl, by simp; omega, fun j hj hr => ?_⟩
                by_cases hjl : j < pre.length
                · rw [List.getElem_append_left hjl]
                  exact hall j hjl hr
                · have hj' : j = pre.length := by simp at hj; omega
                  subst hj'
                  simp only [List.getElem_concat_length]
                  rw [hrt]
                  exact hlr
          · simp only [pure, Except.pure, Except.ok.injEq] at hf
            exact .inr ⟨_, hf.symm, _, rfl⟩
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem roundsOpenAfterRoots : RoundsOpenAfterRoots := by
  intro c roots openings h
  unfold PouwWindow.roundsReason at h
  split at h
  · simp [pure, Except.pure] at h
  · rename_i hne
    have he : PouwWindow.liveEpochReason c = none := by
      cases hc : PouwWindow.liveEpochReason c with
      | none => rfl
      | some w => exact absurd hc (hne w)
    obtain ⟨hlen, hall⟩ := orderReason_none h
    refine ⟨he, hlen, fun i hi hr => ?_⟩
    obtain ⟨hl, hx, ha⟩ := liveReason_none (hall i hi hr)
    exact ⟨hl, hx, ha, hall i hi hr⟩

end Pouw.SecurityProofs.Lead.Commit
