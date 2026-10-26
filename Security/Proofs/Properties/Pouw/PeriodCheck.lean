import Properties.Pouw.PeriodCheck

/-!
# Proofs of PoUW's period check properties (`Properties.Pouw.PeriodCheck`)

By unfolding `Credit.periodCredit`: each of its steps before the loop over the windows accepts (`bind_eq_ok`), and the
loop yields every window it doesn't refuse with that window's credited calls appended (`forIn_append`), so the period's
credited calls are its windows', one list a window, in order. `acceptMeans` is that at the check, whose accept is
`periodCredit`'s; `neverUndecided` reads the check's two arms.
-/

namespace Properties.Pouw.PeriodCheck

open Lean (Json toJson)
open PouwCheck PouwCheck.Credit

theorem bind_eq_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

theorem mapError_eq_ok {ε ε' α : Type} {x : Except ε α} {f : ε → ε'} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => simp [Except.mapError] at h
  | ok b =>
    simp only [Except.mapError, Except.ok.injEq] at h
    rw [h]

/-- **A loop that appends one item a step**: when every step of `f` that doesn't refuse yields a state whose list
(`proj`) is the old one with one item `g` appended, `g` related to its element by `P`, the loop's list on success is the
start's with one item a element appended, each related to its element. -/
theorem forIn_append {α β ε σ : Type} (proj : σ → List β) (P : α → β → Prop)
    (f : α → σ → Except ε (ForInStep σ))
    (hf : ∀ x s y, f x s = .ok y → ∃ s' g, y = .yield s' ∧ proj s' = proj s ++ [g] ∧ P x g) :
    ∀ (l : List α) (s r : σ), forIn l s f = .ok r →
      ∃ gs : List β, proj r = proj s ++ gs ∧ gs.length = l.length ∧ ∀ q ∈ l.zip gs, P q.1 q.2
  | [], s, r, h => by
    simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨[], by simp, rfl, by simp⟩
  | x :: l, s, r, h => by
    rw [List.forIn_cons] at h
    obtain ⟨y, hy, h⟩ := bind_eq_ok h
    obtain ⟨s', g, rfl, hp, hP⟩ := hf x s y hy
    obtain ⟨gs, h1, h2, h3⟩ := forIn_append proj P f hf l s' r h
    refine ⟨g :: gs, by rw [h1, hp]; simp, by simp [h2], ?_⟩
    intro q hq
    simp only [List.zip_cons_cons, List.mem_cons] at hq
    rcases hq with rfl | hq
    · exact hP
    · exact h3 q hq

theorem zip_zipIdx {α β : Type} {P : α → β → Prop} :
    ∀ (l : List α) (n : Nat) (gs : List β), (∀ q ∈ (l.zipIdx n).zip gs, P q.1.1 q.2) → ∀ q ∈ l.zip gs, P q.1 q.2
  | [], _, _, _, q, hq => by simp at hq
  | _ :: _, _, [], _, q, hq => by simp at hq
  | x :: l, n, g :: gs, h, q, hq => by
    simp only [List.zip_cons_cons, List.mem_cons] at hq
    rcases hq with rfl | hq
    · exact h ((x, n), g) (by simp [List.zipIdx_cons])
    · exact zip_zipIdx l (n + 1) gs (fun q' hq' => h q' (by simp [List.zipIdx_cons, hq'])) q hq

theorem periodCreditMeans : PeriodCreditMeans := by
  intro key ledger pins main name windows pc h
  unfold periodCredit at h
  split at h
  · rename_i ws hws
    obtain ⟨u, hper, h⟩ := bind_eq_ok h
    obtain ⟨bigK, hK, h⟩ := bind_eq_ok h
    obtain ⟨t, hT, h⟩ := bind_eq_ok h
    obtain ⟨_, _, h⟩ := bind_eq_ok h
    obtain ⟨k, _, h⟩ := bind_eq_ok h
    obtain ⟨r, hr, hpc⟩ := bind_eq_ok h
    obtain ⟨gs, h1, h2, h3⟩ := forIn_append (fun s => s.1)
      (fun (x : Json × Nat) got => ∃ layout rows, windowAt key ledger pins main bigK t x.1 = .ok (got, layout, rows))
      _ (by
        intro x s y hy
        obtain ⟨w, i⟩ := x
        simp only at hy
        split at hy
        · rename_i d _
          obtain ⟨a, _, hy⟩ := bind_eq_ok hy
          split at hy
          · simp [throw, throwThe, MonadExceptOf.throw, Except.bind, bind] at hy
          · obtain ⟨v, hv, hy⟩ := bind_eq_ok hy
            obtain ⟨got, layout, rows⟩ := v
            simp only [pure, Except.pure, Except.ok.injEq] at hy
            subst hy
            exact ⟨_, got, rfl, rfl, layout, rows, mapError_eq_ok hv⟩
        · simp [throw, throwThe, MonadExceptOf.throw, Functor.map, Except.map] at hy) _ _ _ hr
    obtain ⟨out, work, eps⟩ := r
    simp only [pure, Except.pure, Except.ok.injEq] at hpc
    subst hpc
    cases u
    refine ⟨ws, hws, hper, hK, t, hT, ?_, ?_⟩
    · simp only at h1
      simp [h1, h2]
    · simp only at h1
      simp only [List.nil_append] at h1
      rw [h1]
      exact zip_zipIdx (P := fun w got => ∃ layout rows, windowAt key ledger pins main bigK t w = .ok (got, layout, rows))
        ws 0 gs h3
  · simp [throw, throwThe, MonadExceptOf.throw] at h

theorem acceptMeans : AcceptMeans := by
  intro p c e pc prof h
  change Period.check p c e = .accept pc prof at h
  unfold Period.check at h
  split at h
  · rename_i pc' hpc
    simp only [Core.Outcome.accept.injEq] at h
    obtain ⟨h1, h2⟩ := h
    subst h1
    subst h2
    exact ⟨rfl, rfl, periodCreditMeans _ _ _ _ _ _ _ hpc⟩
  · cases h

theorem neverUndecided : NeverUndecided := by
  intro p c e r h
  change Period.check p c e = .undecided r at h
  unfold Period.check at h
  split at h <;> cases h

end Properties.Pouw.PeriodCheck
