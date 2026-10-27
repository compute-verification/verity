import Properties.Pouw.PeriodCheck

/-!
# Proofs of PoUW's period check properties (`Properties.Pouw.PeriodCheck`)

By unfolding `Credit.periodCredit`: each of its steps before the loop over the windows accepts (`bind_eq_ok`), and the
loop yields every window it doesn't refuse with that window's credited calls appended (`forIn_append`), so the period's
credited calls are its windows', one list a window, in order. `acceptMeans` is that at the check, whose accept is
`periodCredit`'s once its form step (`Period.form`) passes; `neverUndecided` reads the check's three arms.
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
  · cases h
  · split at h
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
  split at h
  · cases h
  · split at h <;> cases h

theorem bind_ok {ε α β : Type} {x : Except ε α} {a : α} {f : α → Except ε β} (hx : x = .ok a) :
    x >>= f = f a := by
  rw [hx]; rfl

theorem mapError_ok {ε ε' α : Type} {x : Except ε α} {f : ε → ε'} {a : α} (hx : x = .ok a) :
    x.mapError f = .ok a := by
  rw [hx]; rfl

theorem exists_bind {ε α β : Type} {x : Except ε α} {a : α} {f : α → Except ε β} (hx : x = .ok a)
    (h : ∃ b, f a = .ok b) : ∃ b, x >>= f = .ok b := by
  rw [hx]; exact h

theorem exists_bind_of {ε α β : Type} {x : Except ε α} {f : α → Except ε β} (hx : ∃ a, x = .ok a)
    (h : ∀ a, ∃ b, f a = .ok b) : ∃ b, x >>= f = .ok b := by
  obtain ⟨a, hx⟩ := hx
  exact exists_bind hx (h a)

/-- **A loop that never refuses or stops early succeeds**: when every step of `f` at an element meeting `P` yields,
the loop over a list of such elements returns. -/
theorem forIn_ok {α ε σ : Type} (P : α → Prop) (f : α → σ → Except ε (ForInStep σ))
    (hf : ∀ x s, P x → ∃ s', f x s = .ok (.yield s')) :
    ∀ (l : List α) (s : σ), (∀ x ∈ l, P x) → ∃ r, forIn l s f = .ok r
  | [], s, _ => ⟨s, rfl⟩
  | x :: l, s, h => by
    obtain ⟨s', hs'⟩ := hf x s (h x (List.mem_cons_self ..))
    obtain ⟨r, hr⟩ := forIn_ok P f hf l s' fun y hy => h y (List.mem_cons_of_mem _ hy)
    exact ⟨r, by rw [List.forIn_cons, hs']; exact hr⟩

/-- **`periodCredit` credits an honest period**: one whose evidence is a list of windows, whose ledger verifies for it
(`LedgerVerifies`) and each of whose windows verifies (`WindowVerifies`). -/
theorem periodCredit_honest (key ledger pins main : Json) (name : String) (windows : Json) (ws : List Json)
    (K T : Nat) (k : SampledProofs.Challenger.ChallengerKey) (hws : windowsOf windows = some ws)
    (hl : LedgerVerifies key ledger name ws K T k)
    (hw : ∀ q ∈ ws.zipIdx, WindowVerifies k key ledger pins main name K T q.2 q.1) :
    ∃ pc, periodCredit key ledger pins main (.str name) windows = .ok pc := by
  obtain ⟨hper, hK, hT, ⟨W, hW⟩, hk⟩ := hl
  unfold periodCredit
  split
  · rename_i ws' hws'
    have hw' : ws' = ws := Option.some.inj (hws'.symm.trans hws)
    subst hw'
    refine exists_bind hper ?_
    refine exists_bind hK ?_
    refine exists_bind hT ?_
    refine exists_bind (x := discard (committedWeights ledger name)) (by rw [hW]; rfl) ?_
    refine exists_bind (mapError_ok hk) ?_
    refine exists_bind_of (forIn_ok (fun q : Json × Nat => WindowVerifies k key ledger pins main name K T q.2 q.1)
      _ ?_ _ _ hw) fun _ => ⟨_, rfl⟩
    intro ⟨w, i⟩ s hv
    obtain ⟨d, hd, a, ha, hp, got, layout, rows, hwl⟩ := hv
    refine ⟨(s.1 ++ [got], s.2.1 ++ [creditedWork layout got rows], s.2.2 ++ [established layout got rows K]), ?_⟩
    dsimp only
    split
    · rename_i d' hd'
      have hdd : d' = d := Option.some.inj (hd'.symm.trans hd)
      subst hdd
      refine (bind_ok (mapError_ok ha)).trans ?_
      split
      · rename_i why hwhy
        cases hp.symm.trans hwhy
      · refine (bind_ok (mapError_ok hwl)).trans ?_
        rfl
    · rename_i hd'
      exact (hd' d hd).elim
  · rename_i hws'
    exact (hws' ws hws).elim

/-- **The period check accepts an honest period** (`AcceptMeans`'s honest lemma, a completeness theorem): one whose
windows are in the form its guarantee covers (`Period.form`) is accepted with its `n` windows credited, `n` the
evidence's. -/
theorem honestAccepts (p : Period.Params) (c : Period.Commitment) (e : Period.Evidence) (ws : List Json) (K T : Nat)
    (k : SampledProofs.Challenger.ChallengerKey) (hf : Period.form p.name e.windows = none)
    (hws : windowsOf e.windows = some ws)
    (hl : LedgerVerifies p.key c.ledger p.name ws K T k)
    (hw : ∀ q ∈ ws.zipIdx, WindowVerifies k p.key c.ledger e.pins p.main p.name K T q.2 q.1) :
    ∃ pc, Period.protocol.check p c e = .accept pc { period := p.name, windows := pc.windows.length } ∧
      pc.windows.length = ws.length := by
  obtain ⟨pc, hpc⟩ := periodCredit_honest _ _ _ _ _ _ ws K T k hws hl hw
  obtain ⟨ws', hws', _, _, _, _, hlen, _⟩ := periodCreditMeans _ _ _ _ _ _ _ hpc
  have : ws' = ws := Option.some.inj (hws'.symm.trans hws)
  subst this
  refine ⟨pc, ?_, hlen.symm⟩
  change Period.check p c e = _
  unfold Period.check
  rw [hf, hpc]
  rfl

theorem refuses : Refuses := by
  refine ⟨{ key := .null, main := .null, name := "" }, { ledger := .null, bytes := .empty },
    { pins := .null, windows := .null }, fun pc prof h => ?_⟩
  obtain ⟨_, _, ws, hws, _⟩ := acceptMeans _ _ _ _ _ h
  simp [windowsOf] at hws

end Properties.Pouw.PeriodCheck
