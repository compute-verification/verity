import Properties.SampledProofs.Halt

/-! # Proofs of the proof system's halt (`Properties.SampledProofs.Halt`)

Each op's step is evaluated once, as a lemma over its state monad (`Halt.step_halt`, `Halt.attached_run`,
`Halt.failures_run`, `Halt.attach_run`, `Halt.register_run`): its guards are matched or tested one at a time, and the
op's decision and state are read off the lemma. `Fleet.run` keeps the state on a refusal (`Halt.fleet_run_ok`), and
`runs` takes one op at a time (`Halt.runs_cons`), so a halted stream's later decisions are its refusals by induction
(`Halt.runs_halted`).
-/

open Lean (Json)
open SampledProofs

namespace Properties.SampledProofs

namespace Halt

theorem step_halted (p : Stream.Params) (s : Stream.State) (op : Json) (c : Json) (hc : s.halted = some c) :
    ((Stream.step p op).run s).run = .ok (.error (.refused #["halted"])) ∨
      ∃ e, ((Stream.step p op).run s).run = .error e := by
  unfold Stream.step
  simp only [StateT.run_bind, StateT.run_get, pure_bind]
  split
  · split
    · exact .inr ⟨_, rfl⟩
    · simp [hc]; exact .inl rfl
  · exact .inr ⟨_, rfl⟩

end Halt

open Halt

theorem haltIsFinal : HaltIsFinal := by
  intro p s op d s' h c hc
  unfold Stream.run at h
  rcases step_halted p s op c hc with e | ⟨e, e'⟩
  · simp only [e, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact ⟨h.1.symm, h.2.symm⟩
  · simp [e', bind, Except.bind] at h

namespace Halt

def haltedState (s : Stream.State) (cause : Json) : Stream.State :=
  { s with halted := some cause, opened := #[], entries := s.entries.map (Stream.haltEntry s.opened) }

theorem step_halt (p : Stream.Params) (s : Stream.State) (op : Json) (hop : getD op "op" = .str "halt") :
    ((Stream.step p op).run s).run =
      if s.halted.isSome then .ok (.error (.refused #["halted"]))
      else if !(getD op "cause" matches .obj _) then .error "a halt's cause is an object"
      else if !s.opened.all (fun o => s.entries.any (·.order == o)) then .error "the state holds every open entry"
      else .ok (.ok (.ok (natsJson s.opened), haltedState s (getD op "cause"))) := by
  unfold Stream.step
  simp only [StateT.run_bind, StateT.run_get, pure_bind, hop]
  have : Stream.OPS.contains "halt" = true := by simp [Stream.OPS]
  simp only [this, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  split
  · rfl
  · simp only [Stream.act]
    unfold Stream.halt
    simp only [StateT.run_bind, StateT.run_get, pure_bind]
    generalize getD op "cause" = c
    by_cases hall : (s.opened.all fun o => s.entries.any fun x => x.order == o) = true
    · cases c <;> simp only [hall, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte] <;> rfl
    · cases c <;> simp only [hall, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte] <;> rfl

end Halt

theorem haltEndsOpen : HaltEndsOpen := by
  intro p s op d s' h v hv hop
  subst hv
  unfold Stream.run at h
  rw [step_halt p s op hop] at h
  by_cases h1 : s.halted.isSome = true
  · simp [h1, bind, Except.bind, pure, Except.pure] at h
  by_cases h3 : (s.opened.all fun o => s.entries.any (·.order == o)) = true
  · generalize hc : getD op "cause" = c at h ⊢
    cases c <;> simp only [h1, h3, Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte, bind, Except.bind,
      pure, Except.pure, reduceCtorEq, Except.ok.injEq, Prod.mk.injEq, Stream.Decision.ok.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩
  · generalize hc : getD op "cause" = c at h ⊢
    cases c <;> simp [h1, h3, bind, Except.bind] at h

namespace Halt

theorem runs_cons {p : Stream.Params} {s : Stream.State} {op : Json} {ops : List Json} {ds : List Stream.Decision}
    {s' : Stream.State} (h : runs p s (op :: ops) = .ok (ds, s')) :
    ∃ d s₁ ds', Stream.run p s op = .ok (d, s₁) ∧ runs p s₁ ops = .ok (ds', s') ∧ ds = d :: ds' := by
  simp only [runs, bind, Except.bind] at h
  split at h
  · cases h
  · rename_i x hx
    split at h
    · cases h
    · rename_i y hy
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨x.1, x.2, y.1, hx, hy, rfl⟩

theorem runs_halted (p : Stream.Params) (c : Json) :
    ∀ (s : Stream.State) (ops : List Json) (ds : List Stream.Decision) (s' : Stream.State), s.halted = some c →
      runs p s ops = .ok (ds, s') → (∀ d ∈ ds, d = .refused #["halted"]) ∧ s' = s := by
  intro s ops
  induction ops generalizing s with
  | nil =>
    intro ds s' _ h
    simp only [runs, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨by simp, rfl⟩
  | cons op ops ih =>
    intro ds s' hc h
    obtain ⟨d, s₁, ds', h₁, h₂, rfl⟩ := runs_cons h
    obtain ⟨rfl, rfl⟩ := haltIsFinal p s op d s₁ h₁ c hc
    obtain ⟨hds, rfl⟩ := ih s₁ ds' s' hc h₂
    exact ⟨by simpa using hds, rfl⟩

end Halt

theorem noAcceptAfterHalt : NoAcceptAfterHalt := by
  intro p s ops
  induction ops generalizing s with
  | nil => intro ds s' h k v hk; simp [runs, pure, Except.pure] at h; obtain ⟨rfl, -⟩ := h; simp at hk
  | cons op ops ih =>
    intro ds s' h k v hk hop
    obtain ⟨d, s₁, ds', h₁, h₂, rfl⟩ := runs_cons h
    cases k with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq, Option.map_some] at hk hop
      obtain ⟨-, hh, -⟩ := haltEndsOpen p s op d s₁ h₁ v hk hop
      simpa using (runs_halted p _ s₁ ops ds' s' hh h₂).1
    | succ k =>
      simp only [List.getElem?_cons_succ] at hk hop
      simpa using ih s₁ ds' s' h₂ k v hk hop

namespace Halt

theorem attached_run (op : Json) (s : Fleet.State) :
    ((Fleet.attached op).run s).run =
      match nat? (getD op "stream") with
      | none => .ok (.error (.refused #["unattached"]))
      | some i => match s.streams[i]? with
        | none => .ok (.error (.refused #["unattached"]))
        | some a => if (Fleet.haltOf s a.operator).isSome then .ok (.error (.refused #["halted"]))
          else .ok (.ok ((i, a), s)) := by
  unfold Fleet.attached
  simp only [StateT.run_bind, StateT.run_get, pure_bind]
  rcases nat? (getD op "stream") with _ | i
  · rfl
  · dsimp only
    rcases s.streams[i]? with _ | a
    · rfl
    · dsimp only
      by_cases hh : (Fleet.haltOf s a.operator).isSome = true <;> simp only [hh, ↓reduceIte] <;> rfl

theorem haltOf_push (s : Fleet.State) (o o' : String) (c c' : Json) (h : Fleet.haltOf s o = some c) :
    Fleet.haltOf { s with halted := s.halted.push (o', c') } o = some c := by
  unfold Fleet.haltOf at *
  rw [Array.find?_push]
  cases hf : s.halted.find? (·.1 == o) with
  | none => simp [hf] at h
  | some x => simp [hf] at h ⊢; exact h

theorem fleet_run_ok {s : Fleet.State} {op : Json} {d : Stream.Decision} {s' : Fleet.State}
    (h : Fleet.run s op = .ok (d, s')) :
    ((Fleet.step op).run s).run = .ok (.ok (d, s')) ∨ (((Fleet.step op).run s).run = .ok (.error d) ∧ s' = s) := by
  unfold Fleet.run at h
  generalize ((Fleet.step op).run s).run = x at h
  rcases x with e | e | ⟨d₀, s₀⟩
  · simp [bind, Except.bind] at h
  · simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact .inr ⟨by rw [h.1], h.2.symm⟩
  · simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact .inl (by rw [h.1, h.2])

theorem register_run (op : Json) (s : Fleet.State) :
    ((Fleet.register op).run s).run = (match ((Fleet.attached op).run s).run with
      | .ok (.ok (_, s₁)) => .ok (.ok (.ok (Lean.toJson s₁.size), { s₁ with size := s₁.size + 1 }))
      | .ok (.error d) => .ok (.error d)
      | .error e => .error e) := by
  unfold Fleet.register
  simp only [StateT.run_bind, ExceptT.run_bind]
  generalize ((Fleet.attached op).run s).run = x
  rcases x with e | e | ⟨x, s₁⟩ <;> rfl

theorem failures_run (op : Json) (s : Fleet.State) :
    ((Fleet.failures op).run s).run = (match ((Fleet.attached op).run s).run with
      | .ok (.ok ((i, a), s₁)) => match nat? (getD op "failures") with
        | none => .error "failures is a count"
        | some f => if f < a.budget then .ok (.ok (.ok Json.null, s₁))
          else .ok (.ok (.ok (Fleet.haltValue (Fleet.causeOf a.operator i s₁.size) (Fleet.streamsOf s₁ a.operator)),
            { s₁ with halted := s₁.halted.push (a.operator, Fleet.causeOf a.operator i s₁.size) }))
      | .ok (.error d) => .ok (.error d)
      | .error e => .error e) := by
  unfold Fleet.failures
  simp only [StateT.run_bind, ExceptT.run_bind]
  generalize ((Fleet.attached op).run s).run = x
  rcases x with e | e | ⟨⟨i, a⟩, s₁⟩
  · rfl
  · rfl
  · simp only [bind, Except.bind, StateT.run_get]
    rcases nat? (getD op "failures") with _ | f
    · rfl
    · by_cases hf : f < a.budget
      · simp only [hf, ↓reduceIte]; rfl
      · simp only [hf, ↓reduceIte]; rfl

theorem attach_run (op : Json) (s : Fleet.State) :
    ((Fleet.attach op).run s).run = (match getD op "operator", nat? (getD op "budget") with
      | .str o, some b => if b == 0 then .error "a failure budget is a positive integer"
        else if (Fleet.haltOf s o).isSome then .ok (.error (.refused #["halted"]))
        else .ok (.ok (.ok (Lean.toJson s.streams.size), { s with streams := s.streams.push { operator := o, budget := b } }))
      | .str _, none => .error "a failure budget is a positive integer"
      | _, _ => .error "an operator is a string") := by
  unfold Fleet.attach
  simp only [StateT.run_bind, StateT.run_get, pure_bind]
  generalize getD op "operator" = j
  generalize nat? (getD op "budget") = b
  cases j <;> cases b <;> try rfl
  rename_i o b
  by_cases h0 : (b == 0) = true
  · simp only [h0, ↓reduceIte]; rfl
  · by_cases hh : (Fleet.haltOf s o).isSome = true
    · simp only [h0, hh, ↓reduceIte]; rfl
    · simp only [h0, hh, Bool.false_eq_true, ↓reduceIte]; rfl

theorem attached_ok {op : Json} {s s₁ : Fleet.State} {i : Nat} {a : Fleet.Attached}
    (h : ((Fleet.attached op).run s).run = .ok (.ok ((i, a), s₁))) :
    s₁ = s ∧ nat? (getD op "stream") = some i ∧ s.streams[i]? = some a ∧ Fleet.haltOf s a.operator = none := by
  rw [attached_run] at h
  split at h
  · cases h
  · rename_i j hn
    split at h
    · cases h
    · rename_i b hb
      split at h
      · cases h
      · rename_i hh
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h
        exact ⟨rfl, hn, hb, by simpa using hh⟩

theorem attached_err {op : Json} {s : Fleet.State} {d : Stream.Decision}
    (h : ((Fleet.attached op).run s).run = .ok (.error d)) (o : String)
    (ho : ((nat? (getD op "stream")).bind fun i => s.streams[i]?.map (·.operator)) = some o) :
    d = .refused #["halted"] := by
  rw [attached_run] at h
  split at h
  · rename_i hn; rw [hn] at ho; cases ho
  · rename_i j hn
    split at h
    · rename_i hb; rw [hn] at ho; simp [hb] at ho
    · split at h
      · simp only [Except.ok.injEq, Except.error.injEq] at h; exact h.symm
      · cases h

theorem attach_ok {op : Json} {s s' : Fleet.State} {d : Stream.Decision}
    (h : ((Fleet.attach op).run s).run = .ok (.ok (d, s'))) :
    ∃ o b, getD op "operator" = .str o ∧ Fleet.haltOf s o = none ∧
      s' = { s with streams := s.streams.push { operator := o, budget := b } } := by
  rw [attach_run] at h
  split at h
  · rename_i o b hj _
    split at h
    · cases h
    · split at h
      · cases h
      · rename_i hh
        simp only [Except.ok.injEq, Prod.mk.injEq] at h
        exact ⟨o, b, hj, by simpa using hh, h.2.symm⟩
  · cases h
  · cases h

theorem attach_err {op : Json} {s : Fleet.State} {d : Stream.Decision}
    (h : ((Fleet.attach op).run s).run = .ok (.error d)) : d = .refused #["halted"] := by
  rw [attach_run] at h
  split at h
  · split at h
    · cases h
    · split at h
      · simp only [Except.ok.injEq, Except.error.injEq] at h; exact h.symm
      · cases h
  · cases h
  · cases h

theorem haltOf_streams (s : Fleet.State) (xs : Array Fleet.Attached) (o : String) :
    Fleet.haltOf { s with streams := xs } o = Fleet.haltOf s o := rfl

theorem haltOf_size (s : Fleet.State) (n : Nat) (o : String) :
    Fleet.haltOf { s with size := n } o = Fleet.haltOf s o := rfl

end Halt

theorem fleetHaltIsFinal : FleetHaltIsFinal := by
  intro s op d s' h o c hc
  unfold opOperator
  rcases fleet_run_ok h with e | ⟨e, rfl⟩
  · unfold Fleet.step at e
    split at e
    · rename_i hk
      obtain ⟨o', b, hj, hh, rfl⟩ := attach_ok e
      refine ⟨by rw [haltOf_streams]; exact hc, ?_⟩
      simp only [hk, hj, Option.some.injEq]
      rintro rfl
      simp [hc] at hh
    · rename_i hk
      rw [register_run] at e
      split at e
      · rename_i i a s₁ ha
        obtain ⟨rfl, hn, hb, hh⟩ := attached_ok ha
        simp only [Except.ok.injEq, Prod.mk.injEq] at e
        obtain ⟨rfl, rfl⟩ := e
        refine ⟨by rw [haltOf_size]; exact hc, ?_⟩
        simp only [hk, hn, hb, Option.bind_some, Option.map_some, Option.some.injEq]
        rintro rfl
        simp [hc] at hh
      · cases e
      · cases e
    · rename_i hk
      rw [failures_run] at e
      split at e
      · rename_i i a s₁ ha
        obtain ⟨rfl, hn, hb, hh⟩ := attached_ok ha
        have hne : a.operator ≠ o := by rintro rfl; simp [hc] at hh
        split at e
        · cases e
        · split at e
          · simp only [Except.ok.injEq, Prod.mk.injEq] at e
            obtain ⟨rfl, rfl⟩ := e
            refine ⟨hc, ?_⟩
            simp only [hk, hn, hb, Option.bind_some, Option.map_some, Option.some.injEq]
            exact fun h => absurd h hne
          · simp only [Except.ok.injEq, Prod.mk.injEq] at e
            obtain ⟨rfl, rfl⟩ := e
            refine ⟨haltOf_push _ o _ c _ hc, ?_⟩
            simp only [hk, hn, hb, Option.bind_some, Option.map_some, Option.some.injEq]
            exact fun h => absurd h hne
      · cases e
      · cases e
    · cases e
  · refine ⟨hc, fun ho => ⟨?_, rfl⟩⟩
    unfold Fleet.step at e
    split at e
    · exact attach_err e
    · rename_i hk
      simp only [hk] at ho
      rw [register_run] at e
      split at e
      · cases e
      · rename_i d' ha
        simp only [Except.ok.injEq, Except.error.injEq] at e
        subst e
        exact attached_err ha o ho
      · cases e
    · rename_i hk
      simp only [hk] at ho
      rw [failures_run] at e
      split at e
      · split at e
        · cases e
        · split at e <;> cases e
      · rename_i d' ha
        simp only [Except.ok.injEq, Except.error.injEq] at e
        subst e
        exact attached_err ha o ho
      · cases e
    · cases e

namespace Halt

theorem attached_err_halted {op : Json} {s : Fleet.State} {d : Stream.Decision} {i : Nat} {a : Fleet.Attached}
    (h : ((Fleet.attached op).run s).run = .ok (.error d)) (hn : nat? (getD op "stream") = some i)
    (hb : s.streams[i]? = some a) : (Fleet.haltOf s a.operator).isSome = true := by
  rw [attached_run, hn] at h
  simp only [hb] at h
  split at h
  · assumption
  · cases h

theorem haltOf_push_new (s : Fleet.State) (o : String) (c : Json) (h : Fleet.haltOf s o = none) :
    Fleet.haltOf { s with halted := s.halted.push (o, c) } o = some c := by
  unfold Fleet.haltOf at *
  rw [Array.find?_push]
  cases hf : s.halted.find? (·.1 == o) with
  | none => simp
  | some x => simp [hf] at h

theorem step_failures {op : Json} (hk : getD op "op" = .str "failures") : Fleet.step op = Fleet.failures op := by
  unfold Fleet.step; rw [hk]; rfl

end Halt

theorem haltCoversOperator : HaltCoversOperator := by
  intro s op d s' h i f a hk hn hf hb hle
  rcases fleet_run_ok h with e | ⟨e, rfl⟩
  · rw [step_failures hk, failures_run] at e
    split at e
    · rename_i i' a' s₁ ha
      obtain ⟨rfl, hn', hb', hh⟩ := attached_ok ha
      rw [hn] at hn'
      obtain rfl := Option.some.inj hn'
      rw [hb] at hb'
      obtain rfl := Option.some.inj hb'
      rw [hf] at e
      have hlt : ¬ f < a.budget := Nat.not_lt.mpr hle
      simp only [hlt, ↓reduceIte, Except.ok.injEq, Prod.mk.injEq] at e
      obtain ⟨rfl, rfl⟩ := e
      exact ⟨_, haltOf_push_new _ _ _ hh, rfl, fun _ => ⟨rfl, rfl⟩⟩
    · cases e
    · cases e
  · rw [step_failures hk, failures_run] at e
    split at e
    · rename_i i' a' s₁ ha
      obtain ⟨rfl, hn', hb', hh⟩ := attached_ok ha
      rw [hn] at hn'
      obtain rfl := Option.some.inj hn'
      rw [hb] at hb'
      obtain rfl := Option.some.inj hb'
      rw [hf] at e
      have hlt : ¬ f < a.budget := Nat.not_lt.mpr hle
      simp only [hlt, ↓reduceIte] at e
      cases e
    · rename_i d' ha
      have hh := attached_err_halted ha hn hb
      obtain ⟨c, hc⟩ := Option.isSome_iff_exists.mp hh
      exact ⟨c, hc, rfl, fun hnone => by rw [hc] at hnone; cases hnone⟩
    · cases e

end Properties.SampledProofs
