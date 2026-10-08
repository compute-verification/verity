import Properties.Firewall

/-!
# The firewall's release properties, proved

Each statement of `Properties.Firewall`, proved of `Core.Firewall.check` from lemmas about its steps: `sendCheck`, the
record's `op`, `decide`, `admit` and `Decision.after`, and the lookups `openOf`, `stopOf`, `finished` and `nextOf`.
-/

namespace Properties.Firewall.SecurityProofs

open Core Core.Firewall
open Flock.Firewall.Contract (PublicInput Transcript Holds)

theorem sendCheck_ok {pub : PublicInput} {t : Transcript} (h : sendCheck pub t = .ok) : Holds pub t := by
  unfold sendCheck at h
  unfold Holds Flock.Firewall.Contract.Verdict.accepted
  cases hc : (Flock.Firewall.Contract.run pub t).cut with
  | none => rfl
  | some w => simp [hc] at h

theorem after_go {a a' : Admitted} {now : Nat} {ready : Run} (h : Decision.after a now .ok ready = .go a') :
    ready = .ok ∧ a' = a ∧ ∀ t, a.releaseMs = some t → now ≤ t := by
  unfold Decision.after at h
  cases ready with
  | err w => simp at h
  | fault m => simp at h
  | ok =>
    cases hr : a.releaseMs with
    | none => simp [hr] at h; exact ⟨rfl, h.symm, by simp⟩
    | some t =>
      simp only [hr] at h
      by_cases hn : now > t
      · simp [hn] at h
      · simp [hn] at h
        exact ⟨rfl, h.symm, fun t' ht => by cases ht; omega⟩

theorem op_ok {α : Type} {r r' : Record} {f : Events → α × Events} {x : α} (h : r.op f = .ok (x, r')) :
    ∃ r₀, Record.read r.text = .ok r₀ ∧ (f r₀.events).1 = x ∧ r' = r₀.extend (f r₀.events).2 := by
  unfold Record.op at h
  cases hr : Record.read r.text with
  | error w => simp [hr] at h; cases h
  | ok r₀ =>
    simp only [hr, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
    exact ⟨r₀, rfl, h.1, h.2.symm⟩

theorem decide_go {r r' : Record} {s part : String} {now : Nat} {prove ready : Run} {a : Admitted}
    (h : Core.Firewall.decide r s part now prove ready = .ok (.go a, r')) :
    ∃ r₀ r₁ a₀ evs, Record.read r.text = .ok r₀ ∧ Record.read (r₀.extend (slot r₀.events s part)).text = .ok r₁ ∧
      admit r₁.events s part = (.ok a₀, evs) ∧ Decision.after a₀ now prove ready = .go a := by
  unfold Core.Firewall.decide at h
  cases h1 : r.op (fun evs => ((), slot evs s part)) with
  | error w => simp [h1, bind, Except.bind] at h
  | ok x =>
    obtain ⟨⟨⟩, r₁'⟩ := x
    obtain ⟨r₀, h0, -, e1⟩ := op_ok h1
    subst e1
    cases h2 : (r₀.extend ((), slot r₀.events s part).2).op (fun x => admit x s part) with
    | error w => simp [h1, h2, bind, Except.bind] at h
    | ok y =>
      obtain ⟨res, r₂⟩ := y
      obtain ⟨r₁, h1', hres, -⟩ := op_ok h2
      cases res with
      | error w => simp [h1, h2, bind, Except.bind, pure, Except.pure] at h
      | ok a₀ =>
        simp only [h1, h2, bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        exact ⟨r₀, r₁, a₀, (admit r₁.events s part).2, h0, h1', by rw [← hres], h.1⟩

theorem admit_ok {evs evs' : Events} {s part : String} {a : Admitted} (h : admit evs s part = (.ok a, evs')) :
    ∃ o j, openOf evs s = some o ∧ stopOf evs s = none ∧ nextOf evs o = some j ∧ finished evs s = true ∧
      o.parts[j]! = part ∧ (∀ b, budgetOf evs o.stream = some b → failures evs o.stream < b) ∧
      a = { stream := o.stream, index := j, releaseMs := releaseMs o.release j } ∧ evs' = evs := by
  unfold admit at h
  cases ho : openOf evs s with
  | none => simp [ho, admit.refuse] at h
  | some o =>
    cases hs : stopOf evs s with
    | some p => simp [ho, hs, admit.refuse] at h
    | none =>
      cases hn : nextOf evs o with
      | none => simp [ho, hs, hn, admit.refuse] at h
      | some j =>
        have served : admit.served evs s part o j = (.ok a, evs') →
            finished evs s = true ∧ o.parts[j]! = part ∧
              a = { stream := o.stream, index := j, releaseMs := releaseMs o.release j } ∧ evs' = evs := by
          intro h
          unfold admit.served at h
          by_cases hf : finished evs s = true
          · by_cases hp : (o.parts[j]! != part) = true
            · simp [hf, hp, admit.refuse] at h
            · simp only [hf, hp, Bool.not_true, Bool.false_eq_true, ↓reduceIte, Prod.mk.injEq, Except.ok.injEq] at h
              simp at hp
              exact ⟨hf, hp, h.1.symm, h.2.symm⟩
          · simp [hf, admit.refuse] at h
        cases hb : budgetOf evs o.stream with
        | none =>
          simp only [ho, hs, hn, hb] at h
          obtain ⟨a1, a2, a3, a4⟩ := served h
          exact ⟨o, j, rfl, rfl, hn, a1, a2, fun b' hb' => by (rw [hb] at hb'; cases hb'), a3, a4⟩
        | some b =>
          simp only [ho, hs, hn, hb] at h
          by_cases hl : failures evs o.stream ≥ b
          · simp [hl, admit.refuse] at h
          · simp only [hl, ↓reduceIte] at h
            obtain ⟨a1, a2, a3, a4⟩ := served h
            exact ⟨o, j, rfl, rfl, hn, a1, a2, fun b' hb' => by (rw [hb] at hb'; cases hb'; omega), a3, a4⟩

theorem openOf_some {evs : Events} {s : String} {o : Open} (h : openOf evs s = some o) :
    Event.opened o ∈ evs ∧ o.session = s := by
  unfold openOf at h
  obtain ⟨ys, a, zs, he, ha, -⟩ := Array.findSome?_eq_some_iff.1 h
  have mem : a ∈ evs := by rw [he]; exact Array.mem_append.2 (.inl (Array.mem_push.2 (.inr rfl)))
  cases a with
  | opened o' =>
    by_cases hs : (o'.session == s) = true
    · simp [hs] at ha
      subst ha
      exact ⟨mem, beq_iff_eq.1 hs⟩
    · simp [hs] at ha
  | _ => simp at ha

theorem stopOf_none {evs : Events} {s : String} (h : stopOf evs s = none) : ∀ p why, Event.stop s p why ∉ evs := by
  intro p why hm
  have := Array.findSome?_eq_none_iff.1 h _ hm
  simp at this

theorem finished_true {evs : Events} {s : String} (h : finished evs s = true) : Event.finish s ∈ evs := by
  unfold finished at h
  obtain ⟨x, hx, hp⟩ := Array.any_eq_true'.1 h
  cases x with
  | finish s' => rw [beq_iff_eq.1 hp] at hx; exact hx
  | _ => simp at hp

theorem nextOf_some {evs : Events} {o : Open} {j : Nat} (h : nextOf evs o = some j) :
    ∃ hj : j < o.parts.size, Event.release o.session o.parts[j] ∉ evs ∧
      ∀ i (hi : i < j), Event.release o.session o.parts[i] ∈ evs := by
  unfold nextOf at h
  obtain ⟨hj, hp, hlt⟩ := Array.findIdx?_eq_some_iff_getElem.1 h
  simp only [Bool.not_eq_eq_eq_not, Bool.not_true, Bool.eq_false_iff, ne_eq, Array.contains_iff_mem,
    Array.mem_filterMap, Classical.not_not] at hp hlt
  refine ⟨hj, fun hm => hp ⟨_, hm, by simp⟩, fun i hi => ?_⟩
  obtain ⟨a, ha, hf⟩ := hlt i hi
  cases a with
  | release s' p' =>
    by_cases hs : (s' == o.session) = true
    · simp [hs] at hf
      rw [beq_iff_eq.1 hs, hf] at ha
      exact ha
    · simp [hs] at hf
  | _ => simp at hf

theorem releaseMs_eq (release : Option Schedule) (j : Nat) :
    releaseMs release j = release.map fun r => r.start + (r.windows.toList.take (j + 1)).sum := by
  unfold releaseMs
  cases release <;> simp [List.sum_eq_foldl]

/-- Any protocol's `answer` accepts only inputs that decode and that its check accepts. -/
theorem answer_accept (P : Protocol) {params commitment evidence : ByteArray}
    (h : (P.answer params commitment evidence).verdict = .accept) :
    ∃ p c e, (canonicalJson params >>= P.params) = .ok p ∧ P.commitment commitment = .ok c ∧
      P.evidence evidence = .ok e ∧ P.Accepts p c e := by
  unfold Protocol.answer at h
  cases hp : canonicalJson params >>= P.params with
  | error w => simp [hp] at h
  | ok p =>
    cases hc : P.commitment commitment with
    | error w => simp [hp, hc] at h
    | ok c =>
      cases he : P.evidence evidence with
      | error w => simp [hp, hc, he] at h
      | ok e =>
        cases ho : P.check p c e with
        | accept pub prof => exact ⟨p, c, e, rfl, rfl, rfl, pub, prof, ho⟩
        | reject w => simp [hp, hc, he, ho] at h
        | undecided w => simp [hp, hc, he, ho] at h

theorem CheckAccepts : Properties.Firewall.CheckAccepts := by
  intro pub r e p c h
  unfold check at h
  split at h
  · simp at h
  · simp at h
  · simp at h
  · rename_i a r' hd
    simp only [Core.Outcome.accept.injEq] at h
    obtain ⟨hp, hc⟩ := h
    obtain ⟨r₀, r₁, a₀, evs, h0, h1, ha, hgo⟩ := decide_go hd
    obtain ⟨hready, rfl, hnow⟩ := after_go hgo
    obtain ⟨o, j, ho, hs, hn, hf, hpart, hb, rfl, -⟩ := admit_ok ha
    obtain ⟨hmo, hos⟩ := openOf_some ho
    obtain ⟨hj, hnot, hprev⟩ := nextOf_some hn
    have hpart' : o.parts[j] = e.part := by rw [← hpart, getElem!_pos o.parts j hj]
    refine ⟨sendCheck_ok hready, r₀, r₁, o, j, h0, h1, hmo, hos, ho, stopOf_none hs, finished_true hf,
      ⟨hj, hpart', ?_, ?_⟩, hb, ?_, ?_, hc.symm⟩
    · rw [← hos, ← hpart']; exact hnot
    · rw [← hos]; exact hprev
    · rw [← hp, releaseMs_eq]
    · rw [← hp]; exact hnow

theorem AcceptsHolds : Properties.Firewall.AcceptsHolds := by
  intro pub r e h
  obtain ⟨p, c, h⟩ := h
  exact (CheckAccepts pub r e p c h).1

theorem AnswerHolds : Properties.Firewall.AnswerHolds := by
  intro params commitment evidence h
  obtain ⟨pub, r, e, hp, hr, he, ha⟩ := answer_accept protocol h
  exact ⟨pub, r, e, hp, hr, he, ha, AcceptsHolds pub r e ha⟩

theorem AcceptsComputed : Properties.Firewall.AcceptsComputed :=
  fun _ pub r e h hc _ _ _ _ _ hm hs => Flock.Firewall.Contract.holds_computed (AcceptsHolds pub r e h) hc hm hs

end Properties.Firewall.SecurityProofs
