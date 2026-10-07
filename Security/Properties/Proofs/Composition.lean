import Properties.Composition

/-!
# Proofs of the composition's properties (`Properties.Composition`)

By unfolding the code: `run` returns the components' answers exactly when the cases fit, `judge` reads them reject first,
and `Protocol.answer` passes the check's outcome through once the inputs decode.
-/

namespace Properties.Composition

open Core Core.Composition

private theorem verdictOf_accept {α β : Type} (o : Outcome α β) :
    verdictOf o = .accept ↔ ∃ pub prof, o = .accept pub prof := by
  cases o <;> simp [verdictOf]

theorem run_fit {cs : List Protocol} {ps : List Named} {ev : List Case} (h : Fits cs ps ev) :
    run cs ps ev = .ok (answersOf cs ps ev) := by
  obtain ⟨h1, h2, h3⟩ := h
  unfold run
  simp only [h1, h2, ne_eq, not_true_eq_false, ite_false]
  split
  · rename_i heq
    have hm := List.mem_of_find?_eq_some heq
    have hp := List.find?_some heq
    simp only [decide_eq_true_eq] at hp
    exact absurd (h3 _ hm) hp
  · rfl

theorem run_misfit {cs : List Protocol} {ps : List Named} {ev : List Case} (h : ¬ Fits cs ps ev) :
    ∃ r, run cs ps ev = .error r := by
  unfold run
  by_cases h1 : ps.map (·.id) = cs.map (·.id)
  · by_cases h2 : ev.map (·.id) = cs.map (·.id)
    · simp only [h1, h2, ne_eq, not_true_eq_false, ite_false]
      split
      · exact ⟨_, rfl⟩
      · rename_i heq
        refine absurd ⟨h1, h2, fun x hx => ?_⟩ h
        have := List.find?_eq_none.mp heq x hx
        simpa using this
    · simp [h1, h2]
  · simp [h1]

theorem check_fit {id : String} {cs : List Protocol} {ps : List Named} {c : ByteArray} {ev : List Case}
    (h : Fits cs ps ev) : check id cs ps c ev = judge id c (answersOf cs ps ev) := by
  simp [check, run_fit h]

theorem check_misfit {id : String} {cs : List Protocol} {ps : List Named} {c : ByteArray} {ev : List Case}
    (h : ¬ Fits cs ps ev) : verdictOf (check id cs ps c ev) = .reject := by
  obtain ⟨r, hr⟩ := run_misfit h
  simp [check, hr, verdictOf]

theorem mem_answersOf {cs : List Protocol} {ps : List Named} {ev : List Case} {x : String × Answer} :
    x ∈ answersOf cs ps ev ↔ ∃ y ∈ cs.zip (ps.zip ev), x = (y.1.id, answerOn y.1 y.2.1 y.2.2) := by
  unfold answersOf answerOn
  simp only [List.mem_map]
  constructor
  · rintro ⟨⟨P, n, k⟩, hm, rfl⟩
    exact ⟨(P, n, k), hm, rfl⟩
  · rintro ⟨⟨P, n, k⟩, hm, rfl⟩
    exact ⟨(P, n, k), hm, rfl⟩

theorem any_answersOf {cs : List Protocol} {ps : List Named} {ev : List Case} {v : Verdict} :
    (answersOf cs ps ev).any (·.2.verdict = v) = true ↔
      ∃ y ∈ cs.zip (ps.zip ev), (answerOn y.1 y.2.1 y.2.2).verdict = v := by
  simp only [List.any_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨x, hx, hv⟩
    obtain ⟨y, hy, rfl⟩ := mem_answersOf.mp hx
    exact ⟨y, hy, hv⟩
  · rintro ⟨y, hy, hv⟩
    exact ⟨_, mem_answersOf.mpr ⟨y, hy, rfl⟩, hv⟩

theorem verdictOf_judge (id : String) (c : ByteArray) (as : Answers) :
    verdictOf (judge id c as) =
      if as.any (·.2.verdict = .reject) then .reject
      else if binding id as ≠ some c then .reject
      else if as.any (·.2.verdict = .undecided) then .undecided
      else .accept := by
  unfold judge
  by_cases h1 : as.any (·.2.verdict = .reject) = true
  · rw [ite_eq_left h1, ite_eq_left h1]; rfl
  · rw [ite_eq_right h1, ite_eq_right h1]
    by_cases h2 : binding id as ≠ some c
    · rw [ite_eq_left h2, ite_eq_left h2]; rfl
    · rw [ite_eq_right h2, ite_eq_right h2]
      by_cases h3 : as.any (·.2.verdict = .undecided) = true
      · rw [ite_eq_left h3, ite_eq_left h3]; rfl
      · rw [ite_eq_right h3, ite_eq_right h3]; rfl

private theorem accept_of {v : Verdict} (r : v ≠ .reject) (u : v ≠ .undecided) : v = .accept := by
  cases v
  · rfl
  · exact absurd rfl r
  · exact absurd rfl u

private theorem any_iff {as : Answers} {v : Verdict} :
    as.any (·.2.verdict = v) = true ↔ ∃ x ∈ as, x.2.verdict = v := by
  simp only [List.any_eq_true, decide_eq_true_eq]

theorem judge_accept_iff {id : String} {c : ByteArray} {as : Answers} :
    verdictOf (judge id c as) = .accept ↔ (∀ x ∈ as, x.2.verdict = .accept) ∧ binding id as = some c := by
  rw [verdictOf_judge]
  constructor
  · intro hv
    by_cases h1 : as.any (·.2.verdict = .reject) = true
    · rw [ite_eq_left h1] at hv; cases hv
    · rw [ite_eq_right h1] at hv
      by_cases h2 : binding id as ≠ some c
      · rw [ite_eq_left h2] at hv; cases hv
      · rw [ite_eq_right h2] at hv
        by_cases h3 : as.any (·.2.verdict = .undecided) = true
        · rw [ite_eq_left h3] at hv; cases hv
        · exact ⟨fun x hx => accept_of (fun h => h1 (any_iff.mpr ⟨x, hx, h⟩)) (fun h => h3 (any_iff.mpr ⟨x, hx, h⟩)),
            Decidable.not_not.mp h2⟩
  · rintro ⟨hall, hb⟩
    have h1 : ¬ as.any (·.2.verdict = .reject) = true := by
      rw [any_iff]; rintro ⟨x, hx, hr⟩; rw [hall x hx] at hr; cases hr
    have h3 : ¬ as.any (·.2.verdict = .undecided) = true := by
      rw [any_iff]; rintro ⟨x, hx, hu⟩; rw [hall x hx] at hu; cases hu
    have h2 : ¬ binding id as ≠ some c := fun h => h hb
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3]

theorem judge_undecided_iff {id : String} {c : ByteArray} {as : Answers} :
    verdictOf (judge id c as) = .undecided ↔
      (∀ x ∈ as, x.2.verdict ≠ .reject) ∧ binding id as = some c ∧ ∃ x ∈ as, x.2.verdict = .undecided := by
  rw [verdictOf_judge]
  constructor
  · intro hv
    by_cases h1 : as.any (·.2.verdict = .reject) = true
    · rw [ite_eq_left h1] at hv; cases hv
    · rw [ite_eq_right h1] at hv
      by_cases h2 : binding id as ≠ some c
      · rw [ite_eq_left h2] at hv; cases hv
      · rw [ite_eq_right h2] at hv
        by_cases h3 : as.any (·.2.verdict = .undecided) = true
        · exact ⟨fun x hx h => h1 (any_iff.mpr ⟨x, hx, h⟩), Decidable.not_not.mp h2, any_iff.mp h3⟩
        · rw [ite_eq_right h3] at hv; cases hv
  · rintro ⟨hr, hb, hu⟩
    have h1 : ¬ as.any (·.2.verdict = .reject) = true := by
      rw [any_iff]; rintro ⟨x, hx, h⟩; exact hr x hx h
    have h2 : ¬ binding id as ≠ some c := fun h => h hb
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_left (any_iff.mpr hu)]

theorem judge_accept_eq {id : String} {c : ByteArray} {as : Answers} {pub prof : Keyed}
    (h : judge id c as = .accept pub prof) :
    pub = as.map (fun x => (x.1, x.2.published.getD .null)) ∧ prof = as.map (fun x => (x.1, x.2.profile.getD .null)) := by
  unfold judge at h
  by_cases h1 : as.any (·.2.verdict = .reject) = true
  · rw [ite_eq_left h1] at h; cases h
  · rw [ite_eq_right h1] at h
    by_cases h2 : binding id as ≠ some c
    · rw [ite_eq_left h2] at h; cases h
    · rw [ite_eq_right h2] at h
      by_cases h3 : as.any (·.2.verdict = .undecided) = true
      · rw [ite_eq_left h3] at h; cases h
      · rw [ite_eq_right h3] at h
        injection h with hp hq
        exact ⟨hp.symm, hq.symm⟩

private theorem forall_answersOf {cs : List Protocol} {ps : List Named} {ev : List Case} {p : Answer → Prop} :
    (∀ x ∈ answersOf cs ps ev, p x.2) ↔ ∀ y ∈ cs.zip (ps.zip ev), p (answerOn y.1 y.2.1 y.2.2) := by
  constructor
  · intro h y hy; exact h _ (mem_answersOf.mpr ⟨y, hy, rfl⟩)
  · intro h x hx; obtain ⟨y, hy, rfl⟩ := mem_answersOf.mp hx; exact h y hy

private theorem exists_answersOf {cs : List Protocol} {ps : List Named} {ev : List Case} {p : Answer → Prop} :
    (∃ x ∈ answersOf cs ps ev, p x.2) ↔ ∃ y ∈ cs.zip (ps.zip ev), p (answerOn y.1 y.2.1 y.2.2) := by
  constructor
  · rintro ⟨x, hx, h⟩; obtain ⟨y, hy, rfl⟩ := mem_answersOf.mp hx; exact ⟨y, hy, h⟩
  · rintro ⟨y, hy, h⟩; exact ⟨_, mem_answersOf.mpr ⟨y, hy, rfl⟩, h⟩

theorem answer_cases (P : Protocol) (params commitment evidence : ByteArray) :
    ((P.answer params commitment evidence).verdict = .accept ∧ (P.answer params commitment evidence).published.isSome ∧
        (P.answer params commitment evidence).profile.isSome) ∨
      ((P.answer params commitment evidence).verdict ≠ .accept ∧ (P.answer params commitment evidence).published = none ∧
        (P.answer params commitment evidence).profile = none) := by
  unfold Protocol.answer
  dsimp only
  split
  · simp
  · split
    · simp
    · split
      · simp
      · split <;> simp

theorem covers : Covers := by
  intro cs ps ev ⟨h1, h2, _⟩
  have l1 : ps.length = cs.length := by simpa using congrArg List.length h1
  have l2 : ev.length = cs.length := by simpa using congrArg List.length h2
  have z1 : (cs.zip (ps.zip ev)).map (·.1) = cs := List.map_fst_zip (by simp [l1, l2])
  have z2 : (cs.zip (ps.zip ev)).map (·.2) = ps.zip ev := List.map_snd_zip (by simp [l1, l2])
  have z3 : (ps.zip ev).map (·.1) = ps := List.map_fst_zip (by simp [l1, l2])
  have z4 : (ps.zip ev).map (·.2) = ev := List.map_snd_zip (by simp [l1, l2])
  refine ⟨z1, ?_, ?_, ?_⟩
  · have e : (cs.zip (ps.zip ev)).map (·.2.1.id) = (((cs.zip (ps.zip ev)).map (·.2)).map (·.1)).map (·.id) := by
      simp only [List.map_map, Function.comp_def]
    rw [e, z2, z3, h1]
  · have e : (cs.zip (ps.zip ev)).map (·.2.2.id) = (((cs.zip (ps.zip ev)).map (·.2)).map (·.2)).map (·.id) := by
      simp only [List.map_map, Function.comp_def]
    rw [e, z2, z4, h2]
  · have e : (answersOf cs ps ev).map (·.1) = ((cs.zip (ps.zip ev)).map (·.1)).map (·.id) := by
      unfold answersOf
      simp only [List.map_map]
      apply List.map_congr_left
      rintro ⟨P, n, k⟩ _
      rfl
    rw [e, z1]

theorem rejectDominates : RejectDominates := by
  intro id cs ps c ev P n k hm hv
  show verdictOf (check id cs ps c ev) = .reject
  by_cases h : Fits cs ps ev
  · rw [check_fit h, verdictOf_judge, ite_eq_left (any_answersOf.mpr ⟨(P, n, k), hm, hv⟩)]
  · exact check_misfit h

theorem misfitRejects : MisfitRejects := fun _ _ _ _ _ h => check_misfit h

theorem acceptsIff : AcceptsIff := by
  intro id cs ps c ev
  show (∃ pub prof, check id cs ps c ev = .accept pub prof) ↔ _
  rw [← verdictOf_accept]
  by_cases h : Fits cs ps ev
  · rw [check_fit h, judge_accept_iff, forall_answersOf (p := fun a => a.verdict = .accept)]
    exact ⟨(fun ⟨ha, hb⟩ => ⟨h, ha, hb⟩), (fun ⟨_, ha, hb⟩ => ⟨ha, hb⟩)⟩
  · rw [check_misfit h]
    exact ⟨(fun hv => nomatch hv), (fun ⟨hf, _⟩ => absurd hf h)⟩

theorem undecidedIff : UndecidedIff := by
  intro id cs ps c ev
  show verdictOf (check id cs ps c ev) = .undecided ↔ _
  by_cases h : Fits cs ps ev
  · rw [check_fit h, judge_undecided_iff, forall_answersOf (p := fun a => a.verdict ≠ .reject),
      exists_answersOf (p := fun a => a.verdict = .undecided)]
    exact ⟨(fun ⟨hr, hb, hu⟩ => ⟨h, hr, hb, hu⟩), (fun ⟨_, hr, hb, hu⟩ => ⟨hr, hb, hu⟩)⟩
  · rw [check_misfit h]
    exact ⟨(fun hv => nomatch hv), (fun ⟨hf, _⟩ => absurd hf h)⟩

theorem publishesEach : PublishesEach := by
  intro id cs ps c ev pub prof hacc
  change check id cs ps c ev = .accept pub prof at hacc
  by_cases h : Fits cs ps ev
  · rw [check_fit h] at hacc
    obtain ⟨hp, hq⟩ := judge_accept_eq hacc
    have hall := (judge_accept_iff.mp ((verdictOf_accept _).mpr ⟨pub, prof, hacc⟩)).1
    refine ⟨hp, hq, fun x hx => ?_⟩
    obtain ⟨y, _, rfl⟩ := mem_answersOf.mp hx
    rcases answer_cases y.1 y.2.1.params.toUTF8 y.2.2.commitment y.2.2.evidence with ⟨_, hs, ht⟩ | ⟨hn, _, _⟩
    · exact ⟨hs, ht⟩
    · exact absurd (hall _ hx) hn
  · have hr := check_misfit (id := id) (c := c) h
    rw [hacc] at hr
    cases hr

theorem answerIsCheck : AnswerIsCheck := by
  intro P params commitment evidence
  unfold Protocol.answer
  dsimp only
  cases canonicalJson params >>= P.params with
  | error => rfl
  | ok p =>
    cases P.commitment commitment with
    | error => rfl
    | ok c =>
      cases P.evidence evidence with
      | error => rfl
      | ok e => dsimp only; cases P.check p c e <;> rfl

theorem nothingPublicUnlessAccepted : NothingPublicUnlessAccepted := by
  intro P params commitment evidence hn
  rcases answer_cases P params commitment evidence with ⟨ha, _, _⟩ | ⟨_, hp, hq⟩
  · exact absurd ha hn
  · exact ⟨hp, hq⟩

end Properties.Composition
