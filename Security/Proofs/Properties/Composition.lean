import Properties.Composition

/-!
# The composition's properties, proved

`answerAccepts` and `acceptMeans` of `Properties.Composition`, by unfolding `Core.Protocol.answer`, and
`Core.Composition.check` into `run` and `judge`.
-/

namespace Properties.Composition

open Core Core.Composition

/-- `answer` accepts exactly when the three decode and the check accepts what they decode to. -/
theorem answer_accepts_iff (P : Protocol) (a b e : ByteArray) :
    (P.answer a b e).verdict = .accept ↔ DecodedAccepted P a b e := by
  unfold Protocol.answer
  constructor
  · intro h
    cases hp : (canonicalJson a >>= P.params) with
    | error _ => rw [hp] at h; cases h
    | ok p =>
      rw [hp] at h
      cases hc : P.commitment b with
      | error _ => rw [hc] at h; cases h
      | ok c =>
        rw [hc] at h
        cases he : P.evidence e with
        | error _ => simp only [he] at h; cases h
        | ok ev =>
          simp only [he] at h
          cases hch : P.check p c ev with
          | accept pub prof => exact ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩
          | reject _ => rw [hch] at h; cases h
          | undecided _ => rw [hch] at h; cases h
  · rintro ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩
    simp only [hp, hc, he, hch]

theorem answerAccepts : AnswerAccepts := fun P a b e => (answer_accepts_iff P a b e).mp

/-- An accepted answer's digest is that of the commitment as the protocol decodes it. -/
theorem answer_digest {P : Protocol} {a b e : ByteArray} (h : (P.answer a b e).verdict = .accept) :
    (P.answer a b e).digest = digestOf P b := by
  obtain ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩ := answerAccepts P a b e h
  simp only [Protocol.answer, digestOf, hp, hc, he, hch, Except.toOption, Option.map]

theorem run_ok {cs : List Protocol} {ps : List Named} {ev : List Case} {answers : Answers}
    (h : run cs ps ev = .ok answers) :
    ps.map (·.id) = cs.map (·.id) ∧ ev.map (·.id) = cs.map (·.id) ∧
      (∀ x ∈ triples cs ps ev, x.2.2.params = x.2.1.params) ∧ answers = answersOf cs ps ev := by
  unfold run at h
  by_cases h1 : ps.map (·.id) = cs.map (·.id)
  · by_cases h2 : ev.map (·.id) = cs.map (·.id)
    · simp only [h1, h2, ne_eq, not_true_eq_false, ite_false] at h
      split at h
      · cases h
      · rename_i hf
        refine ⟨h1, h2, ?_, ?_⟩
        · intro x hx
          have hx' := List.find?_eq_none.mp hf x hx
          obtain ⟨P, n, k⟩ := x
          simpa using hx'
        · cases h; rfl
    · simp only [h1, h2, ne_eq, not_true_eq_false, not_false_eq_true, ite_false, ite_true] at h
      cases h
  · simp only [h1, ne_eq, not_false_eq_true, ite_true] at h
    cases h

theorem judge_accept {id : String} {c : ByteArray} {answers : Answers} {pub prof : Keyed}
    (h : judge id c answers = .accept pub prof) :
    (∀ x ∈ answers, x.2.verdict = .accept) ∧ binding id answers = some c ∧
      pub = answers.map (fun (i, a) => (i, a.published.getD .null)) ∧
      prof = answers.map (fun (i, a) => (i, a.profile.getD .null)) := by
  unfold judge at h
  split at h
  · cases h
  rename_i hr
  split at h
  · cases h
  rename_i hb
  split at h
  · cases h
  rename_i hu
  cases h
  refine ⟨?_, Decidable.of_not_not hb, rfl, rfl⟩
  intro x hx
  simp only [List.any_eq_true, not_exists, not_and, decide_eq_true_eq] at hr hu
  cases hv : x.2.verdict with
  | accept => rfl
  | reject => exact (hr x hx hv).elim
  | undecided => exact (hu x hx hv).elim

theorem mapM_map_congr.{u, v} {α : Type u} {β : Type v} {γ : Type} (l : List α) (f : α → β) (g : β → Option γ) (k : α → Option γ)
    (hfg : ∀ x ∈ l, g (f x) = k x) : (l.map f).mapM g = l.mapM k := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.map_cons, List.mapM_cons]
    rw [hfg a (by simp), ih (fun x hx => hfg x (by simp [hx]))]

theorem acceptMeans : AcceptMeans := by
  intro id cs ps c ev pub prof h
  unfold check at h
  split at h
  · cases h
  rename_i answers hrun
  obtain ⟨h1, h2, hpar, rfl⟩ := run_ok hrun
  obtain ⟨hacc, hbind, rfl, rfl⟩ := judge_accept h
  have hlen1 : ps.length = cs.length := by simpa using congrArg List.length h1
  have hlen2 : ev.length = cs.length := by simpa using congrArg List.length h2
  have hfst : (triples cs ps ev).map (·.1) = cs :=
    List.map_fst_zip (by simp [List.length_zip, hlen1, hlen2])
  have hone : ∀ x ∈ triples cs ps ev,
      (x.1.answer x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence).verdict = .accept := by
    intro x hx
    have := hacc _ (List.mem_map_of_mem (f := fun (x : Protocol × Named × Case) =>
      (x.1.id, x.1.answer x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence)) hx)
    exact this
  have hkeys : ∀ g : String × Answer → String × Lean.Json, (∀ x, (g x).1 = x.1) →
      ((answersOf cs ps ev).map g).map (·.1) = cs.map (·.id) := by
    intro g hg
    rw [List.map_map]
    have hm : (answersOf cs ps ev).map ((·.1) ∘ g) = (triples cs ps ev).map (·.1.id) := by
      unfold answersOf triples
      rw [List.map_map]
      exact List.map_congr_left (fun x _ => by simp only [Function.comp_apply, hg]; try rfl)
    rw [hm]
    exact (List.map_map ..).symm.trans (congrArg (List.map (·.id)) hfst)
  refine ⟨⟨h1, h2, hfst, hpar⟩, fun x hx => answerAccepts _ _ _ _ (hone x hx), ?_,
    hkeys _ (fun _ => rfl), hkeys _ (fun _ => rfl)⟩
  unfold binding at hbind
  unfold digests
  unfold answersOf at hbind
  rw [← hbind]
  congr 1
  refine (mapM_map_congr _ _ _ _ ?_).symm
  intro x hx
  have hd := answer_digest (hone x hx)
  obtain ⟨P, n, k⟩ := x
  simp only at hd ⊢
  rw [hd]

end Properties.Composition
