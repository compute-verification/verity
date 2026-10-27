import Properties.Composition
import Proofs.Properties.Pouw.LiveSalt.Canon
import Proofs.Flock.Claims.Complete.Json.Hex

/-!
# The composition's properties, proved

`answerAccepts`, `acceptMeans` and `acceptRegistration` of `Properties.Composition`, by unfolding
`Core.Protocol.answer`, `Core.Composition.check` into `run` and `judge`, and `Core.Composition.checkRegistered` into its
registration check and then `check`. `bindCollision`, because `bindMessage` is injective in the digests: past the id
and the zero byte, its JSON reads back entry by entry, each escaped string ending at its first unescaped quote
(`canonStr_split`) and lowercase hex decoding to its bytes (`unhex_hexOf`).
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

theorem acceptRegistration : AcceptRegistration := by
  intro id R s cs n ps b d k ev pub prof h
  simp only [checkRegistered] at h
  split at h
  · cases h
  rename_i hn
  split at h
  · cases h
  rename_i hk
  split at h
  · cases h
  rename_i hp
  split at h
  · cases h
  rename_i hv
  split at h
  · cases h
  rename_i hd
  split at h
  · cases h
  rename_i answers hrun
  split at h
  · cases h
  rename_i hs
  have hv' : (registrationAnswer R n k).verdict = .accept := Decidable.of_not_not hv
  have hc : check id cs ps b ev = .accept pub prof := by
    unfold check
    rw [hrun]
    exact h
  refine ⟨Decidable.of_not_not hn, Decidable.of_not_not hk, Decidable.of_not_not hp, answerAccepts _ _ _ _ hv', ?_, ?_,
    acceptMeans _ _ _ _ _ _ _ hc⟩
  · rw [← answer_digest hv']
    exact Decidable.of_not_not hd
  · have hs' : committedBy s cs ev = some d := Decidable.of_not_not hs
    unfold committedBy at hs'
    cases hf : (cs.zip ev).find? (fun x => x.1.id == s) with
    | none => rw [hf] at hs'; cases hs'
    | some x =>
      rw [hf] at hs'
      exact ⟨x, List.mem_of_find?_eq_some hf, by simpa using List.find?_some hf, hs'⟩

/-! ## `bind`'s messages, read back

Over any quoting `q` that ends where it ends and any injective encoding `h` of the digests: `bodyOf` is the JSON `bind`
writes, and it determines the digests (`body_inj`). -/

namespace BindInj

variable (q : String → String) (h : ByteArray → String)

def entryOf (x : String × ByteArray) : String :=
  "{\"commitment\":" ++ q (h x.2) ++ ",\"id\":" ++ q x.1 ++ "}"

def restOf : List (String × ByteArray) → List Char
  | [] => [']', '}']
  | x :: xs => ',' :: ((entryOf q h x).toList ++ restOf xs)

def itemsOf : List (String × ByteArray) → List Char
  | [] => [']', '}']
  | x :: xs => (entryOf q h x).toList ++ restOf q h xs

theorem intercalate_items (ds : List (String × ByteArray)) :
    (",".intercalate (ds.map (entryOf q h))).toList ++ [']', '}'] = itemsOf q h ds := by
  match ds with
  | [] => simp [itemsOf]
  | [x] => simp [itemsOf, restOf]
  | x :: y :: ds =>
    rw [List.map_cons, List.map_cons, String.intercalate_cons_cons, String.toList_append, String.toList_append]
    have ih := intercalate_items (y :: ds)
    rw [List.map_cons] at ih
    rw [List.append_assoc, List.append_assoc, ih]
    simp [itemsOf, restOf]

theorem entry_split (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) {x y : String × ByteArray} {r r' : List Char}
    (e : (entryOf q h x).toList ++ r = (entryOf q h y).toList ++ r') : x = y ∧ r = r' := by
  simp only [entryOf, String.toList_append, List.append_assoc] at e
  have e1 := List.append_cancel_left e
  obtain ⟨h2, e2⟩ := hq _ _ _ _ e1
  have e3 := List.append_cancel_left e2
  obtain ⟨h1, e4⟩ := hq _ _ _ _ e3
  have e5 := List.append_cancel_left e4
  obtain ⟨x1, x2⟩ := x
  obtain ⟨y1, y2⟩ := y
  exact ⟨Prod.ext h1 (hh _ _ h2), e5⟩

theorem entry_head (x : String × ByteArray) (r : List Char) : ∃ t, (entryOf q h x).toList ++ r = '{' :: t := by
  simp only [entryOf, String.toList_append, List.append_assoc]
  exact ⟨_, rfl⟩

theorem rest_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) :
    ∀ ds ds' : List (String × ByteArray), restOf q h ds = restOf q h ds' → ds = ds'
  | [], [], _ => rfl
  | [], _ :: _, e => by simp [restOf] at e
  | _ :: _, [], e => by simp [restOf] at e
  | x :: xs, y :: ys, e => by
    simp only [restOf, List.cons.injEq, true_and] at e
    obtain ⟨rfl, e'⟩ := entry_split q h hq hh e
    rw [rest_inj hq hh xs ys e']

theorem items_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) :
    ∀ ds ds' : List (String × ByteArray), itemsOf q h ds = itemsOf q h ds' → ds = ds'
  | [], [], _ => rfl
  | [], y :: ys, e => by
    obtain ⟨t, ht⟩ := entry_head q h y (restOf q h ys)
    simp only [itemsOf] at e
    rw [ht] at e
    simp at e
  | x :: xs, [], e => by
    obtain ⟨t, ht⟩ := entry_head q h x (restOf q h xs)
    simp only [itemsOf] at e
    rw [ht] at e
    simp at e
  | x :: xs, y :: ys, e => by
    simp only [itemsOf] at e
    obtain ⟨rfl, e'⟩ := entry_split q h hq hh e
    rw [rest_inj q h hq hh xs ys e']

def bodyOf (ds : List (String × ByteArray)) : String :=
  "{\"components\":[" ++ ",".intercalate (ds.map fun (i, d) =>
    "{\"commitment\":" ++ q (h d) ++ ",\"id\":" ++ q i ++ "}") ++ "]}"

theorem body_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) {ds ds' : List (String × ByteArray)} (e : bodyOf q h ds = bodyOf q h ds') :
    ds = ds' := by
  have e1 := congrArg String.toList e
  have hm : ∀ ds : List (String × ByteArray), (ds.map fun (i, d) =>
      "{\"commitment\":" ++ q (h d) ++ ",\"id\":" ++ q i ++ "}") = ds.map (entryOf q h) := fun _ => rfl
  simp only [bodyOf, hm, String.toList_append, List.append_assoc] at e1
  have e2 := List.append_cancel_left e1
  have hr : ("]}" : String).toList = [']', '}'] := by simp
  rw [hr, intercalate_items, intercalate_items] at e2
  exact items_inj q h hq hh ds ds' e2

theorem msg_inj (id : String) {B B' : String}
    (e : id.toUTF8 ++ ByteArray.mk #[0] ++ B.toUTF8 = id.toUTF8 ++ ByteArray.mk #[0] ++ B'.toUTF8) : B = B' := by
  have e1 := congrArg (fun b : ByteArray => b.data.toList) e
  simp only [ByteArray.data_append, Array.toList_append, List.append_assoc] at e1
  have e2 := List.append_cancel_left (List.append_cancel_left e1)
  exact String.toByteArray_inj.1 (ByteArray.ext (Array.ext' e2))

end BindInj

theorem bindCollision : BindCollision := by
  intro id ds ds'
  by_cases e : bind id ds = bind id ds'
  · refine Or.inr ?_
    by_cases hd : ds = ds'
    · exact Or.inl hd
    · refine Or.inr ⟨fun hm => hd ?_, e⟩
      unfold bindMessage at hm
      refine BindInj.body_inj Flock.canonStr Flock.hexOf
        (fun _ _ _ _ h => Properties.Pouw.LiveSalt.Canon.canonStr_split h) (fun a b h => ?_) (BindInj.msg_inj id hm)
      have e := FlockComplete.JsonRT.unhex_hexOf a
      rw [h, FlockComplete.JsonRT.unhex_hexOf] at e
      exact (Except.ok.inj e).symm
  · exact Or.inl e

end Properties.Composition
