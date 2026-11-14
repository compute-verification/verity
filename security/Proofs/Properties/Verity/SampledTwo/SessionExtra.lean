module

public import Proofs.Properties.Verity.Sampled.MemberForm.Order

/-!
# A session's coin tree opens no round its record doesn't give

`Spot.Challenger.sessionTreeReason` ends by refusing a record whose tree shows opened a round the record does not give
that round's stream (`Spot.Challenger.EXTRA_OPENING`, `Spot.Challenger.recordRounds`).
`Verity.Sampled.Trees.round_held` reads the lines before that one: each round the record gives is a round the log shows
opened. This module reads that last line, the converse:

* `sessionTree_recorded`: where `sessionTreeReason` accepts a record, each round its tree shows opened is below the
  rounds the record gives that round's stream. The streams' loop returns at its first reason, so where the check
  returns none the loop returned none and the line after it ran (`forIn_ret_none`).
* `memberTree_recorded`: where `memberTreeReason` accepts a member's session's record, the same of each round its tree
  shows opened but the closing round, round 1 of the stream `TREE_SESSION`: the check holds the tree less its last
  opening, and that opening closes the session (`mem_pop_or_back`).
* `membersTree_recorded`, `membersTree_recorded_at`: the same of each session `membersTreeReason` accepts; and, where
  the trees' roots are distinct (as `Verity.Sampled.Lines.trees_log` gives them), of any tree of the log whose root the
  session's record names.

So a release log the members' check accepts opens, of a session's tree, only its closing round and rounds the
session's record gives.

**Not established here**: that the session-round locator puts no release on a round the record gives nothing of (it
needs each line that opens a round to be a round its tree shows opened, which no module this one imports proves), and
how many rounds a session's record gives a stream (the session's verifier's).
-/

@[expose] public section

namespace Verity.SampledTwo.SessionCoins

open Lean (Json)
open Spot Spot.Challenger

/-- **A loop that returns at its first reason, then what follows it**: where none returns, no step of the loop
returned and what follows the loop returns none. -/
theorem forIn_ret_none {α γ σ : Type} {Q : Prop}
    {f : α → Option (Option γ) × σ → Id (ForInStep (Option (Option γ) × σ))}
    {g : Option (Option γ) × σ → Id (Option γ)}
    (hf : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u')))
    (hg : ∀ w u, g (some (some w), u) ≠ none) (hq : ∀ u, g (none, u) = none → Q) {xs : Array α} {u : σ}
    (h : (forIn (m := Id) xs (none, u) f >>= g) = none) : Q := by
  have hf' : ∀ x u s, f x (none, u) = s →
      (∃ w u', s = ForInStep.done (some (some w), u')) ∨ (∃ u', s = ForInStep.yield (none, u') ∧ True) := by
    intro x u s hs
    rcases hf x u s hs with h1 | ⟨u', h2⟩
    · exact .inl h1
    · exact .inr ⟨u', h2, trivial⟩
  rcases Verity.Sampled.Trees.forIn_held_arr (P := fun _ => True) hf' xs u with ⟨w, hw⟩ | ⟨hn, -⟩
  · have e : forIn (m := Id) xs (none, u) f = (some (some w), Prod.snd (forIn (m := Id) xs (none, u) f)) :=
      Prod.ext hw rfl
    rw [e] at h
    exact absurd h (hg w _)
  · have e : forIn (m := Id) xs (none, u) f = (none, Prod.snd (forIn (m := Id) xs (none, u) f)) := Prod.ext hn rfl
    rw [e] at h
    exact hq _ h

/-- **A record `sessionTreeReason` accepts, at any tables, gives each round its tree's log shows opened**: its coin
tree's root is that of a tree of the log, and each round that tree shows opened is below the rounds the record gives
the round's stream (`Spot.Challenger.recordRounds`): the refusal `Spot.Challenger.EXTRA_OPENING`. -/
theorem sessionTree_recorded {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    (h : sessionTreeReason trees tables record = none) :
    ∃ t ∈ trees, Verity.Sampled.recordTree record = t.root ∧
      ∀ o ∈ t.opened, o.round < Spot.Challenger.recordRounds record o.stream := by
  unfold sessionTreeReason at h
  dsimp only [Id.run] at h
  split at h
  case h_2 => cases h
  rename_i t ht
  refine ⟨t, Array.mem_of_find?_eq_some ht, ?_, ?_⟩
  · have hp := Array.find?_some ht
    unfold Verity.Sampled.recordTree
    rw [Verity.Sampled.Trees.strIs hp]
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  split at h
  case isFalse => cases h
  split at h
  case h_2 => cases h
  rcases hlo : t.opening TREE_SESSION 0 with _ | o
  · simp only [hlo, Bool.false_eq_true, ↓reduceIte] at h
    all_goals (split at h <;> (try split at h) <;> cases h)
  simp only [hlo] at h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  split at h
  case isFalse => cases h
  refine forIn_ret_none ?_ (fun w _ hw => by cases hw) (fun u hu => ?_) h
  · intro s u res hres
    split at hres
    case h_2 => exact .inl ⟨_, _, hres.symm⟩
    rename_i name rounds hsr
    refine Or.imp id ?_ (Verity.Sampled.Trees.step_held (P := fun _ => True) ?_ ?_ ?_ hres)
    · rintro ⟨u', hy, -⟩
      exact ⟨u', hy⟩
    · intro x u res' hres'
      split at hres'
      case h_2 => exact .inl ⟨_, _, hres'.symm⟩
      rcases ho : t.opening name x.2 with _ | o
      · simp only [ho, Bool.false_eq_true, ↓reduceIte] at hres'
        all_goals (split at hres' <;> exact .inl ⟨_, _, hres'.symm⟩)
      simp only [ho] at hres'
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      split at hres'
      case isFalse => exact .inl ⟨_, _, hres'.symm⟩
      exact .inr ⟨_, hres'.symm, trivial⟩
    · exact fun _ _ => ⟨(), rfl⟩
    · exact fun _ => ⟨(), rfl⟩
  · cases u
    cases hb : t.opened.all fun o => decide (o.round < Spot.Challenger.recordRounds record o.stream) with
    | true =>
      intro o ho
      rw [← Array.all_toList, List.all_eq_true] at hb
      exact of_decide_eq_true (hb o (Array.mem_def.1 ho))
    | false =>
      rw [hb] at hu
      cases hu

/-- Each member of an array is a member of it less its last, or its last. -/
theorem mem_pop_or_back {α : Type} {xs : Array α} {a c : α} (ha : a ∈ xs) (hc : xs.back? = some c) :
    a ∈ xs.pop ∨ a = c := by
  obtain ⟨ys, rfl⟩ := Array.back?_eq_some_iff.1 hc
  rw [Array.pop_push]
  exact Array.mem_push.1 ha

/-- **A member's session's record `memberTreeReason` accepts gives each round its tree's log shows opened, the closing
round aside**: its coin tree's root is that of a tree of the log, and each round that tree shows opened, but round 1 of
the stream `TREE_SESSION`, is below the rounds the record gives the round's stream. The tree's last opening is the
closing round, and the check holds the tree less it as `sessionTreeReason` holds one (`sessionTree_recorded`). -/
theorem memberTree_recorded {trees : Array LoggedTree} {tables : Array (String × Option ℕ)} {record : Json}
    {evidence : String} (h : memberTreeReason trees tables record evidence = none) :
    ∃ t ∈ trees, getD (getD record "coin_tree") "root" = Json.str t.root ∧
      ∀ o ∈ t.opened, (o.stream, o.round) ≠ (TREE_SESSION, 1) →
        o.round < Spot.Challenger.recordRounds record o.stream := by
  obtain ⟨t, ht, hroot, c, hc, hcl, hs⟩ := Verity.Sampled.MemberForm.memberTreeReason_none h
  obtain ⟨t', ht', -, hall⟩ := sessionTree_recorded hs
  obtain rfl : t' = t.unclosed := Array.mem_singleton.1 ht'
  refine ⟨t, ht, hroot, fun o ho hne => ?_⟩
  rcases mem_pop_or_back ho hc with hp | rfl
  · exact hall o hp
  · exact absurd (Prod.ext (Verity.Sampled.MemberForm.closes_fields hcl).1
      (Verity.Sampled.MemberForm.closes_fields hcl).2.1) hne

/-- **Each session `membersTreeReason` accepts gives each round its tree's log shows opened, the closing round
aside** (`memberTree_recorded`): session `j`'s record names the root of a tree of the log, and each round that tree
shows opened, but round 1 of the stream `TREE_SESSION`, is below the rounds the record gives the round's stream. -/
theorem membersTree_recorded {trees : Array LoggedTree} {lines : List String}
    {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (h : membersTreeReason trees lines records = none) {j : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x) :
    ∃ t ∈ trees, getD (getD x.2.2.1 "coin_tree") "root" = Json.str t.root ∧
      ∀ o ∈ t.opened, (o.stream, o.round) ≠ (TREE_SESSION, 1) →
        o.round < Spot.Challenger.recordRounds x.2.2.1 o.stream :=
  memberTree_recorded
    ((Verity.Sampled.MemberForm.membersTreeReason_none h).1 x (Array.mem_iff_getElem?.2 ⟨j, hx⟩))

/-- **The same of any tree of the log whose root the session's record names**, where the trees' roots are distinct
(as `Verity.Sampled.Lines.trees_log` gives them): the tree is the one `membersTree_recorded` finds. -/
theorem membersTree_recorded_at {trees : Array LoggedTree} {lines : List String}
    {records : Array (String × Array (String × Option ℕ) × Json × String)}
    (h : membersTreeReason trees lines records = none) (hn : (trees.toList.map (·.root)).Nodup) {j : ℕ}
    {x : String × Array (String × Option ℕ) × Json × String} (hx : records[j]? = some x) {t : LoggedTree}
    (ht : t ∈ trees) (hroot : getD (getD x.2.2.1 "coin_tree") "root" = Json.str t.root) {o : TreeOpening}
    (ho : o ∈ t.opened) (hne : (o.stream, o.round) ≠ (TREE_SESSION, 1)) :
    o.round < Spot.Challenger.recordRounds x.2.2.1 o.stream := by
  obtain ⟨t', ht', hroot', hall⟩ := membersTree_recorded h hx
  have e : t' = t := List.inj_on_of_nodup_map hn (Array.mem_toList_iff.2 ht') (Array.mem_toList_iff.2 ht)
    (Json.str.inj (hroot'.symm.trans hroot))
  rw [e] at hall
  exact hall o ho hne

end Verity.SampledTwo.SessionCoins
