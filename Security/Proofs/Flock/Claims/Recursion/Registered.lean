import Proofs.Flock.Claims.Finder
import Definitions.Claims.Recursion

/-!
# The registered finder's search

`Br` (`Claims.Recursion.Br`) searches the V* setup's evaluations outside the oracle (`setupCalls`) for a collision among
them, or with a recorded call (`findCollIn`). This file is what a proof reads of that search:

* `findCollIn_spec`: its output is a true collision wherever every pair it searches is an evaluation of `H`, as every
  setup evaluation is (`setupCalls_true`);
* `findCollIn_isSome_of_sub`: a collision the search finds in smaller lists it finds in larger ones, so a session's
  tie, between its own recorded calls and its statement's setup evaluations (`SetupColl`), is one `Br` finds at the
  run;
* `setupCalls_eq_nil`: a record whose setup makes no evaluation gives `Br` nothing to find.
-/

namespace Claims.Recursion

variable {Coin : Type}

/-- **A setup tie at one V* statement**: among the setup's evaluations at the V* statement `xO`, or between one of
them and the calls `cs`, two different inputs have one output. -/
def SetupColl (H : Msg → Msg) (D : Deploy Coin) (xO : Msg) (cs : List (Msg × Msg)) : Prop :=
  (findCollIn cs (setupCalls H D [xO])).isSome = true

theorem setupCalls_true (H : Msg → Msg) (D : Deploy Coin) (xs : List Msg) :
    ∀ e ∈ setupCalls H D xs, H e.1 = e.2 := by
  intro e he
  simp only [setupCalls, List.mem_flatMap, List.mem_map] at he
  obtain ⟨_, _, u, _, rfl⟩ := he
  rfl

theorem setupCalls_eq_nil (H : Msg → Msg) (D : Deploy Coin) (h : ∀ xO, D.setup xO = []) (xs : List Msg) :
    setupCalls H D xs = [] := by
  simp [setupCalls, h]

theorem mem_setupCalls {H : Msg → Msg} {D : Deploy Coin} {xs : List Msg} {xO : Msg} (hx : xO ∈ xs) :
    ∀ e ∈ setupCalls H D [xO], e ∈ setupCalls H D xs := by
  intro e he
  simp only [setupCalls, List.flatMap_cons, List.flatMap_nil, List.append_nil] at he
  exact List.mem_flatMap.2 ⟨xO, hx, he⟩

/-- `findCollIn`'s output is a true collision where every pair it searches is an evaluation of `H`. -/
theorem findCollIn_spec (H : Msg → Msg) (l : List (Msg × Msg)) :
    ∀ rs : List (Msg × Msg), (∀ p ∈ rs ++ l, H p.1 = p.2) → ∀ u v, findCollIn l rs = some (u, v) →
      Collision H u v := by
  intro rs
  induction rs with
  | nil => intro _ u v h; simp [findCollIn] at h
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro hl u v h
    simp only [findCollIn] at h
    split at h
    · rename_i p hp
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      have hmem := List.mem_of_find?_eq_some hp
      have hpr := List.find?_some hp
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpr
      refine ⟨fun e => hpr.1 e.symm, ?_⟩
      rw [hl (q, a) List.mem_cons_self, hl p (List.mem_cons_of_mem _ hmem), hpr.2]
    · exact ih (fun p hp => hl p (List.mem_cons_of_mem _ hp)) u v h

/-- **`findCollIn` finds a collision wherever one is there**: a pair `e` of `rs` and a pair `e'` of `rs ++ l`, with
different inputs and one output. -/
theorem findCollIn_isSome {l : List (Msg × Msg)} :
    ∀ {rs : List (Msg × Msg)} {e e' : Msg × Msg}, e ∈ rs → e' ∈ rs ++ l → e'.1 ≠ e.1 → e'.2 = e.2 →
      (findCollIn l rs).isSome = true := by
  intro rs
  induction rs with
  | nil => intro e _ he; simp at he
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro e e' he he' h1 h2
    simp only [findCollIn]
    split
    · rfl
    · rename_i hnone
      have hno : ∀ p ∈ tl ++ l, ¬ (p.1 ≠ q ∧ p.2 = a) := fun p hp hc => by
        have := List.find?_eq_none.1 hnone p hp
        simp only [Bool.and_eq_true, decide_eq_true_eq] at this
        exact this hc
      rcases List.mem_cons.1 he with rfl | he
      · rcases List.mem_cons.1 he' with rfl | he'
        · exact absurd rfl h1
        · exact absurd ⟨h1, h2⟩ (hno e' he')
      · rcases List.mem_cons.1 he' with rfl | he'
        · exact absurd ⟨fun h => h1 h.symm, h2.symm⟩ (hno e (List.mem_append_left _ he))
        · exact ih he he' h1 h2

/-- `findCollIn`'s output is two pairs of the lists it searched: a pair of `rs`, and a different input of `rs ++ l`
with the same output. -/
theorem findCollIn_mem {l : List (Msg × Msg)} :
    ∀ {rs : List (Msg × Msg)} {u v : Msg}, findCollIn l rs = some (u, v) →
      ∃ e ∈ rs, ∃ e' ∈ rs ++ l, e.1 = u ∧ e'.1 = v ∧ e'.1 ≠ e.1 ∧ e'.2 = e.2 := by
  intro rs
  induction rs with
  | nil => intro u v h; simp [findCollIn] at h
  | cons hd tl ih =>
    obtain ⟨q, a⟩ := hd
    intro u v h
    simp only [findCollIn] at h
    split at h
    · rename_i p hp
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
      have hmem := List.mem_of_find?_eq_some hp
      have hpr := List.find?_some hp
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hpr
      exact ⟨(q, a), List.mem_cons_self .., p, by simp [hmem], rfl, rfl, hpr.1, hpr.2⟩
    · obtain ⟨e, he, e', he', h⟩ := ih h
      exact ⟨e, List.mem_cons_of_mem _ he, e', by simp only [List.cons_append, List.mem_cons]; exact Or.inr he', h⟩

/-- **A collision `findCollIn` finds in smaller lists it finds in larger ones**: `rs` among `rs'`, and `l` among
`rs' ++ l'`. -/
theorem findCollIn_isSome_of_sub {l rs l' rs' : List (Msg × Msg)} (hr : ∀ e ∈ rs, e ∈ rs')
    (hl : ∀ e ∈ l, e ∈ rs' ++ l') (h : (findCollIn l rs).isSome = true) : (findCollIn l' rs').isSome = true := by
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 h
  obtain ⟨e, he, e', he', -, -, h1, h2⟩ := findCollIn_mem huv
  refine findCollIn_isSome (hr e he) ?_ h1 h2
  rcases List.mem_append.1 he' with h | h
  · exact List.mem_append_left _ (hr e' h)
  · exact hl e' h

/-- **A setup tie at a V* statement is one at every list of statements holding it, and every list of calls holding
the session's**: the search `Br` makes at a run, of its statements' setup evaluations and its recorded calls, finds it. -/
theorem setupColl_lift {H : Msg → Msg} {D : Deploy Coin} {xO : Msg} {cs : List (Msg × Msg)} {xs : List Msg}
    {l : List (Msg × Msg)} (hx : xO ∈ xs) (hcs : ∀ e ∈ cs, e ∈ l) (h : SetupColl H D xO cs) :
    (findCollIn l (setupCalls H D xs)).isSome = true :=
  findCollIn_isSome_of_sub (mem_setupCalls hx) (fun e he => List.mem_append_right _ (hcs e he)) h

end Claims.Recursion
