module

public import Proofs.Properties.Verity.Sampled.Window.Sessions
public import Proofs.Properties.Verity.Sampled.Window.Order
public import Proofs.Properties.Verity.Sampled.Window.Answered

/-!
# Sampled proofs' coupling over a window: which session each answered release is of

The members' sessions' traces answer releases of the one play's log (`Window/Answered.lean`), and the window's
readings say which opening is which session's round by the order the trees are first opened in
(`Window/Sessions.lean`). This module ties the two, on lists alone: the walk's `u`-th session answers exactly the
releases the `u`-th session's rounds pick.

* **The release that first opens a tree** (`mem_openedTrees`, `openedTrees_first_at`, `openedTrees_new_at`): each
  tree a log opened is opened by one of its releases, the `u`-th after a log that opened `u` trees and not it.
* **The sessions held one after another** (`OpensLast`, `opensLast_of_inOrder`): an opening of a tree opened before
  is of the last tree opened, which `OpensInOrder` gives where the sessions' trees are the log's opened trees in
  order.
* **The sessions' rounds in a log** (`SessionPicks`): what the lemmas below read of the readings, which the window's
  readings have in a log that serves each opening (`sessionPicks_windowReadingAt`).
* **What the walk has answered is the sessions before the `u`-th** (`SessDone`): it holds with nothing answered
  (`sessDone_nil`); while it holds and the `u`-th tree is opened, not every pick is answered (`SessDone.not_all`);
  the next release picked is the `u`-th session's link's (`SessDone.first_stop`), the ones a session's trace answers
  after it are the `u`-th session's other rounds' (`SessDone.rep_stop`), the stop after them is the session's closing
  round's or another of its rounds' (`SessDone.close_stop`), and past the closing round, the tree's last opening, the
  sessions before the `u + 1`-th are answered (`SessDone.closed`).
-/

@[expose] public section

namespace Verity.Sampled

theorem slotAt_zero : slotAt 0 = .link := rfl

/-- A session's first place holds the link's slot, at more than one round a session. -/
theorem slotOf_zero (T : ℕ) {per : ℕ} (hper : 1 < per) : slotOf T per 0 = .link :=
  (slotOf_of_ne T (by omega)).trans slotAt_zero

/-! ## The trees a log opens, and the release that first opens each -/

theorem _root_.Verity.Frame.tree?_some {fr : Frame} {i : ℕ} (h : fr.tree? = some i) : ∃ s n, fr = .opening i s n := by
  cases fr with
  | opening i' s n =>
    obtain rfl : i' = i := Option.some.inj h
    exact ⟨s, n, rfl⟩
  | _ => cases h

theorem foldl_openStep_mem {i : ℕ} : ∀ (l : List Release) (acc : List ℕ), i ∈ l.foldl openStep acc →
    i ∈ acc ∨ ∃ r ∈ l, ∃ s n, r.frame = .opening i s n
  | [], _, h => Or.inl h
  | r :: l, acc, h => by
    rw [List.foldl_cons] at h
    rcases foldl_openStep_mem l _ h with h | ⟨r', hr', s, n, hf⟩
    · cases ht : r.frame.tree? with
      | none =>
        rw [openStep_none ht] at h
        exact Or.inl h
      | some i' =>
        rw [openStep_some ht] at h
        by_cases hi : i' ∈ acc
        · rw [ite_eq_left hi] at h
          exact Or.inl h
        · rw [ite_eq_right hi] at h
          rcases List.mem_append.1 h with h | h
          · exact Or.inl h
          · obtain rfl : i = i' := List.mem_singleton.1 h
            obtain ⟨s, n, hf⟩ := Frame.tree?_some ht
            exact Or.inr ⟨r, List.mem_cons_self .., s, n, hf⟩
    · exact Or.inr ⟨r', List.mem_cons_of_mem _ hr', s, n, hf⟩

/-- **A tree a log opened is opened by one of its releases.** -/
theorem mem_openedTrees {L : List Release} {i : ℕ} (h : i ∈ openedTrees L) :
    ∃ r ∈ L, ∃ s n, r.frame = .opening i s n := by
  rcases foldl_openStep_mem L [] h with h | h
  · exact absurd h List.not_mem_nil
  · exact h

theorem foldl_openStep_first {i : ℕ} : ∀ (l : List Release) (acc : List ℕ) (u : ℕ),
    (l.foldl openStep acc)[u]? = some i → acc.length ≤ u →
    ∃ L₁ x L₂ s n, l = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧ i ∉ L₁.foldl openStep acc ∧
      (L₁.foldl openStep acc).length = u
  | [], acc, u, h, hu => by
    have := (List.getElem?_eq_some_iff.1 h).1
    rw [List.foldl_nil] at this
    omega
  | r :: l, acc, u, h, hu => by
    rw [List.foldl_cons] at h
    by_cases hle : (openStep acc r).length ≤ u
    · obtain ⟨L₁, x, L₂, s, n, he, hx, hni, hc⟩ := foldl_openStep_first l _ u h hle
      exact ⟨r :: L₁, x, L₂, s, n, by rw [he, List.cons_append], hx, by rw [List.foldl_cons]; exact hni,
        by rw [List.foldl_cons]; exact hc⟩
    · cases ht : r.frame.tree? with
      | none =>
        rw [openStep_none ht] at hle
        exact absurd hu hle
      | some i' =>
        rw [openStep_some ht] at hle h
        by_cases hi : i' ∈ acc
        · rw [ite_eq_left hi] at hle
          exact absurd hu hle
        · rw [ite_eq_right hi] at hle h
          rw [List.length_append, List.length_singleton] at hle
          have hua : u = acc.length := by omega
          obtain ⟨e, he⟩ := foldl_openStep_prefix l (acc ++ [i'])
          rw [he, List.append_assoc, List.singleton_append, hua, List.getElem?_append_right (Nat.le_refl _),
            Nat.sub_self, List.getElem?_cons_zero] at h
          obtain rfl : i' = i := Option.some.inj h
          obtain ⟨s, n, hf⟩ := Frame.tree?_some ht
          exact ⟨[], r, l, s, n, rfl, hf, hi, hua.symm⟩

/-- **The release that first opens the `u`-th tree a log opens**: an opening of that tree after a log that didn't open
it and opened `u` trees. -/
theorem openedTrees_first_at {L : List Release} {u i : ℕ} (h : (openedTrees L)[u]? = some i) :
    ∃ L₁ x L₂ s n, L = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧ i ∉ openedTrees L₁ ∧
      (openedTrees L₁).length = u :=
  foldl_openStep_first L [] u h (Nat.zero_le _)

/-- **A log that opened a tree not opened before has that tree after the ones opened before.** -/
theorem openedTrees_new_at {L : List Release} {r : Release} {i s n : ℕ} (hr : r.frame = .opening i s n)
    (hi : i ∉ openedTrees L) (d : List Release) :
    (openedTrees (L ++ r :: d))[(openedTrees L).length]? = some i ∧
      (openedTrees L).length < (openedTrees (L ++ r :: d)).length := by
  obtain ⟨e, he⟩ := openedTrees_new hr hi d
  rw [he]
  refine ⟨?_, ?_⟩
  · rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self, List.getElem?_cons_zero]
  · rw [List.length_append, List.length_cons]
    omega

/-- **An opening of a tree opened before is of the last tree opened**, at each release of the log `Lg`: the members'
sessions' rounds are opened one session after another. -/
def OpensLast (Lg : List Release) : Prop :=
  ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → ∀ u, (openedTrees L₁)[u]? = some i →
    u + 1 = (openedTrees L₁).length

/-- **The sessions held one after another open each tree opened before last** (`OpensInOrder`), where the sessions'
trees are the log's opened trees in the order it first opens them. -/
theorem opensLast_of_inOrder {tree : ℕ → ℕ} {Lg : List Release}
    (ht : ∀ u i, (openedTrees Lg)[u]? = some i → tree u = i)
    (h : OpensInOrder tree (openedTrees Lg).length Lg) : OpensLast Lg := by
  intro L₁ x L₂ i s n he hx u hu
  have hul := (List.getElem?_eq_some_iff.1 hu).1
  obtain ⟨v, hv⟩ : ∃ v, v + 1 = (openedTrees L₁).length := ⟨(openedTrees L₁).length - 1, by omega⟩
  have hvl : v < (openedTrees L₁).length := by omega
  have hi' : (openedTrees L₁)[v]? = some (openedTrees L₁)[v] := List.getElem?_eq_getElem hvl
  obtain ⟨q, hq, s', n', hqf⟩ := mem_openedTrees (List.mem_of_getElem? hi')
  have hu' := openedTrees_getElem?_append hu (x :: L₂)
  have hv' := openedTrees_getElem?_append hi' (x :: L₂)
  rw [← he] at hu' hv'
  have h1 := ht u i hu'
  have h2 := ht v _ hv'
  have h3 := h L₁ x L₂ he u (List.getElem?_eq_some_iff.1 hu').1 s n (by rw [h1]; exact hx) q hq v
    (List.getElem?_eq_some_iff.1 hv').1 s' n' (by rw [h2]; exact hqf)
  omega

/-! ## The sessions' rounds in a log -/

section Sessions

variable {R : ℕ}

/-- **The sessions' rounds in the log `Lg`**, at the readings `P`, round `j` of session `su j`, the link's where
`lk j`: a link round picks an opening of a tree not opened before, after as many trees were opened as its session's
place; any other round picks an opening of the tree its session's link opened; the release that first opens a tree is
the pick of the link round of the session at as many trees as were opened before; no two rounds pick one release; each
session has one link round; and each round picks at most one release. -/
structure SessionPicks (P : Fin R → List Release → Frame → Prop) (su : Fin R → ℕ) (lk : Fin R → Prop)
    (Lg : List Release) : Prop where
  link : ∀ j L₁, lk j → PicksIn P Lg j L₁ → ∃ x L₂ i s n, Lg = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧
    i ∉ openedTrees L₁ ∧ (openedTrees L₁).length = su j
  rep : ∀ j L₁, ¬ lk j → PicksIn P Lg j L₁ → ∃ x L₂ i s n, Lg = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧
    (openedTrees L₁)[su j]? = some i
  first : ∀ L₁ x L₂ i s n, Lg = L₁ ++ x :: L₂ → x.frame = .opening i s n → i ∉ openedTrees L₁ →
    ∃ j, lk j ∧ su j = (openedTrees L₁).length ∧ PicksIn P Lg j L₁
  unique : ∀ j j' L₁, PicksIn P Lg j L₁ → PicksIn P Lg j' L₁ → j = j'
  link_inj : ∀ j j', lk j → lk j' → su j = su j' → j = j'
  once : PicksOnce P Lg

/-- **What the walk has answered is the sessions before the `u`-th**: a release of `Lg` a round picks is after an
answered log exactly where the round is of a session before the `u`-th. -/
def SessDone (P : Fin R → List Release → Frame → Prop) (su : Fin R → ℕ) (Lg : List Release)
    (done : List (List Release)) (u : ℕ) : Prop :=
  ∀ j L₁, PicksIn P Lg j L₁ → (L₁ ∈ done ↔ su j < u)

/-- **Nothing answered is no session done.** -/
theorem sessDone_nil (P : Fin R → List Release → Frame → Prop) (su : Fin R → ℕ) (Lg : List Release) :
    SessDone P su Lg [] 0 :=
  fun _ _ _ => ⟨fun h => absurd h List.not_mem_nil, fun h => absurd h (Nat.not_lt_zero _)⟩

variable {P : Fin R → List Release → Frame → Prop} {su : Fin R → ℕ} {lk : Fin R → Prop} {Lg : List Release}

/-- **A session's link round's pick opens the session's tree**: the tree at the session's place among the trees `Lg`
opens, not opened before it. -/
theorem SessionPicks.link_opens (hS : SessionPicks P su lk Lg) {j : Fin R} {L₁ : List Release} (hl : lk j)
    (hp : PicksIn P Lg j L₁) : ∃ x L₂ i s n, Lg = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧
      i ∉ openedTrees L₁ ∧ (openedTrees L₁).length = su j ∧ (openedTrees Lg)[su j]? = some i := by
  obtain ⟨x, L₂, i, s, n, he, hx, hni, hc⟩ := hS.link j L₁ hl hp
  refine ⟨x, L₂, i, s, n, he, hx, hni, hc, ?_⟩
  rw [he, ← hc]
  exact (openedTrees_new_at hx hni L₂).1

/-- **Any other round's pick opens the session's tree**, opened before it. -/
theorem SessionPicks.rep_opens (hS : SessionPicks P su lk Lg) {j : Fin R} {L₁ : List Release} (hl : ¬ lk j)
    (hp : PicksIn P Lg j L₁) : ∃ x L₂ i s n, Lg = L₁ ++ x :: L₂ ∧ x.frame = .opening i s n ∧
      (openedTrees L₁)[su j]? = some i ∧ (openedTrees Lg)[su j]? = some i := by
  obtain ⟨x, L₂, i, s, n, he, hx, hti⟩ := hS.rep j L₁ hl hp
  refine ⟨x, L₂, i, s, n, he, hx, hti, ?_⟩
  rw [he]
  exact openedTrees_getElem?_append hti _

/-- **The link round of a session whose tree a prefix of `Lg` opened picks before that prefix's end.** -/
theorem SessionPicks.link_before (hS : SessionPicks P su lk Lg) {L d : List Release} (hLg : Lg = L ++ d) {v : ℕ}
    (hv : v < (openedTrees L).length) :
    ∃ j L₁, lk j ∧ su j = v ∧ PicksIn P Lg j L₁ ∧ L₁.length < L.length := by
  obtain ⟨L₁, x, L₂, s, n, he, hx, hni, hc⟩ := openedTrees_first_at (List.getElem?_eq_getElem hv)
  obtain ⟨j, hl, hsu, hp⟩ := hS.first L₁ x (L₂ ++ d) _ s n
    (by rw [hLg, he, List.append_assoc, List.cons_append]) hx hni
  refine ⟨j, L₁, hl, hsu.trans hc, hp, ?_⟩
  rw [he, List.length_append, List.length_cons]
  omega

/-- **Not every pick is answered while a session's tree is opened and its session not done.** -/
theorem SessDone.not_all (hS : SessionPicks P su lk Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) (hu : u < (openedTrees Lg).length)
    (hall : ∀ j L₁, PicksIn P Lg j L₁ → L₁ ∈ done) : False := by
  obtain ⟨j, L₁, -, hsu, hp, -⟩ := hS.link_before (List.append_nil Lg).symm hu
  have := (hD j L₁ hp).1 (hall j L₁ hp)
  omega

/-- **The first release picked after the sessions before the `u`-th is the `u`-th session's link's**: where every
pick before the log `L₀` is answered and every answered log is shorter than `L₀`, a round that picks after `L₀` is the
link round of session `u`. -/
theorem SessDone.first_stop (hS : SessionPicks P su lk Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) {j₀ : Fin R} {L₀ : List Release} (h₀ : PicksIn P Lg j₀ L₀)
    (hF : ∀ j L₁, PicksIn P Lg j L₁ → L₁.length < L₀.length → L₁ ∈ done)
    (hlt : ∀ L₁ ∈ done, L₁.length < L₀.length) : lk j₀ ∧ su j₀ = u := by
  have hge : ¬ su j₀ < u := fun h => Nat.lt_irrefl _ (hlt L₀ ((hD j₀ L₀ h₀).2 h))
  by_cases hl : lk j₀
  · refine ⟨hl, ?_⟩
    obtain ⟨x, L₂, i, s, n, he, hx, hni, hc⟩ := hS.link j₀ L₀ hl h₀
    by_cases hne : su j₀ = u
    · exact hne
    · obtain ⟨j, L₁, -, hsu, hp, hlen⟩ := hS.link_before (v := u) he (by omega)
      have := (hD j L₁ hp).1 (hF j L₁ hp hlen)
      omega
  · obtain ⟨x, L₂, i, s, n, he, hx, hti⟩ := hS.rep j₀ L₀ hl h₀
    obtain ⟨j, L₁, -, hsu, hp, hlen⟩ := hS.link_before he (List.getElem?_eq_some_iff.1 hti).1
    have := (hD j L₁ hp).1 (hF j L₁ hp hlen)
    omega

/-- **A release picked after the `u`-th session's link's, among releases no link round picks, is the `u`-th
session's**: where the `u`-th session's link round picks after `L₀`, each log of `ms` is longer, picked by a round
that is no link's, and every pick before one of them is after `L₀`, after one of `ms` or answered, a round that picks
after a log of `ms` is of session `u` and no link's. -/
theorem SessDone.rep_stop (hS : SessionPicks P su lk Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) {j₀ : Fin R} {L₀ : List Release} (h₀ : PicksIn P Lg j₀ L₀) (hu₀ : su j₀ = u)
    {ms : List (List Release)} (hms : ∀ m ∈ ms, ∃ j, ¬ lk j ∧ PicksIn P Lg j m)
    (hlt : ∀ L₁ ∈ done, L₁.length < L₀.length) (hgt : ∀ m ∈ ms, L₀.length < m.length) {m : List Release}
    (hm : m ∈ ms) {j : Fin R} (hj : PicksIn P Lg j m)
    (hF : ∀ j' L₁, PicksIn P Lg j' L₁ → L₁.length < m.length → L₁ = L₀ ∨ L₁ ∈ ms ∨ L₁ ∈ done) :
    ¬ lk j ∧ su j = u := by
  obtain ⟨j', hl', hp'⟩ := hms m hm
  obtain rfl := hS.unique j j' m hj hp'
  refine ⟨hl', ?_⟩
  have hge : ¬ su j < u := fun h => by
    have h1 := hlt m ((hD j m hj).2 h)
    have h2 := hgt m hm
    omega
  by_cases hne : su j = u
  · exact hne
  · obtain ⟨x, L₂, i, s, n, he, hx, hti⟩ := hS.rep j m hl' hj
    obtain ⟨j₁, L₁, hl₁, hsu, hp, hlen⟩ := hS.link_before he (List.getElem?_eq_some_iff.1 hti).1
    rcases hF j₁ L₁ hp hlen with rfl | h | h
    · obtain rfl := hS.unique j₁ j₀ L₁ hp h₀
      omega
    · obtain ⟨j₂, hl₂, hp₂⟩ := hms L₁ h
      obtain rfl := hS.unique j₁ j₂ L₁ hp hp₂
      exact absurd hl₁ hl₂
    · have := (hD j₁ L₁ hp).1 h
      omega

/-- **The stop after a session's answered rounds is its closing round's, or another of its rounds'**: where the
sessions before the `u`-th are answered (`SessDone`), session `u`'s link's pick is answered, the walk stands
(`Walked`) with the closing round `jc` of session `u` picking after a log `Lc` not answered, and from where the walk
stands no live round picks through `L`, after which the play ends or a live round picks, the play does not end there:
a live round picks, and it is `jc`, picking after `Lc`, or another round of session `u` that is no link's. An
opening of a tree opened before is of the last tree opened (`OpensLast`), so no later session's round picks before
`Lc`. -/
theorem SessDone.close_stop (hS : SessionPicks P su lk Lg) (hO : OpensLast Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) {j₀ : Fin R} {L₀ : List Release} (h₀ : PicksIn P Lg j₀ L₀) (hl₀ : lk j₀)
    (hu₀ : su j₀ = u) {done' : List (List Release)} (hd : ∀ L₁ ∈ done, L₁ ∈ done') (hL₀ : L₀ ∈ done')
    {pre' : List Release} {a' : Fin R → Prop} (hW : Walked P Lg pre' a' done') {jc : Fin R} {Lc : List Release}
    (hc : PicksIn P Lg jc Lc) (hcl : ¬ lk jc) (hcu : su jc = u) (hcd : Lc ∉ done') {L : List Release}
    (hNP : NoPick P a' pre' L)
    (hstop : Lg = pre' ++ L ∨ ∃ x L' j, Lg = (pre' ++ L) ++ x :: L' ∧ a' j ∧ P j (pre' ++ L) x.frame) :
    ∃ x L' j, Lg = (pre' ++ L) ++ x :: L' ∧ a' j ∧ P j (pre' ++ L) x.frame ∧
      ((j = jc ∧ pre' ++ L = Lc) ∨ (su j = u ∧ ¬ lk j ∧ j ≠ jc)) := by
  rcases hstop with hLg | ⟨x, L', j, hLg, haj, hPj⟩
  · exfalso
    obtain ⟨r, L₂, he, -⟩ := id hc
    have hl' := congrArg List.length (he.symm.trans hLg)
    rw [List.length_append, List.length_cons, List.length_append] at hl'
    exact hcd (hW.seg hS.once hNP (L₃ := []) (by rw [List.append_nil]; exact hLg) hc (by omega))
  · refine ⟨x, L', j, hLg, haj, hPj, ?_⟩
    have hps : PicksIn P Lg j (pre' ++ L) := ⟨x, L', hLg, hPj⟩
    have hnd : pre' ++ L ∉ done' := fun h => hW.live j _ h hps haj
    by_cases hjc : j = jc
    · refine Or.inl ⟨hjc, ?_⟩
      rw [hjc] at hps
      exact hS.once jc _ _ hps hc
    refine Or.inr ?_
    have hle : (pre' ++ L).length ≤ Lc.length := by
      by_contra hlt
      have hseg : Lg = pre' ++ (L ++ x :: L') := by rw [hLg, List.append_assoc]
      rw [List.length_append] at hlt
      exact hcd (hW.seg hS.once hNP hseg hc (by omega))
    have hge : ¬ su j < u := fun h => hnd (hd _ ((hD j _ hps).2 h))
    obtain ⟨xc, Lc₂, ic, sc, nc, hec, hxc, htc⟩ := hS.rep jc Lc hcl hc
    have hcnt := hO Lc xc Lc₂ ic sc nc hec hxc (su jc) htc
    obtain ⟨d, hd'⟩ := prefix_le (hec.symm.trans hLg) hle
    have hmono : (openedTrees (pre' ++ L)).length ≤ (openedTrees Lc).length := by
      obtain ⟨e, he⟩ := openedTrees_append (pre' ++ L) d
      rw [hd', he, List.length_append]
      omega
    by_cases hl : lk j
    · exfalso
      obtain ⟨x₁, L₂, i, s, n, he₁, hx₁, hni, hcn⟩ := hS.link j _ hl hps
      have hxx : x₁ = x := (List.cons.inj (List.append_cancel_left (he₁.symm.trans hLg))).1
      rw [hxx] at hx₁
      cases d with
      | nil =>
        rw [List.append_nil] at hd'
        rw [← hd'] at hps
        exact hjc (hS.unique j jc Lc hps hc)
      | cons y d =>
        have hy : y = x := by
          rw [hd', List.append_assoc, List.cons_append] at hec
          exact (List.cons.inj (List.append_cancel_left (hec.symm.trans hLg))).1
        rw [hy] at hd'
        have h1 := (openedTrees_new_at hx₁ hni d).2
        rw [← hd'] at h1
        have hju : su j = u := by omega
        have hjj := hS.link_inj j j₀ hl hl₀ (hju.trans hu₀.symm)
        rw [hjj] at hps
        have e := hS.once j₀ _ _ hps h₀
        rw [e] at hnd
        exact hnd hL₀
    · obtain ⟨x₁, L₂, i, s, n, he₁, hx₁, hti⟩ := hS.rep j _ hl hps
      have h1 := (List.getElem?_eq_some_iff.1 hti).1
      exact ⟨by omega, hl, hjc⟩

/-- **Past a session's closing round the session is answered**: where the sessions before the `u`-th are answered
(`SessDone`), the answered logs are those, the one session `u`'s link's round picks after, the logs `ms` rounds of
session `u` pick after and the one `Lc` its closing round `jc` picks after, the walk stands at `Lc` with those
answered (`Walked`), and no round of session `u` picks after a log longer than `Lc`, the sessions before the
`u + 1`-th are answered. -/
theorem SessDone.closed (hS : SessionPicks P su lk Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) {j₀ : Fin R} {L₀ : List Release} (h₀ : PicksIn P Lg j₀ L₀) (hu₀ : su j₀ = u)
    {ms : List (List Release)} (hrep : ∀ m ∈ ms, ∀ j, PicksIn P Lg j m → su j = u) {jc : Fin R}
    {Lc : List Release} (hc : PicksIn P Lg jc Lc) (hcu : su jc = u) {done'' : List (List Release)}
    (hd'' : ∀ L₁, L₁ ∈ done'' ↔ L₁ = Lc ∨ L₁ = L₀ ∨ L₁ ∈ ms ∨ L₁ ∈ done) {a'' : Fin R → Prop}
    (hW : Walked P Lg Lc a'' done'')
    (hlast : ∀ j L₁, PicksIn P Lg j L₁ → su j = u → L₁.length ≤ Lc.length) :
    SessDone P su Lg done'' (u + 1) := by
  intro j L₁ hp
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rcases (hd'' L₁).1 h with h | h | h | h
    · rw [h] at hp
      have := hS.unique j jc _ hp hc
      rw [this, hcu]
      exact Nat.lt_succ_self u
    · rw [h] at hp
      have := hS.unique j j₀ _ hp h₀
      rw [this, hu₀]
      exact Nat.lt_succ_self u
    · rw [hrep L₁ h j hp]
      exact Nat.lt_succ_self u
    · exact Nat.lt_succ_of_lt ((hD j L₁ hp).1 h)
  · by_cases hju : su j < u
    · exact (hd'' L₁).2 (Or.inr (Or.inr (Or.inr ((hD j L₁ hp).2 hju))))
    have hle := hlast j L₁ hp (by omega)
    by_cases hlt : L₁.length < Lc.length
    · exact hW.before j L₁ hp hlt
    · obtain ⟨r, L₂, he, -⟩ := id hp
      obtain ⟨rc, Lc₂, hec, -⟩ := id hc
      exact (hd'' L₁).2 (Or.inl (List.append_inj (he.symm.trans hec) (by omega)).1)

end Sessions

/-! ## The window's readings' sessions -/

/-- **The window's readings' rounds are the sessions' rounds** (`SessionPicks`) in a log that serves each opening,
whose trees each name their streams distinctly and are no more than the sessions the readings' rounds hold: round `j`
is session `j / per`'s, the link's at the session's first place. The closing round's pick, as a rep round's, is an
opening of the tree the session's link opened. -/
theorem sessionPicks_windowReadingAt {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup)
    (hT : 0 < tables.size) {per : ℕ} (hper : 1 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) {Lg : List Release}
    (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) (hS : LogServed Lg)
    (hR : (openedTrees Lg).length * per ≤ R) :
    SessionPicks (windowReadingAt (R := R) tables per reveal record recordDigest).round (fun j => j.val / per)
      (fun j => slotOf tables.size per (j.val % per) = .link) Lg where
  link j L₁ hl := by
    rintro ⟨x, L₂, he, hp⟩
    have hp' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) L₁ x.frame := hp
    rw [hl] at hp'
    obtain ⟨i, s, n, hfr, -, hni, hc⟩ := hp'
    exact ⟨x, L₂, i, s, n, he, hfr, hni, hc⟩
  rep j L₁ hl := by
    rintro ⟨x, L₂, he, hp⟩
    have hp' : sessionRound tables (j.val / per) (slotOf tables.size per (j.val % per)) L₁ x.frame := hp
    cases hσ : slotOf tables.size per (j.val % per) with
    | link => exact absurd hσ hl
    | rep t r k =>
      rw [hσ] at hp'
      obtain ⟨i, s, n, hfr, -, hti, -⟩ := hp'
      exact ⟨x, L₂, i, s, n, he, hfr, hti⟩
  first L₁ x L₂ i s n he hx hni := by
    have h1 := (openedTrees_new_at hx hni L₂).2
    rw [← he] at h1
    have h2 : ((openedTrees L₁).length + 1) * per ≤ (openedTrees Lg).length * per := Nat.mul_le_mul_right per h1
    rw [Nat.succ_mul] at h2
    have hlt : (openedTrees L₁).length * per < R := by omega
    have hper0 : 0 < per := by omega
    have hdiv : (openedTrees L₁).length * per / per = (openedTrees L₁).length := Nat.mul_div_cancel _ hper0
    have hmod : (openedTrees L₁).length * per % per = 0 := Nat.mul_mod_left _ _
    refine ⟨⟨(openedTrees L₁).length * per, hlt⟩, ?_, hdiv, x, L₂, he, ?_⟩
    · show slotOf tables.size per ((openedTrees L₁).length * per % per) = .link
      rw [hmod]
      exact slotOf_zero _ hper
    · show sessionRound tables ((openedTrees L₁).length * per / per)
        (slotOf tables.size per ((openedTrees L₁).length * per % per)) L₁ x.frame
      rw [hdiv, hmod, slotOf_zero _ hper]
      exact ⟨i, s, n, hx, hS L₁ x L₂ i s n he hx, hni, rfl⟩
  unique j j' L₁ := by
    rintro ⟨x, L₂, he, hp⟩ ⟨x', L₂', he', hp'⟩
    obtain rfl : x = x' := (List.cons.inj (List.append_cancel_left (he.symm.trans he'))).1
    exact windowReadingAt_unique ht hT (by omega) reveal record recordDigest hp hp'
  link_inj j j' hl hl' hsu := by
    have hper0 : 0 < per := by omega
    have hm : j.val % per = j'.val % per :=
      slotOf_inj hT (Nat.mod_lt _ hper0) (Nat.mod_lt _ hper0) (hl.trans hl'.symm)
    have hd : j.val / per = j'.val / per := hsu
    exact Fin.ext (by rw [← Nat.div_add_mod j.val per, ← Nat.div_add_mod j'.val per, hd, hm])
  once := picksOnce_windowReadingAt tables per reveal record recordDigest hn

end Verity.Sampled
