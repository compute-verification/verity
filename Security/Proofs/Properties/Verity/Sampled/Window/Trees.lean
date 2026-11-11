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
  after it are the `u`-th session's other rounds' (`SessDone.rep_stop`), and where the trace ends after the session's
  last round, at the play's end or at a stop a link round picks, the sessions before the `u + 1`-th are answered
  (`SessDone.next`).
-/

@[expose] public section

namespace Verity.Sampled

theorem slotAt_zero : slotAt 0 = .link := rfl

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

/-- **A session's walk answers the session**: where the sessions before the `u`-th are answered (`SessDone`), the
next release picked is after `L₀`, longer than every answered log, the releases picked after it are after the logs
`ms`, each by a round that is no link's, and the walk stands, with those answered too (`Walked`), after the session's
last round (`per ≤ ms.length + 1`, the session's rounds `per`), at the play's end, or at a stop a link round picks,
the sessions before the `u + 1`-th are answered, where an opening of a tree opened before is of the last tree opened
(`OpensLast`). -/
theorem SessDone.next (hS : SessionPicks P su lk Lg) (hO : OpensLast Lg) {done : List (List Release)} {u : ℕ}
    (hD : SessDone P su Lg done u) {j₀ : Fin R} {L₀ : List Release} (h₀ : PicksIn P Lg j₀ L₀)
    (hlt : ∀ L₁ ∈ done, L₁.length < L₀.length) {ms : List (List Release)}
    (hms : ∀ m ∈ ms, ∃ j, ¬ lk j ∧ PicksIn P Lg j m) (hgt : ∀ m ∈ ms, L₀.length < m.length) (hnd : ms.Nodup)
    {done' : List (List Release)} (hd' : ∀ L₁, L₁ ∈ done' ↔ L₁ = L₀ ∨ L₁ ∈ ms ∨ L₁ ∈ done)
    {pre' : List Release} {a' : Fin R → Prop} (hW : Walked P Lg pre' a' done') {per : ℕ} (e : Fin per → Fin R)
    (he : ∀ j, su j = u → ∃ j', e j' = j)
    (hend : per ≤ ms.length + 1 ∨ ∃ L, NoPick P a' pre' L ∧ (Lg = pre' ++ L ∨
      ∃ x L' j, Lg = (pre' ++ L) ++ x :: L' ∧ a' j ∧ lk j ∧ P j (pre' ++ L) x.frame)) :
    (lk j₀ ∧ su j₀ = u) ∧ (∀ m ∈ ms, ∀ j, PicksIn P Lg j m → ¬ lk j ∧ su j = u) ∧
      SessDone P su Lg done' (u + 1) := by
  have hL₀ : L₀ ∈ done' := (hd' L₀).2 (Or.inl rfl)
  have hF₁ : ∀ j L₁, PicksIn P Lg j L₁ → L₁.length < L₀.length → L₁ ∈ done := fun j L₁ hp hl => by
    have h1 := (hW.picked L₀ hL₀).1
    rcases (hd' L₁).1 (hW.before j L₁ hp (by omega)) with rfl | h | h
    · exact absurd hl (Nat.lt_irrefl _)
    · have := hgt L₁ h
      omega
    · exact h
  obtain ⟨hl₀, hu₀⟩ := hD.first_stop hS h₀ hF₁ hlt
  have hrep : ∀ m ∈ ms, ∀ j, PicksIn P Lg j m → ¬ lk j ∧ su j = u := fun m hm j hj =>
    hD.rep_stop hS h₀ hu₀ hms hlt hgt hm hj fun j' L₁ hp hl => by
      have h1 := (hW.picked m ((hd' m).2 (Or.inr (Or.inl hm)))).1
      exact (hd' L₁).1 (hW.before j' L₁ hp (by omega))
  refine ⟨⟨hl₀, hu₀⟩, hrep, fun j L₁ hp => ⟨fun h => ?_, fun h => ?_⟩⟩
  · rcases (hd' L₁).1 h with rfl | h | h
    · obtain rfl := hS.unique j j₀ L₁ hp h₀
      omega
    · have := (hrep L₁ h j hp).2
      omega
    · have := (hD j L₁ hp).1 h
      omega
  · by_cases hju : su j < u
    · exact (hd' L₁).2 (Or.inr (Or.inr ((hD j L₁ hp).2 hju)))
    have hju : su j = u := by omega
    by_cases hl : lk j
    · obtain rfl := hS.link_inj j j₀ hl hl₀ (hju.trans hu₀.symm)
      exact (hd' L₁).2 (Or.inl (hS.once j L₁ L₀ hp h₀))
    by_cases hlen : L₁.length < pre'.length
    · exact hW.before j L₁ hp hlen
    rcases hend with hend | ⟨L, hNP, hLg | ⟨x, L', js, hLg, haj, hlj, hPj⟩⟩
    · obtain ⟨j', rfl⟩ := he j hju
      obtain ⟨j₀', rfl⟩ := he j₀ hu₀
      have hU' : PicksOnce (fun j' => P (e j')) Lg := fun j L₁ L₁' h h' => hS.once (e j) L₁ L₁' h h'
      have hpick : ∀ m ∈ ms, ∃ j, PicksIn (fun j' => P (e j')) Lg j m := fun m hm => by
        obtain ⟨j₃, -, hp₃⟩ := hms m hm
        obtain ⟨j₃', rfl⟩ := he j₃ (hrep m hm j₃ hp₃).2
        exact ⟨j₃', hp₃⟩
      rcases covers_of_length hU' (j₀ := j₀') h₀ hpick hgt hnd hend j' L₁ hp with h | h
      · exact (hd' L₁).2 (Or.inl h)
      · exact (hd' L₁).2 (Or.inr (Or.inl h))
    · obtain ⟨r, L₂, he₁, -⟩ := id hp
      have hl' := congrArg List.length (he₁.symm.trans hLg)
      rw [List.length_append, List.length_cons, List.length_append] at hl'
      exact hW.seg hS.once hNP (L₃ := []) (by rw [List.append_nil]; exact hLg) hp (by omega)
    · have hseg : Lg = pre' ++ (L ++ x :: L') := by rw [hLg, List.append_assoc]
      by_cases hlen' : L₁.length < pre'.length + L.length
      · exact hW.seg hS.once hNP hseg hp hlen'
      exfalso
      have hps : PicksIn P Lg js (pre' ++ L) := ⟨x, L', hLg, hPj⟩
      obtain ⟨x₁, L₂, i, s, n, he₁, hx₁, hti⟩ := hS.rep j L₁ hl hp
      obtain ⟨d, hd⟩ := prefix_le (he₁.symm.trans hLg) (by rw [List.length_append]; omega)
      cases d with
      | nil =>
        rw [List.append_nil] at hd
        rw [hd] at hp
        exact hl (hS.unique j js _ hp hps ▸ hlj)
      | cons y d =>
        have hy : y = x := by
          rw [hd, List.append_assoc, List.cons_append] at he₁
          exact (List.cons.inj (List.append_cancel_left (he₁.symm.trans hLg))).1
        subst hy
        obtain ⟨xs, Ls, is, ss, ns, hes, hxs, hnis, hcs⟩ := hS.link js (pre' ++ L) hlj hps
        obtain rfl : xs = y := (List.cons.inj (List.append_cancel_left (hes.symm.trans hLg))).1
        have h1 := (openedTrees_new_at hxs hnis d).2
        rw [← hd] at h1
        have h2 := hO L₁ x₁ L₂ i s n he₁ hx₁ (su j) hti
        obtain ⟨x₀, L₂₀, i₀, s₀, n₀, he₀, hx₀, hni₀, hc₀⟩ := hS.link j₀ L₀ hl₀ h₀
        have h3 := (hW.picked L₀ hL₀).1
        have hlt₀ : L₀.length < (pre' ++ L).length := by
          rw [List.length_append]
          by_cases hL : L₀.length = pre'.length + L.length
          · have hLe : L₀ = pre' ++ L :=
              (List.append_inj (he₀.symm.trans hLg) (by rw [List.length_append]; exact hL)).1
            rw [← hLe] at hps
            exact absurd haj (hW.live js L₀ hL₀ hps)
          · omega
        obtain ⟨d₀, hd₀⟩ := prefix_le (hLg.symm.trans he₀) (Nat.le_of_lt hlt₀)
        cases d₀ with
        | nil =>
          rw [List.append_nil] at hd₀
          rw [hd₀] at hlt₀
          exact Nat.lt_irrefl _ hlt₀
        | cons y₀ d₀ =>
          have hy₀ : y₀ = x₀ := by
            rw [hd₀, List.append_assoc, List.cons_append] at hLg
            exact (List.cons.inj (List.append_cancel_left (hLg.symm.trans he₀))).1
          subst hy₀
          have h4 := (openedTrees_new_at hx₀ hni₀ d₀).2
          rw [← hd₀] at h4
          omega

end Sessions

/-! ## The window's readings' sessions -/

/-- **The window's readings' rounds are the sessions' rounds** (`SessionPicks`) in a log that serves each opening,
whose trees each name their streams distinctly and are no more than the sessions the readings' rounds hold: round `j`
is session `j / per`'s, the link's at the layout's first slot. -/
theorem sessionPicks_windowReadingAt {R : ℕ} {tables : Array String} (ht : tables.toList.Nodup) {per : ℕ}
    (hper : 0 < per) (reveal : ℕ → Slot → Exchange → Option Claims.Msg) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {Lg : List Release} (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup)
    (hS : LogServed Lg) (hR : (openedTrees Lg).length * per ≤ R) :
    SessionPicks (windowReadingAt (R := R) tables per reveal record recordDigest).round (fun j => j.val / per)
      (fun j => slotAt (j.val % per) = .link) Lg where
  link j L₁ hl := by
    rintro ⟨x, L₂, he, hp⟩
    have hp' : sessionRound tables (j.val / per) (slotAt (j.val % per)) L₁ x.frame := hp
    rw [hl] at hp'
    obtain ⟨i, s, n, hfr, -, hni, hc⟩ := hp'
    exact ⟨x, L₂, i, s, n, he, hfr, hni, hc⟩
  rep j L₁ hl := by
    rintro ⟨x, L₂, he, hp⟩
    have hp' : sessionRound tables (j.val / per) (slotAt (j.val % per)) L₁ x.frame := hp
    cases hσ : slotAt (j.val % per) with
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
    have hdiv : (openedTrees L₁).length * per / per = (openedTrees L₁).length := Nat.mul_div_cancel _ hper
    have hmod : (openedTrees L₁).length * per % per = 0 := Nat.mul_mod_left _ _
    refine ⟨⟨(openedTrees L₁).length * per, hlt⟩, ?_, hdiv, x, L₂, he, ?_⟩
    · show slotAt ((openedTrees L₁).length * per % per) = .link
      rw [hmod]
      exact slotAt_zero
    · show sessionRound tables ((openedTrees L₁).length * per / per)
        (slotAt ((openedTrees L₁).length * per % per)) L₁ x.frame
      rw [hdiv, hmod, slotAt_zero]
      exact ⟨i, s, n, hx, hS L₁ x L₂ i s n he hx, hni, rfl⟩
  unique j j' L₁ := by
    rintro ⟨x, L₂, he, hp⟩ ⟨x', L₂', he', hp'⟩
    obtain rfl : x = x' := (List.cons.inj (List.append_cancel_left (he.symm.trans he'))).1
    exact windowReadingAt_unique ht per reveal record recordDigest hp hp'
  link_inj j j' hl hl' hsu := by
    have hm : j.val % per = j'.val % per := slotAt_injective (hl.trans hl'.symm)
    have hd : j.val / per = j'.val / per := hsu
    exact Fin.ext (by rw [← Nat.div_add_mod j.val per, ← Nat.div_add_mod j'.val per, hd, hm])
  once := picksOnce_windowReadingAt tables per reveal record recordDigest hn

end Verity.Sampled
