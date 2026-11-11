module

public import Proofs.Properties.Verity.Sampled.Window.Record

/-!
# Sampled proofs' coupling over a window: a session's read with its closing opening dropped

A session's closing round is one more opening of its tree's stream named the session's, after the link's. What the
verifier's read holds the log to at a session, less that round (`SessionOpens`), reads the log's openings only
through three things: the trees opened before a release (`openedTrees`), the trees' shapes (`helloSpecs`), and the
rounds opened of a stream of the session's tree other than the one named the session's (`openedOf`). An opening of
an already-opened tree's stream named the session's changes none of the three, so `SessionOpens` holds of a log
exactly when it holds of the log with that opening dropped:

* **The three reads** (`openedTrees_drop`, `helloSpecs_drop`, `openedOf_drop`, `DropPre.reads`): at a prefix through
  the dropped opening and the same prefix without it.
* **The releases correspond** (`drop_split`, `drop_unsplit`): each release of the log but the dropped opening is a
  release of the log without it, after a prefix that corresponds, and the other way.
* **The read, less the closing round, holds of both or neither** (`SessionOpens.of_drop`, `SessionOpens.drop`,
  `SessionOpens.drop_iff`), for every session of the window, the dropped opening's own and the others'.
* **The read with the closing round** (`SessionRead.of_dropped`): where the dropped opening is the second of the
  `u`-th tree's stream named the session's, answers the SHA-512 of the session's evidence, and no opening of that
  tree follows it, the read less the closing round on the log without it gives the session's read on the log.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open Lean (Json)
open Spot.Challenger

/-! ## The three reads -/

/-- **An opening of a tree the log before it opened leaves the opened trees as they are.** -/
theorem openedTrees_drop {Lc : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    (hi : i ∈ openedTrees Lc) (M : List Release) : openedTrees (Lc ++ x :: M) = openedTrees (Lc ++ M) := by
  have ht : x.frame.tree? = some i := by
    rw [hx]
    rfl
  unfold openedTrees at hi ⊢
  rw [List.foldl_append, List.foldl_append, List.foldl_cons, openStep_some ht, ite_eq_left hi]

/-- **An opening samples no tree.** -/
theorem helloSpecs_drop {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n) (Lc M : List Release) :
    helloSpecs (Lc ++ x :: M) = helloSpecs (Lc ++ M) := by
  have h1 : helloSpecs (x :: M) = helloSpecs M := by
    unfold helloSpecs
    rw [List.filterMap_cons, hx]
  rw [Verity.Sampled.helloSpecs_append, h1, ← Verity.Sampled.helloSpecs_append]

/-- **An opening of one stream opens no round of another.** -/
theorem openedOf_drop {x : Release} {i s n i' s' : ℕ} (hx : x.frame = .opening i s n) (hne : ¬ (i = i' ∧ s = s'))
    (Lc M : List Release) : openedOf (Lc ++ x :: M) i' s' = openedOf (Lc ++ M) i' s' := by
  have h1 : x.frame.opensAt i' s' = false := by
    rw [hx]
    show (i == i' && s == s') = false
    rw [Bool.and_eq_false_iff]
    by_cases h : i = i'
    · exact Or.inr (beq_eq_false_iff_ne.2 fun h' => hne ⟨h, h'⟩)
    · exact Or.inl (beq_eq_false_iff_ne.2 h)
  unfold openedOf
  rw [List.countP_append, List.countP_append, List.countP_cons, h1]
  rfl

/-- **A prefix of the log and the prefix of the log without the opening `x` after `Lc` that corresponds**: the same
prefix where it ends before `x`, the prefix less `x` where it runs through `x`. -/
def DropPre (Lc : List Release) (x : Release) (L₁ L₁' : List Release) : Prop :=
  L₁' = L₁ ∨ ∃ M, L₁ = Lc ++ x :: M ∧ L₁' = Lc ++ M

/-- **Corresponding prefixes read the same**, where the dropped opening is of an already-opened tree's stream named
the session's: the same trees opened, the same trees sampled, and as many rounds opened of every stream not named
the session's. -/
theorem DropPre.reads {Lc : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    (hi : i ∈ openedTrees Lc)
    (hs : ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) {L₁ L₁' : List Release}
    (h : DropPre Lc x L₁ L₁') :
    openedTrees L₁' = openedTrees L₁ ∧ helloSpecs L₁' = helloSpecs L₁ ∧
      ∀ i' s' name, ((helloSpecs L₁)[i']?.bind fun sp => sp.streams[s']?) = some name →
        (name == TREE_SESSION) = false → openedOf L₁' i' s' = openedOf L₁ i' s' := by
  rcases h with rfl | ⟨M, rfl, rfl⟩
  · exact ⟨rfl, rfl, fun _ _ _ _ _ => rfl⟩
  · refine ⟨(openedTrees_drop hx hi M).symm, (helloSpecs_drop hx Lc M).symm, fun i' s' name hname hnT => ?_⟩
    refine (openedOf_drop hx (fun h => ?_) Lc M).symm
    obtain ⟨rfl, rfl⟩ := h
    have h2 := helloSpecs_bind_append hs (x :: M)
    rw [hname] at h2
    rw [Option.some.inj h2, beq_self_eq_true] at hnT
    cases hnT

/-! ## The releases correspond -/

/-- **A release of the log is the dropped opening or a release of the log without it**, after a prefix that
corresponds. -/
theorem drop_split {Lc L₂ : List Release} {x : Release} {L₁ : List Release} {y : Release} {L₃ : List Release}
    (h : Lc ++ x :: L₂ = L₁ ++ y :: L₃) :
    (L₁ = Lc ∧ y = x) ∨ ∃ L₁' L₃', Lc ++ L₂ = L₁' ++ y :: L₃' ∧ DropPre Lc x L₁ L₁' := by
  rcases List.append_eq_append_iff.1 h with ⟨a, rfl, ha⟩ | ⟨a, rfl, ha⟩
  · cases a with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact Or.inl ⟨List.append_nil _, rfl⟩
    | cons b a =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, ha'⟩ := List.cons.inj ha
      exact Or.inr ⟨Lc ++ a, L₃, by rw [ha', List.append_assoc], Or.inr ⟨a, rfl, rfl⟩⟩
  · cases a with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact Or.inl ⟨(List.append_nil _).symm, rfl⟩
    | cons b a =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact Or.inr ⟨L₁, a ++ L₂, by rw [List.append_assoc, List.cons_append], Or.inl rfl⟩

/-- **A release of the log without the dropped opening is a release of the log**, after a prefix that
corresponds. -/
theorem drop_unsplit {Lc L₂ : List Release} (x : Release) {L₁' : List Release} {y : Release} {L₃' : List Release}
    (h : Lc ++ L₂ = L₁' ++ y :: L₃') :
    ∃ L₁ L₃, Lc ++ x :: L₂ = L₁ ++ y :: L₃ ∧ DropPre Lc x L₁ L₁' := by
  rcases List.append_eq_append_iff.1 h with ⟨a, rfl, ha⟩ | ⟨a, hLc, ha⟩
  · exact ⟨Lc ++ x :: a, L₃', by rw [ha, List.append_assoc, List.cons_append], Or.inr ⟨a, rfl, rfl⟩⟩
  · cases a with
    | nil =>
      rw [List.append_nil] at hLc
      rw [List.nil_append] at ha
      refine ⟨Lc ++ x :: [], L₃', ?_, Or.inr ⟨[], rfl, by rw [List.append_nil, hLc]⟩⟩
      rw [← ha, List.append_assoc, List.cons_append, List.nil_append]
    | cons b a =>
      rw [List.cons_append] at ha
      obtain ⟨rfl, -⟩ := List.cons.inj ha
      exact ⟨L₁', a ++ x :: L₂, by rw [hLc, List.append_assoc, List.cons_append], Or.inl rfl⟩

/-! ## The read, less the closing round, with the opening and without it -/

/-- **The read less the closing round holds of the log where it holds of the log without the opening**: an opening
of an already-opened tree's stream named the session's, put into a log, changes no field of `SessionOpens`, at any
session of the log. -/
theorem SessionOpens.of_drop {Lc L₂ : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    (hi : i ∈ openedTrees Lc) (hs : ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    {tables : Array String} {u : ℕ} {text : String} {J : Json} {sess : Flock.RunRecord}
    (h : SessionOpens (Lc ++ L₂) tables u text J sess) : SessionOpens (Lc ++ x :: L₂) tables u text J sess := by
  obtain ⟨hT, hH, hO⟩ :=
    DropPre.reads hx hi hs (Or.inr ⟨L₂, rfl, rfl⟩ : DropPre Lc x (Lc ++ x :: L₂) (Lc ++ L₂))
  refine ⟨h.parse, h.coins, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L₁ y L₃ i' s' n' he hy hni hlen
    rcases drop_split he with ⟨rfl, rfl⟩ | ⟨L₁', L₃', he', hD⟩
    · rw [hx] at hy
      obtain ⟨rfl, -, -⟩ := Frame.opening.inj hy
      exact absurd hi hni
    · obtain ⟨hT', hH', -⟩ := hD.reads hx hi hs
      rw [← hT'] at hni hlen
      rw [← hH']
      exact h.link L₁' y L₃' i' s' n' he' hy hni hlen
  · intro L₁ y L₃ i' s' n' name he hy hτ hname hnT
    rcases drop_split he with ⟨rfl, rfl⟩ | ⟨L₁', L₃', he', hD⟩
    · rw [hx] at hy
      obtain ⟨rfl, rfl, -⟩ := Frame.opening.inj hy
      rw [hs] at hname
      rw [← Option.some.inj hname, beq_self_eq_true] at hnT
      cases hnT
    · obtain ⟨hT', hH', hO'⟩ := hD.reads hx hi hs
      rw [← hO' i' s' name hname hnT]
      rw [← hT'] at hτ
      rw [← hH'] at hname
      exact h.round L₁' y L₃' i' s' n' name he' hy hτ hname hnT
  · intro i' s' name hτ hname hnT
    rw [← hO i' s' name hname hnT]
    rw [← hT] at hτ
    rw [← hH] at hname
    exact h.full i' s' name hτ hname hnT
  · intro name hpos i' hτ
    rw [← hT] at hτ
    rw [← hH]
    exact h.key name hpos i' hτ
  · intro i' hτ
    rw [← hT] at hτ
    rw [← hH]
    exact h.tree i' hτ
  · intro L₁ y L₃ i' s' n' he hy hni hlen
    rcases drop_split he with ⟨rfl, rfl⟩ | ⟨L₁', L₃', he', hD⟩
    · rw [hx] at hy
      obtain ⟨rfl, -, -⟩ := Frame.opening.inj hy
      exact absurd hi hni
    · obtain ⟨hT', -, -⟩ := hD.reads hx hi hs
      rw [← hT'] at hni hlen
      exact h.linkCount L₁' y L₃' i' s' n' he' hy hni hlen
  · intro L₁ y L₃ i' s' n' name he hy hτ hname hnT
    rcases drop_split he with ⟨rfl, rfl⟩ | ⟨L₁', L₃', he', hD⟩
    · rw [hx] at hy
      obtain ⟨rfl, rfl, -⟩ := Frame.opening.inj hy
      rw [hs] at hname
      rw [← Option.some.inj hname, beq_self_eq_true] at hnT
      cases hnT
    · obtain ⟨hT', hH', hO'⟩ := hD.reads hx hi hs
      rw [← hO' i' s' name hname hnT]
      rw [← hT'] at hτ
      rw [← hH'] at hname
      exact h.roundCount L₁' y L₃' i' s' n' name he' hy hτ hname hnT

/-- **The read less the closing round holds of the log without the opening where it holds of the log**: dropping an
opening of an already-opened tree's stream named the session's changes no field of `SessionOpens`, at any session of
the log. -/
theorem SessionOpens.drop {Lc L₂ : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    (hi : i ∈ openedTrees Lc) (hs : ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    {tables : Array String} {u : ℕ} {text : String} {J : Json} {sess : Flock.RunRecord}
    (h : SessionOpens (Lc ++ x :: L₂) tables u text J sess) : SessionOpens (Lc ++ L₂) tables u text J sess := by
  obtain ⟨hT, hH, hO⟩ :=
    DropPre.reads hx hi hs (Or.inr ⟨L₂, rfl, rfl⟩ : DropPre Lc x (Lc ++ x :: L₂) (Lc ++ L₂))
  refine ⟨h.parse, h.coins, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L₁' y L₃' i' s' n' he hy hni hlen
    obtain ⟨L₁, L₃, he', hD⟩ := drop_unsplit x he
    obtain ⟨hT', hH', -⟩ := hD.reads hx hi hs
    rw [hT'] at hni hlen
    rw [hH']
    exact h.link L₁ y L₃ i' s' n' he' hy hni hlen
  · intro L₁' y L₃' i' s' n' name he hy hτ hname hnT
    obtain ⟨L₁, L₃, he', hD⟩ := drop_unsplit x he
    obtain ⟨hT', hH', hO'⟩ := hD.reads hx hi hs
    rw [hT'] at hτ
    rw [hH'] at hname
    rw [hO' i' s' name hname hnT]
    exact h.round L₁ y L₃ i' s' n' name he' hy hτ hname hnT
  · intro i' s' name hτ hname hnT
    rw [hT] at hτ
    rw [hH] at hname
    rw [hO i' s' name hname hnT]
    exact h.full i' s' name hτ hname hnT
  · intro name hpos i' hτ
    rw [hT] at hτ
    rw [hH]
    exact h.key name hpos i' hτ
  · intro i' hτ
    rw [hT] at hτ
    rw [hH]
    exact h.tree i' hτ
  · intro L₁' y L₃' i' s' n' he hy hni hlen
    obtain ⟨L₁, L₃, he', hD⟩ := drop_unsplit x he
    obtain ⟨hT', -, -⟩ := hD.reads hx hi hs
    rw [hT'] at hni hlen
    exact h.linkCount L₁ y L₃ i' s' n' he' hy hni hlen
  · intro L₁' y L₃' i' s' n' name he hy hτ hname hnT
    obtain ⟨L₁, L₃, he', hD⟩ := drop_unsplit x he
    obtain ⟨hT', hH', hO'⟩ := hD.reads hx hi hs
    rw [hT'] at hτ
    rw [hH'] at hname
    rw [hO' i' s' name hname hnT]
    exact h.roundCount L₁ y L₃ i' s' n' name he' hy hτ hname hnT

/-- **Dropping the closing opening changes no field**: `SessionOpens`, at any session, holds of a log exactly when
it holds of the log without an opening of an already-opened tree's stream named the session's. -/
theorem SessionOpens.drop_iff {Lc L₂ : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    (hi : i ∈ openedTrees Lc) (hs : ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION)
    {tables : Array String} {u : ℕ} {text : String} {J : Json} {sess : Flock.RunRecord} :
    SessionOpens (Lc ++ x :: L₂) tables u text J sess ↔ SessionOpens (Lc ++ L₂) tables u text J sess :=
  ⟨SessionOpens.drop hx hi hs, SessionOpens.of_drop hx hi hs⟩

/-! ## The read with the closing round -/

/-- **A session's read from its read less the closing round on the log without the closing opening**: where the
opening `x` after `Lc` is of the `u`-th tree's stream named the session's, the second opening of that stream,
answers the SHA-512 of the session's evidence `w`, and no opening of that tree follows it, `SessionOpens` at session
`u` on the log without `x` gives `SessionRead` at session `u` and `w` on the log. -/
theorem SessionRead.of_dropped {Lc L₂ : List Release} {x : Release} {i s n : ℕ} (hx : x.frame = .opening i s n)
    {u : ℕ} (hu : (openedTrees Lc)[u]? = some i)
    (hs : ((helloSpecs Lc)[i]?.bind fun sp => sp.streams[s]?) = some TREE_SESSION) (hk : openedOf Lc i s = 1)
    {w : Claims.Msg} (hw : sha512Msg w = x.answers.data.toList)
    (hlast : ∀ y ∈ L₂, ∀ s' n', y.frame ≠ .opening i s' n')
    {tables : Array String} {text : String} {J : Json} {sess : Flock.RunRecord}
    (h : SessionOpens (Lc ++ L₂) tables u text J sess) : SessionRead (Lc ++ x :: L₂) tables u text J sess w := by
  refine ⟨h.of_drop hx (List.mem_of_getElem? hu) hs, fun i' hτ => ?_⟩
  obtain rfl : i = i' := Option.some.inj ((openedTrees_getElem?_append hu (x :: L₂)).symm.trans hτ)
  refine ⟨Lc, x, L₂, s, n, rfl, hx, hu, hs, hk, hw, fun L₁ y L₃ s' n' he hy => ?_⟩
  rcases List.append_eq_append_iff.1 he with ⟨a, rfl, ha⟩ | ⟨a, rfl, -⟩
  · cases a with
    | nil => exact Nat.le_of_eq (congrArg List.length (List.append_nil Lc))
    | cons b a =>
      rw [List.cons_append] at ha
      obtain ⟨-, ha'⟩ := List.cons.inj ha
      exact absurd hy (hlast y (by rw [ha']; exact List.mem_append_right _ List.mem_cons_self) s' n')
  · rw [List.length_append]
    exact Nat.le_add_right _ _

end Verity.Sampled
