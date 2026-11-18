module

public import InferenceOnlyVerifier.RecordWired
public import Properties.InferenceOnly.RecordEdges
public import Mathlib.Tactic

/-!
# The wired reading agrees with `read` where the root has no edge

`RecordWired.readWired` reads the root's edges (`RecordWired.hasEdge`) and otherwise is `Record.read`
(`readWired_eq_read`): its front is `front`'s with no edge, unless `front` refuses `own-rows` on a root with an edge
(`frontWired_eq`), and its finish with no edge is `finish` (`finishWired_nil`). With no edge, the widened property
`OwnOrEdgeRows` is `OwnRows` (`ownOrEdgeRows_nil_iff`).
-/

@[expose] public section

namespace InferenceOnlyVerifier.RecordWired.SecurityProofs

open Lean (Json)
open InferenceOnlyVerifier.Record
open Flock.TemplateQuery (TRange)

/-- With no edge, `finishWired` is `finish`. -/
theorem finishWired_nil (f : Front) (pin : String → Option String) : finishWired f [] pin = finish f pin := by
  unfold finishWired
  cases finish f pin with
  | error e => rfl
  | ok c =>
    cases c with
    | mk =>
      rename_i x _ _ _ _ _ _
      cases x
      simp [bind, Except.bind, pure, Except.pure, Spot.Wires.pairwiseDisjoint]

/-- Where the root the instance decodes to has no edge, `frontWired` is `front` with no edge. -/
theorem frontWired_eq {j tr : Json}
    (h : ∀ P, (Record.parse j >>= fun i => decodeProgram i.program) = .ok P → hasEdge P = false) :
    frontWired j tr = (front j tr).map (·, []) := by
  unfold frontWired
  split
  · rename_i he
    rw [he]
    cases hp : (Record.parse j >>= fun i => decodeProgram i.program) with
    | error e => simp [Except.toOption, Except.map, bind, Except.bind]
    | ok P => simp [Except.toOption, h P hp, Except.map, bind, Except.bind]
  · rfl

/-- **The wired reading agrees with `read`** on every instance whose root has no edge. -/
theorem readWired_eq_read {j tr : Json}
    (h : ∀ P, (Record.parse j >>= fun i => decodeProgram i.program) = .ok P → hasEdge P = false) :
    readWired j tr = read j tr := by
  unfold readWired read
  rw [frontWired_eq h]
  cases front j tr with
  | error e => rfl
  | ok f => simp [Except.map, bind, Except.bind, finishWired_nil]

open InferenceOnlyVerifier.Properties.Record in
/-- **With no edge, `OwnOrEdgeRows` is `OwnRows`.** -/
theorem ownOrEdgeRows_nil_iff {P : Flock.Program.Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} :
    OwnOrEdgeRows P rs W (edges []) ↔ OwnRows P rs W := by
  have hn : ∀ t u p r, (edges []).writerOf t u p r = none := fun _ _ _ _ => rfl
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun r l g lo hi hr => ?_, fun r l lo hi r' l' lo' hi' g ha hb =>
      h2 r l lo hi r' l' lo' hi' g (hn _ _ _ _) (hn _ _ _ _) ha hb⟩
    rcases h1 r l g lo hi hr with ⟨-, h⟩ | ⟨-, wt, wu, out, i, i', he, -⟩
    · exact h
    · rw [hn] at he
      cases he
  · rintro ⟨h1, h2⟩
    exact ⟨fun r l g lo hi hr => Or.inl ⟨hn _ _ _ _, h1 r l g lo hi hr⟩,
      fun r l lo hi r' l' lo' hi' g _ _ ha hb => h2 r l lo hi r' l' lo' hi' g ha hb⟩

end InferenceOnlyVerifier.RecordWired.SecurityProofs
